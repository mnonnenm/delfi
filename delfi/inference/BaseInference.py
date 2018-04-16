import abc
import numpy as np

from delfi.neuralnet.NeuralNet import NeuralNet
from delfi.neuralnet.Trainer import Trainer
from delfi.utils.meta import ABCMetaDoc

import theano
dtype = theano.config.floatX

class BaseInference(metaclass=ABCMetaDoc):
    def __init__(self, generator, 
                 prior_norm=True,  
                 pilot_samples=100,
                 seed=None, verbose=True, **kwargs):
        """Abstract base class for inference algorithms

        Inference algorithms must at least implement abstract methods of this
        class.

        Parameters
        ----------
        generator : generator instance
            Generator instance
        prior_norm : bool
            If set to True, will z-transform params based on mean/std of prior
        pilot_samples : None or int
            If an integer is provided, a pilot run with the given number of
            samples is run. The mean and std of the summary statistics of the
            pilot samples will be subsequently used to z-transform summary
            statistics.
        seed : int or None
            If provided, random number generator will be seeded
        kwargs : additional keyword arguments
            Additional arguments used when creating the NeuralNet instance

        Attributes
        ----------
        observables : dict
            Dictionary containing theano variables that can be monitored while
            training the neural network.
        """
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()
        self.verbose = verbose

        # bind generator, reset proposal attribute
        self.generator = generator
        self.generator.proposal = None

        # generate a sample to get input and output dimensions
        params, stats = generator.gen(1, skip_feedback=True, verbose=False)
        kwargs.update({'n_inputs': stats.shape[1:],
                       'n_outputs': params.shape[1],
                       'seed': self.gen_newseed()})

        self.network = NeuralNet(**kwargs)
        self.svi = self.network.svi
        self.kwargs = kwargs

        # parameters for z-transform of params
        if prior_norm:
            # z-transform for params based on prior
            self.params_mean = self.generator.prior.mean
            self.params_std = self.generator.prior.std
        else:
            # parameters are set such that z-transform has no effect
            self.params_mean = np.zeros((params.shape[1],))
            self.params_std = np.ones((params.shape[1],))

        # parameters for z-transform for stats
        if pilot_samples is not None and pilot_samples != 0:
            # determine via pilot run
            self.pilot_run(pilot_samples)
        else:
            # parameters are set such that z-transform has no effect
            self.stats_mean = np.zeros((stats.shape[1],))
            self.stats_std = np.ones((stats.shape[1],))

        # observables contains vars that can be monitored during training
        self.compile_observables()

    @abc.abstractmethod
    def loss(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    def reinit_network(self):
        """Reinitializes the network instance (re-setting the weights!) 

        """
        self.network = NeuralNet(**self.kwargs)
        self.svi = self.network.svi

    def centre_on_obs(self):
        """ Centres first-layer input onto observed summary statistics 

        Ensures x' = x - xo, i.e. first-layer input x' = 0 for x = xo.
        """

        self.stats_mean = self.obs.flatten()

    def remove_hidden_biases(self):
        """ Resets all bias weights in hidden layers to zero.

        """
        def idx_hiddens(x):
            return x.name[0]=='h'

        for b in filter(idx_hiddens, self.network.mps_bp):
            b.set_value(np.zeros_like(b.get_value()))

    def conditional_norm(self, fcv = 0.8):
        """ Normalizes current network output at obersved summary statistics


        Parameters
        ----------
        fcv : float
            Fraction of total that comes from uncertainty over components, i.e.
            Var[th] = E[Var[th|z]] + Var[E[th|z]]
                    =  (1-fcv)     +     fcv       = 1
        """
        # avoiding CDELFI.predict() attempt to analytically correct for proposal
        obz = (self.obs - self.stats_mean) / self.stats_std
        posterior = self.network.get_mog(obz, deterministic=True)
        mog =  posterior.ztrans_inv(self.params_mean, self.params_std)

        assert np.all(np.diff(mog.a)==0.) # assumes uniform alpha

        n_dim = self.kwargs['n_outputs']
        triu_mask = np.triu(np.ones([n_dim, n_dim], dtype=dtype), 1)
        diag_mask = np.eye(n_dim, dtype=dtype)

        mu, Sig = np.zeros_like(mog.xs[0].m), np.zeros_like(mog.xs[0].S)
        for i in range(self.network.n_components):
            Sig += mog.a[i] * mog.xs[i].S
            mu  += mog.a[i] * mog.xs[i].m
        C = np.zeros_like(Sig)
        for i in range(self.network.n_components):
            dmu = mog.xs[i].m - mu if self.network.n_components > 1 else mog.xs[i].m
            C   += mog.a[i] * np.outer(dmu, dmu)

        Z1inv = np.sqrt((1.-fcv) / np.diag(Sig)).reshape(-1)
        Z2inv = np.sqrt(  fcv    / np.diag( C )).reshape(-1)

        def idx_MoG(x):
            return x.name[:5]=='means'

        # first we need the center of means
        mu_ = np.zeros_like(mog.xs[0].m)
        for b in filter(idx_MoG, self.network.mps_bp):
            mu_ += b.get_value()
        mu_ /= self.network.n_components

        # center and normalize means
        for b in filter(idx_MoG, self.network.mps_bp):
            b.set_value(Z2inv * (b.get_value() - mu_))

        # normalize covariances
        def idx_MoG(x):
            return x.name[:10]=='precisions'
        for b in filter(idx_MoG, self.network.mps_bp):
            val = b.get_value().copy()
            val = val.reshape(n_dim,n_dim)
            val = diag_mask * (val - np.diag(np.log(Z1inv))) + triu_mask * val.dot(np.diag(1./Z1inv))
            b.set_value(val.flatten())


    def standardize_init(self, fcv = 0.8):
        """ Standardizes the network initialization on obs

        Ensures output distributions for xo have mean zero and unit variance.
        Alters hidden layers to propagates x=xo as zero to the last layer, and
        alters the MoG layers to produce the desired output distribution. 
        """

        # ensure x' = x - xo
        self.centre_on_obs()

        # ensure x' = 0 stays zero up to MoG layer (setting biases to zero)
        self.remove_hidden_biases()

        # ensure MoG returns standardized output on x' = 0
        self.conditional_norm(fcv)


    def init_single_layer_net(self, trn_data, obs_stats):
        """ Initializes network with zero hidden layers.

        Without hidden layers, posterior means are linear functions Ax+b,
        and posterior precisions are exp(Cx + d)**2.

        We can initialize A,b,C,d from a homoscedastic linear fit assuming
        theta = f(x) = Ax + b + eps, where eps ~ N(0, Sig)
        and Sig = exp(d)**2, C = 0.
        We assume diagonal noise covariance Sig. 

        """
        assert self.network.n_components == 1
        assert self.network.diag_cov
        assert np.all(obs_stats==self.stats_mean) # assumes self.centre_on_obs()

        ndim, nstats = self.params_mean.size, self.stats_mean.size
        th, x, w = trn_data
        w = w.reshape(-1, 1)
        wth =  w * th

        # solve means
        X = np.hstack((np.ones((th.shape[0], 1)), x))
        ndim, nstats = 3, 13
        beta = np.linalg.solve( X.T.dot(w * X), X.T.dot(wth))
        A, b = beta[1:,:], beta[0,:]

        # solve variances
        Sig = (th.T.dot(wth) - X.dot(beta).T.dot(wth))/th.shape[0]

        C = np.zeros((nstats, ndim**2))
        d = - np.diag(np.log(np.sqrt(np.diag(Sig)))).reshape(-1)

        aps = self.network.aps
        names = np.array([aps[i].name for i in range(len(aps))])

        self.network.aps[np.where(names=='means.mW0')[0][0]].set_value(A)
        self.network.aps[np.where(names=='means.mb0')[0][0]].set_value(b)
        self.network.aps[np.where(names=='precisions.mW0')[0][0]].set_value(C)
        self.network.aps[np.where(names=='precisions.mb0')[0][0]].set_value(d)


    def gen(self, n_samples, n_reps=1, prior_mixin=0, verbose=None):
        """Generate from generator and z-transform

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        n_reps : int
            Number of repeats per parameter
        verbose : None or bool or str
            If None is passed, will default to self.verbose
        """
        verbose = self.verbose if verbose is None else verbose
        params, stats = self.generator.gen(n_samples, prior_mixin=prior_mixin, verbose=verbose)

        # z-transform params and stats
        params = (params - self.params_mean) / self.params_std
        stats = (stats - self.stats_mean) / self.stats_std

        return params, stats

    def gen_newseed(self):
        """Generates a new random seed"""
        if self.seed is None:
            return None
        else:
            return self.rng.randint(0, 2**31)

    def pilot_run(self, n_samples):
        """Pilot run in order to find parameters for z-scoring stats
        """
        verbose = '(pilot run) ' if self.verbose else False
        params, stats = self.generator.gen(n_samples, verbose=verbose)
        self.stats_mean = np.nanmean(stats, axis=0)
        self.stats_std = np.nanstd(stats, axis=0)

    def predict(self, x, deterministic=True):
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        deterministic : bool
            if True, mean weights are used for Bayesian network
        """
        x_zt = (x - self.stats_mean) / self.stats_std
        posterior = self.network.get_mog(x_zt, deterministic=deterministic)
        return posterior.ztrans_inv(self.params_mean, self.params_std)

    def compile_observables(self):
        """Creates observables dict"""
        self.observables = {}
        self.observables['loss.lprobs'] = self.network.lprobs
        for p in self.network.aps:
            self.observables[str(p)] = p

    def monitor_dict_from_names(self, monitor=None):
        """Generate monitor dict from list of variable names"""
        if monitor is not None:
            observe = {}
            if isinstance(monitor, str):
                monitor = [monitor]
            for m in monitor:
                if m in self.observables:
                    observe[m] = self.observables[m]
        else:
            observe = None
        return observe
