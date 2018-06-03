import delfi.distribution as dd
import numpy as np
import theano.tensor as tt

from delfi.inference.BaseInference import BaseInference
from delfi.neuralnet.NeuralNet import NeuralNet
from delfi.neuralnet.Trainer import Trainer
from delfi.neuralnet.loss.regularizer import svi_kl_zero

def per_round(y):

    if type(y) == list:
        try:
            y_round = y[r-1]
        except:
            y_round = y[-1]
    else:
        y_round = y

    return y_round

def logdet(M):
    slogdet = np.linalg.slogdet(M)
    return slogdet[0] * slogdet[1]

class CDELFI(BaseInference):
    def __init__(self, generator, obs, reg_lambda=0.01, 
                 **kwargs):
        """Conditional density estimation likelihood-free inference (CDE-LFI)

        Implementation of algorithms 1 and 2 of Papamakarios and Murray, 2016.

        Parameters
        ----------
        generator : generator instance
            Generator instance
        obs : array
            Observation in the format the generator returns (1 x n_summary)
        prior_norm : bool
            If set to True, will z-transform params based on mean/std of prior
        pilot_samples : None or int
            If an integer is provided, a pilot run with the given number of
            samples is run. The mean and std of the summary statistics of the
            pilot samples will be subsequently used to z-transform summary
            statistics.
        reg_lambda : float
            Precision parameter for weight regularizer if svi is True
        seed : int or None
            If provided, random number generator will be seeded
        verbose : bool
            Controls whether or not progressbars are shown
        kwargs : additional keyword arguments
            Additional arguments for the NeuralNet instance, including:
                n_hiddens : list of ints
                    Number of hidden units per layer of the neural network
                svi : bool
                    Whether to use SVI version of the network or not

        Attributes
        ----------
        observables : dict
            Dictionary containing theano variables that can be monitored while
            training the neural network.
        """
        # Algorithm 1 of PM requires a single component
        if 'n_components' in kwargs:
            assert kwargs['n_components'] == 1 # moved n_components argument to run()
        super().__init__(generator, **kwargs)

        self.obs = obs
        if np.any(np.isnan(self.obs)):
            raise ValueError("Observed data contains NaNs")

        self.reg_lambda = reg_lambda
        self.round = 0 # total round counter

    def loss(self, N):
        """Loss function for training

        Parameters
        ----------
        N : int
            Number of training samples
        """
        loss = -tt.mean(self.network.lprobs)

        if self.svi:

            if self.reg_lambda > 0:
                kl, imvs = svi_kl_zero(self.network.mps, self.network.sps,
                                       self.reg_lambda)
            else:
                kl, imvs = 0, {}

            loss = loss + 1 / N * kl

            # adding nodes to dict s.t. they can be monitored
            self.observables['loss.kl'] = kl
            self.observables.update(imvs)

        return loss

    def run(self, n_train=100, n_rounds=2, epochs=100, minibatch=50,
            monitor=None, n_components=1, **kwargs):
        """Run algorithm

        Parameters
        ----------
        n_train : int or list of ints
            Number of data points drawn per round. If a list is passed, the
            nth list element specifies the number of training examples in the
            nth round. If there are fewer list elements than rounds, the last
            list element is used.
        n_rounds : int
            Number of rounds
        epochs: int
            Number of epochs used for neural network training
        minibatch: int
            Size of the minibatches used for neural network training
        monitor : list of str
            Names of variables to record during training along with the value
            of the loss function. The observables attribute contains all
            possible variables that can be monitored
        n_components : int
            Number of components in final round (if > 1, gives PM's algorithm 2)
        kwargs : additional keyword arguments
            Additional arguments for the Trainer instance

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged while training the networks
        trn_datasets : list of (params, stats)
            training datasets, z-transformed
        posteriors : list of posteriors
            posterior after each round
        """
        logs = []
        trn_datasets = []
        posteriors = []

        #assert self.kwargs['n_components'] == 1 
        # could also allow to go back to single Gaussian via project_to_gaussian()

        for r in range(1, n_rounds + 1):  # start at 1

            self.round += 1

            if self.round > 1:
                # posterior becomes new proposal prior
                if self.round==2 or isinstance(self.generator.proposal, (dd.Uniform,dd.Gaussian)):  
                    proposal = self.predict(self.obs)
                elif len(self.generator.proposal.xs) == n_components:                    
                    print('correcting for MoG proposal')
                    proposal = self.predict_from_MoG_prop(self.obs)
                else:
                    raise NotImplementedError

                if isinstance(proposal, dd.MoG) and len(proposal.xs) == 1:  
                    proposal = proposal.project_to_gaussian()               
                self.generator.proposal = proposal 

            # number of training examples for this round
            epochs_round = per_round(epochs)
            n_train_round = per_round(n_train)

            # draw training data (z-transformed params and stats)
            verbose = '(round {}) '.format(self.round) if self.verbose else False
            trn_data = self.gen(n_train_round, verbose=verbose)[:2]

            if r == n_rounds: 
                self.kwargs.update({'n_components': n_components})
                self.split_components()

            if r > 1:
                self.reinit_network() # reinits network if flag is set


            if hasattr(self.network, 'extra_stats'):
                trn_inputs = [self.network.params, self.network.stats, self.network.extra_stats]
            else:
                trn_inputs = [self.network.params, self.network.stats]

            t = Trainer(self.network, self.loss(N=n_train_round),
                        trn_data=trn_data, trn_inputs=trn_inputs,
                        monitor=self.monitor_dict_from_names(monitor),
                        seed=self.gen_newseed(), **kwargs)
            logs.append(t.train(epochs=epochs_round, minibatch=minibatch,
                                verbose=verbose,n_inputs=self.network.n_inputs,
                                n_inputs_hidden=self.network.n_inputs_hidden))
            trn_datasets.append(trn_data)

            if self.round==1 or isinstance(self.generator.proposal, (dd.Uniform,dd.Gaussian)):  
                posterior = self.predict(self.obs)
            elif len(self.generator.proposal.xs) == n_components:                    
                print('correcting for MoG proposal')
                posterior = self.predict_from_MoG_prop(self.obs)
            else:
                raise NotImplementedError

            posteriors.append(posterior)

            #except:
            #    posteriors.append(None)
            #    print('analytic correction for proposal seemingly failed!')
            #    break

        return logs, trn_datasets, posteriors

    def predict(self, x, threshold=0.01):
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        threshold: float
            Threshold for pruning MoG components (percent of posterior mass)
        """
        if self.generator.proposal is None:
            # no correction necessary
            return super(CDELFI, self).predict(x)  # via super
        else:
            # mog is posterior given proposal prior
            mog = super(CDELFI, self).predict(x)  # via super

            mog.prune_negligible_components(threshold=threshold)

            # compute posterior given prior by analytical division step
            if 'Uniform' in str(type(self.generator.prior)):
                posterior = mog / self.generator.proposal
            elif 'Gaussian' in str(type(self.generator.prior)):
                posterior = (mog * self.generator.prior) / \
                    self.generator.proposal
            else:
                raise NotImplemented

            return posterior

    def predict_from_MoG_prop(self, x, threshold=0.01):
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        threshold: float
            Threshold for pruning MoG components (percent of posterior mass)
        """
        # mog is posterior given proposal prior
        mog = super(CDELFI, self).predict(x)  # via super
        mog.prune_negligible_components(threshold=threshold)

        prop  = self.generator.proposal
        prior = self.generator.prior
        ldetP0, d0  = logdet(prior.P), prior.m.dot(prior.Pm)
        means = np.vstack([c.m for c in prop.xs])

        xs_new, a_new = [], []
        for c in mog.xs:

            dists = np.sum( (means - np.atleast_2d(c.m))**2, axis=1)
            i = np.argmin(dists)

            c_prop = prop.xs[i]
            a_prop = prop.a[i]

            c_post = (c * prior) / c_prop

            # prefactors
            log_a = np.log(mog.a[i]) - np.log(a_prop) 
            # determinants
            log_a += 0.5 * (logdet(c.P)+ldetP0-logdet(c_prop.P)-logdet(c_post.P))
            # Mahalanobis distances
            log_a -= 0.5 * c.m.dot(c.Pm)
            log_a -= 0.5 * d0
            log_a += 0.5 * c_prop.m.dot(c_prop.Pm)
            log_a += 0.5 * c_post.m.dot(c_post.Pm)
            
            a_i = np.exp(log_a)
            
            xs_new.append(c_post)
            a_new.append(a_i)

        a_new = np.array(a_new)
        a_new /= a_new.sum()

        return dd.MoG( xs = xs_new, a = a_new )


    def predict_uncorrected(self, x):
            """Predict posterior given x under proposal prior

            Predicts the uncorrected posterior associated with the proposal
            prior (versus the original prior). 

            Allows to obtain some posterior estimates when the analytical 
            correction for the proposal prior fails. 

            Parameters
            ----------
            x : array
                Stats for which to compute the posterior
            """

            return super(CDELFI, self).predict(x)  # via super


    def split_components(self, split_mode=None):

        if self.network.n_components == 1 and self.kwargs['n_components'] > 1:

            if split_mode is None:
                # get parameters of current network
                old_params = self.network.params_dict.copy()

                # create new network
                self.network = NeuralNet(**self.kwargs)
                new_params = self.network.params_dict

                # set weights of new network
                # weights of additional components are duplicates
                for p in [s for s in new_params if 'means' in s or
                          'precisions' in s]:

                    print('- perturbing mode')
                    old_params[p] = old_params[p[:-1] + '0'].copy()
                    old_params[p] += 1.0e-2*self.rng.randn(*new_params[p].shape)

                old_params['weights.mW'] = 0. * new_params['weights.mW']
                old_params['weights.mb'] = 0. * new_params['weights.mb']

                self.network.params_dict = old_params


            elif split_mode=='spread_out':
                pass

            else:
                raise NotImplementedError
