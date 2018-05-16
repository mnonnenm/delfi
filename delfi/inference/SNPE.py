import numpy as np
import theano
import theano.tensor as tt

from delfi.inference.BaseInference import BaseInference
from delfi.neuralnet.Trainer import Trainer
from delfi.neuralnet.loss.regularizer import svi_kl_init, svi_kl_zero
from delfi.kernel.Kernel_learning import kernel_opt

import lasagne.layers as ll

dtype = theano.config.floatX


class SNPE(BaseInference):
    def __init__(self, generator, obs, prior_norm=False, init_norm=False,
                 pilot_samples=100, convert_to_T=3, 
                 reg_lambda=0.01, prior_mixin=0., seed=None, verbose=True,
                 reinit_weights=False, **kwargs):
        """Sequential neural posterior estimation (SNPE)

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
        convert_to_T : None or int
            Convert proposal distribution to Student's T? If a number if given,
            the number specifies the degrees of freedom. None for no conversion
        reg_lambda : float
            Precision parameter for weight regularizer if svi is True
        prior_mixin : float
            Percentage of the prior mixed into the proposal prior. While training,
            an additional prior_mixin * N samples will be drawn from the actual prior
            in each round            
        seed : int or None
            If provided, random number generator will be seeded
        verbose : bool
            Controls whether or not progressbars are shown
        kwargs : additional keyword arguments
            Additional arguments for the NeuralNet instance, including:
                n_components : int
                    Number of components of the mixture density
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
        super().__init__(generator, prior_norm=prior_norm,
                         pilot_samples=pilot_samples, seed=seed,
                         verbose=verbose, **kwargs)

        self.obs = obs

        # optional: z-transform output for obs (also re-centres x onto obs!)
        self.init_norm = init_norm
        self.init_fcv = 0.8 if self.network.n_components > 1 else 0.
        if self.init_norm:
            print('standardizing network initialization')
            self.standardize_init(fcv = self.init_fcv)

        self.reg_lambda = reg_lambda
        self.round = 0
        self.convert_to_T = convert_to_T

        self.prior_mixin = 0. if prior_mixin is None else prior_mixin

        self.reinit_weights = reinit_weights

        # placeholder for importance weights
        self.network.iws = tt.vector('iws', dtype=dtype)

    def loss(self, N, round_cl=1):
        """Loss function for training

        Parameters
        ----------
        N : int
            Number of training samples
        """
        loss = -tt.mean(self.network.iws * self.network.lprobs)

        # adding nodes to dict s.t. they can be monitored during training
        self.observables['loss.lprobs'] = self.network.lprobs
        self.observables['loss.iws'] = self.network.iws
        self.observables['loss.raw_loss'] = loss

        if self.svi:
            if self.round <= round_cl:
                # weights close to zero-centered prior in the first round
                if self.reg_lambda > 0:
                    kl, imvs = svi_kl_zero(self.network.mps, self.network.sps,
                                           self.reg_lambda)
                else:
                    kl, imvs = 0, {}
            else:
                # weights close to those of previous round
                kl, imvs = svi_kl_init(self.network.mps, self.network.sps)

            loss = loss + 1 / N * kl

            # adding nodes to dict s.t. they can be monitored
            self.observables['loss.kl'] = kl
            self.observables.update(imvs)

        return loss

    def run(self, n_train=100, n_rounds=2, epochs=100, minibatch=50, clip_IW=None,
            round_cl=1, stop_on_nan=False, monitor=None, kernel_loss=None, 
            epochs_cbk=None, cbk_feature_layer=0, minibatch_cbk=None, reg_lambdas=None,
            init_single_layer_net=False, naive_weights=False,
            **kwargs):
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
        epochs : int
            Number of epochs used for neural network training
        minibatch : int
            Size of the minibatches used for neural network training
        monitor : list of str
            Names of variables to record during training along with the value
            of the loss function. The observables attribute contains all
            possible variables that can be monitored
        round_cl : int
            Round after which to start continual learning
        stop_on_nan : bool
            If True, will halt if NaNs in the loss are encountered
        kwargs : additional keyword arguments
            Additional arguments for the Trainer instance

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged while training the networks
        trn_datasets : list of (params, stats)
            training datasets, z-transformed
        posteriors : list of distributions
            posterior after each round
        """
        logs = []
        trn_datasets = []
        posteriors = []

        minibatch_cbk = minibatch if minibatch_cbk is None else minibatch_cbk

        assert (clip_IW is None) or (clip_IW >= 0. and clip_IW <= 1.) 
        for r in range(n_rounds):
            self.round += 1

            # if round > 1, set new proposal distribution before sampling
            if self.round > 1:
                # posterior becomes new proposal prior
                proposal = self.predict(self.obs)  # see super

                # convert proposal to student's T?
                if self.convert_to_T is not None:
                    if type(self.convert_to_T) == int:
                        dofs = self.convert_to_T
                    else:
                        dofs = 10
                    proposal = proposal.convert_to_T(dofs=dofs)

                self.generator.proposal = proposal

            if self.round > 1 and self.reinit_weights:
                print('re-initializing network weights')
                self.reinit_network()                
                if self.init_norm:
                    print('standardizing network initialization')
                    self.standardize_init(fcv = self.init_fcv)

            # number of training examples for this round
            if type(n_train) == list:
                try:
                    n_train_round = n_train[self.round-1]
                except:
                    n_train_round = n_train[-1]
            else:
                n_train_round = n_train

            if type(epochs) == list:
                try:
                    epochs_round = epochs[self.round-1]
                except:
                    epochs_round = epochs[-1]
            else:
                epochs_round = epochs

            epochs_cbk_round = epochs_round if epochs_cbk is None else epochs_cbk

            # draw training data (z-transformed params and stats)
            verbose = '(round {}) '.format(self.round) if self.verbose else False
            trn_data = self.gen(n_train_round, prior_mixin=self.prior_mixin, verbose=verbose)
            n_train_round = trn_data[0].shape[0]

            # precompute importance weights
            iws = np.ones((n_train_round,))


            print('sources: ', np.unique(trn_data[2]))

            cbkrnl, cbl = None, None
            if self.generator.proposal is not None:
                params = self.params_std * trn_data[0] + self.params_mean
                p_prior = self.generator.prior.eval(params, log=False)
                p_proposal = self.generator.proposal.eval(params, log=False)
                iws *= p_prior / (self.prior_mixin * p_prior + (1. - self.prior_mixin) * p_proposal)

                # train calibration kernel (learns own normalization)
                if not kernel_loss is None:
                    if verbose:
                        print('fitting calibration kernel ...')

                    ks = list(self.network.layer.keys())
                    #hiddens = np.where([i[:6]=='hidden' for i in ks])[0]
                    #cbk_feature_layer = hiddens[-1] # pick last hidden layer
                    hl = self.network.layer[ks[cbk_feature_layer]]

                    stat_features = theano.function(
                        inputs=[self.network.stats],
                        outputs=ll.get_output(hl))

                    idx_proposal = np.where(trn_data[2])[0]
                    minibatch_cbk = np.min((minibatch_cbk, idx_proposal.size))

                    fstats = stat_features(trn_data[1][idx_proposal,:]).reshape(idx_proposal.size,-1)
                    obs_z = (self.obs - self.stats_mean) / self.stats_std
                    fobs_z = stat_features(obs_z).reshape(1,-1)

                    cbkrnl, cbl = kernel_opt(
                        iws=iws[idx_proposal].astype(np.float32), 
                        stats=fstats,
                        obs=fobs_z, 
                        kernel_loss=kernel_loss, 
                        epochs=epochs_cbk_round, #epochs 
                        minibatch=minibatch_cbk, #minibatch, 
                        stop_on_nan=stop_on_nan,
                        seed=self.gen_newseed(), 
                        monitor=self.monitor_dict_from_names(monitor),
                        **kwargs)
                    if verbose: 
                        print('done.')
                        
                    fstats = stat_features(trn_data[1]).reshape(n_train_round,-1)
                    iws *= cbkrnl.eval(fstats)

            # normalize weights
            iws = (iws/np.sum(iws))*n_train_round

            if not clip_IW is None:
                idx = np.argsort(iws)[int((1-clip_IW)*n_train_round):]
                iws[idx] = 0.

            if naive_weights:
                iws = np.ones((n_train_round,))

            trn_data = (trn_data[0], trn_data[1], iws)
            trn_inputs = [self.network.params, self.network.stats, self.network.extra_stats,
                          self.network.iws]

            if init_single_layer_net:
                print('initializing network from homoscedastic linear-affine fit')
                self.init_single_layer_net(trn_data, self.obs)                          

            if not reg_lambdas is None:
                self.reg_lambda = reg_lambdas[self.round-1]
                print('resetting regularization strength to ' + str(self.reg_lambda))

            t = Trainer(self.network,
                        self.loss(N=n_train_round, round_cl=round_cl),
                        trn_data=trn_data, trn_inputs=trn_inputs,
                        seed=self.gen_newseed(),
                        monitor=self.monitor_dict_from_names(monitor),
                        **kwargs)
            logs.append(t.train(epochs=epochs_round, minibatch=minibatch,
                                verbose=verbose, stop_on_nan=stop_on_nan))

            logs[-1]['cbkrnl'] = cbkrnl
            logs[-1]['cbk_loss'] = cbl

            trn_datasets.append(trn_data)

            try:
                posteriors.append(self.predict(self.obs))
            except:
                posteriors.append(None)
                print('analytic correction for proposal seemingly failed!')
                break

        return logs, trn_datasets, posteriors
