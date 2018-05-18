import abc
import numpy as np

from delfi.distribution.Discrete import Discrete
from delfi.utils.meta import ABCMetaDoc


class BaseMixture(metaclass=ABCMetaDoc):
    """Abstract base class for mixture distributions

    Distributions must at least implement abstract methods of this class.

    Component distributions should be added to self.xs, which is a list
    containing the distributions of individual components.

    Parameters
    ----------
    a : list or np.array, 1d
        Mixing coefficients
    ncomp : int
        Number of components
    ndim : int
        Number of ndimensions of the component distributions
    seed : int or None
        If provided, random number generator will be seeded
    """
    def __init__(self, a, ncomp, ndim, seed=None):
        self.a = np.asarray(a)
        self.ncomp = ncomp
        self.ndim = ndim

        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

        self.discrete_sample = Discrete(p=self.a, seed=self.gen_newseed())

    @abc.abstractmethod
    def eval(self, x, ii=None, log=True):
        """Method to evaluate pdf

        Parameters
        ----------
        x : int or list or np.array
            Rows are inputs to evaluate at
        ii : list
            A list of indices specifying which marginal to evaluate.
            If None, the joint pdf is evaluated
        log : bool, defaulting to True
            If True, the log pdf is evaluated

        Returns
        -------
        scalar
        """
        pass

    @abc.abstractmethod
    def gen(self, n_samples=1):
        """Method to generate samples

        Parameters
        ----------
        n_samples : int
            Number of samples to generate

        Returns
        -------
        n_samples x self.ndim
        """
        pass

    @property
    def n_components(self):
        return self.ncomp

    def gen_comp(self, n_samples):
        """Generate component index according to self.a"""
        return self.discrete_sample.gen(n_samples).reshape(-1)  # n_samples,

    def gen_newseed(self):
        """Generates a new random seed"""
        if self.seed is None:
            return None
        else:
            return self.rng.randint(0, 2**31)

    def kl(self, other, n_samples=10000):
        """Estimates the KL from this to another PDF

        KL(this | other), using Monte Carlo"""
        x = self.gen(n_samples)
        lp = self.eval(x, log=True)
        lq = other.eval(x, log=True)
        t = lp - lq

        res = np.mean(t)
        err = np.std(t, ddof=1) / np.sqrt(n_samples)

        return res, err

    def prune_negligible_components(self, threshold):
        """Prune components

        Removes all the components whose mixing coefficient is less
        than a threshold.
        """
        ii = np.nonzero((self.a < threshold).astype(int))[0]
        total_del_a = np.sum(self.a[ii])
        del_count = ii.size

        self.ncomp -= del_count
        self.a = np.delete(self.a, ii)
        self.a += total_del_a / self.n_components
        self.xs = [x for i, x in enumerate(self.xs) if i not in ii]
