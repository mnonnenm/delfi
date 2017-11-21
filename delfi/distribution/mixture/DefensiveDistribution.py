import numpy as np
import scipy.misc
import scipy.stats

from delfi.distribution.Gaussian import Gaussian
from delfi.distribution.mixture.BaseMixture import BaseMixture


class DefensiveDistribution(BaseMixture):
    def __init__(
            self,
            a,
            xs,
            seed=None):
        """Mixture of two arbitrary distributions

    Creates a mixture distribution with a an already given list of distributions.

    Parameters
    ----------
    a : list or np.array, 1d
        Mixing coefficients
    xs : list, length n_components
        List of distributions
    seed : int or None
        If provided, random number generator will be seeded
    """

        super().__init__(
            a=np.asarray(a),
            ncomp=len(xs),
            ndim=xs[0].ndim,
            seed=seed)
        self.xs = xs


    @property
    def mean(self):
        """Means"""
        ms = [x.m for x in self.xs]
        return np.dot(self.a, np.array(ms))

    @property
    def std(self):
        """Standard deviations of marginals"""
        stds = [(x.m-self.mean)**2 + x.std**2 for x in self.xs]
        return np.sqrt(np.dot(self.a, np.array(stds)))


    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
        # See BaseMixture.py for docstring
        ps = np.array([c.eval(x, ii, log) for c in self.xs]).T
        res = scipy.misc.logsumexp(
            ps +
            np.log(
                self.a),
            axis=1) if log else np.dot(
            ps,
            self.a)

        return res

    @copy_ancestor_docstring
    def gen(self, n_samples=1):
        # See BaseMixture.py for docstring
        ii = self.gen_comp(n_samples)  # n_samples,

        ns = [np.sum((ii == i).astype(int)) for i in range(self.n_components)]
        samples = [x.gen(n) for x, n in zip(self.xs, ns)]
        samples = np.concatenate(samples, axis=0)
        self.rng.shuffle(samples)

        return samples