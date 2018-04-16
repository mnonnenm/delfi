import numpy as np

from delfi.distribution.Gaussian import Gaussian


class TruncatedGaussian(Gaussian):

    def __init__(self, m=None, P=None, U=None, S=None, Pm=None, 
                 upper=None, lower=None, seed=None):
        """Truncated Gaussian distribution

        	Quick implementation of truncated Gaussians meant only for use
        	as proposals. 
        	It rests on rejection sampling as implemented in the DELFI
        	Default() generator, but otherwise behaves as a Gaussian object. 

        """

        super().__init__(m=m, P=P, U=U, S=S, Pm=Pm, seed=seed)	        

        self.lower = np.atleast_1d(lower)
        self.upper = np.atleast_1d(upper)

        assert self.lower.ndim == self.upper.ndim
        assert self.lower.ndim == 1