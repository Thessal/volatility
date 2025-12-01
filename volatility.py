import numpy as np
from typing import Dict, List
from informationmeasure import Filtration
import scipy
import statsmodels.tsa.stattools as st


class VarianceEstimation:
    # Returns estimated variance

    def __init__(self, logprc, logprc_denoise, K = 60 * 10):
        self.logprc = logprc
        self.logprc_denoise = logprc_denoise

        self.n = len(self.logprc)
        self.K = K 
    
    def rv(self):
        sq = np.square(np.diff(self.logprc))
        kernel = np.ones(self.K) / self.K
        return np.convolve(sq, kernel, mode='valid')

    def tsrv(self):
        # Two-scale Realized variance
        logprc = self.logprc.copy()
        x1 = np.square(logprc[:-K] -
                       logprc[K:]).mean()
        x2 = np.square(np.diff(logprc)).mean()
        tsrv = x1 - x2
        return tsrv

    def prv(self):
        sq = np.square(np.diff(self.logprc_denoise))
        kernel = np.ones(self.K) / self.K
        rv = np.convolve(sq, kernel, mode='valid')

        window = self.K
        kernel = np.minimum(np.linspace(0, 1, window+2),
                            np.linspace(1, 0, window+2))[1:-1]
        kernel /= np.sqrt((np.square(kernel).sum())) 
        noise_adjustment = np.square(logret).mean() * 0.5 * (np.square(np.diff(kernel)).sum())

        return rv - noise_adjustment
