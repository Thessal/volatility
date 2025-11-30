
# Preaveraging
# Frequency cut
# Fracdiff
import numpy as np 

class Denoiser:
    # Jump adjust / Heavy tail truncation
    # Detrend
    # Windowing
    # --> (Microstructure) noise removal
    def __init__(self, logprc):
        self.logprc = logprc
        self.logret = np.diff(self.logprc)
        self.logret -= np.mean(self.logret) # detrend
        self.logprc = np.cumsum(self.logret)
        assert self.logprc.shape == self.logret.shape

class Truncate(Denoiser):
    def pred(self, thres= ):
        # huber truncation with 2 sigma
        huber = 
        np.std(self.logret)