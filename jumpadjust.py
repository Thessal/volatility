import numpy as np 

class JumpAdjust:
    # Jump adjust / Heavy tail truncation
    # Returns jump adjusted log-price and the estimated jump size

    def __init__(self, logprc, M = 60*60*24):
        self.logprc = logprc
        self.logret = np.diff(self.logprc)
        self.logret -= np.mean(self.logret) # detrend
        self.logprc = np.cumsum(self.logret)
        assert self.logprc.shape == self.logret.shape

        self.n = len(self.logprc)
        self.M = M
    
    def _tau(self):
        M = self.M
        sq_cumsum = np.cumsum(np.square(self.logret))
        sq_sum = sq_cumsum[M:] - sq_cumsum[:-M]
        sq_sum = np.pad(sq_sum, (M,0), mode='edge')
        tau = np.sqrt(np.sum)
        return tau

    def truncate(self):
        # Truncation at 1sigma
        tau = self._tau()
        truncated = np.sign(self.logret) * np.minimum(np.abs(self.logret), tau)
        assert truncated.shape == self.logret.shape
        return truncated.cumsum(), tau
    
    def truncate_MLE(self):
        # Truncation threshold with MLE
        daily_ret = self.logprc[M:] - self.logprc[:-M]
        daily_vol = np.mean(np.square(daily_ret / M))
        tau_ = self._tau()
        huber_loss = lambda x: np.minimum(0.5*x*x, tau_*np.abs(x) - 0.5*tau_**2)
        trunc = lambda x, tau: np.sign(x) * np.minimum(np.abs(x), tau)
        losses = {
            tau: huber_loss(daily_vol, trunc(self.logret, tau)) for tau in np.arange(tau_*0.1, tau*3.0, tau*0.1)
        }
        optimal_tau = min(losses, key=losses.get)
        truncated = np.sign(self.logret) * np.minimum(np.abs(self.logret), optimal_tau)
        assert truncated.shape == self.logret.shape
        return truncated.cumsum(), tau

    def benchmark(self):
        # Do not remove jumps
        return self.logprc, -1