import numpy as np 

class JumpAdjust:
    # Jump adjust / Heavy tail truncation
    # Returns jump adjusted log-price and the estimated jump size

    def __init__(self, logprc, M = 60*60*24):
        self.logprc = logprc
        self.logret = np.diff(self.logprc, append=[logprc[-1]])
        self.logret -= np.mean(self.logret) # detrend
        self.logprc = np.cumsum(self.logret)
        assert self.logprc.shape == self.logret.shape == logprc.shape

        self.n = len(self.logprc)
        self.M = M
    
    def _tau(self):
        return self._rolling_std(self.logret, self.M)

    def _rolling_std(self, logret, M):
        sq_cumsum = np.cumsum(np.square(logret))
        sq_mean = (sq_cumsum[M:] - sq_cumsum[:-M])/float(M)
        sq_mean = np.pad(sq_mean, (M,0), mode='edge')
        tau = np.sqrt(sq_mean)
        return tau

    def truncate(self):
        # Truncation at 1sigma
        tau = self._tau()
        truncated = np.sign(self.logret) * np.minimum(np.abs(self.logret), tau)
        assert truncated.shape == self.logret.shape
        return truncated.cumsum(), tau

    def huber(self, x, tau_):
        return np.where(
            np.abs(x) <= tau_,
            0.5*x*x, 
            tau_*np.abs(x) - 0.5*(tau_**2)
            )
    def trunc(self, x, tau):
        return np.sign(x) * np.minimum(np.abs(x), tau)

    def truncate_MLE(self):
        # Truncation threshold with MLE
        M = self.M

        daily_prc = (self.logprc[M:] - self.logprc[:-M])/np.sqrt(M)
        daily_prc = np.pad(daily_prc, (0,M), mode='edge')
        daily_std = self._rolling_std(daily_prc, self.M)

        tau_ = self._tau() # Huber theshold
        tau_scan = np.arange( tau_.mean()*0.1, tau_.max()*5, tau_.mean()*0.1 )
        # Truncation threshold from 0.1 sigma to 5 sigma
        losses = {
            tau: 
                self.huber(
                    daily_std - 
                    self._rolling_std(self.trunc(self.logret, tau), self.M),
                    tau_
                ).mean()
            for tau in tau_scan
        }
        # pd.Series(losses).plot()
        optimal_tau = min(losses, key=losses.get)
        truncated = np.sign(self.logret) * np.minimum(np.abs(self.logret), optimal_tau)
        assert truncated.shape == self.logret.shape
        return truncated.cumsum(), optimal_tau

    def benchmark(self):
        # Do not remove jumps
        return self.logprc, -1