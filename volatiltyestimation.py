import numpy as np
from typing import Dict, List
from informationmeasure import Filtration
import scipy
import statsmodels.tsa.stattools as st


class VolatilityEstimator:

    def __init__(self, x: np.ndarray):
        assert x.ndim == 1
        self.logprc = np.log(x)
        assert np.isfinite(self.logprc).all()
        self.params: List[Dict] = []
        self.optimal_param: Dict = None

    def _pred(self, logprc, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def pred(self, **kwargs) -> float:
        if not self.optimal_param:
            raise Exception("Need to call fit() first")
        logprc = self.logprc.copy()
        result = self._pred(logprc, **self.optimal_param)
        return float(result)

    def fit(self) -> dict:
        # Determine optimal parameter
        raise NotImplementedError


class TSRV(VolatilityEstimator):

    def _pred(self, logprc, sampling_period: int) -> np.ndarray:
        # Two-scale Realized variance
        x1 = np.square(logprc[:-sampling_period] - logprc[sampling_period:]).mean() 
        x2 = np.square(np.diff(logprc)).mean() 
        tsrv = x1 - x2
        # spot_variance = (tsrv[t] - tsrv[t-h]) / h #???
        return tsrv


class PRV(VolatilityEstimator):

    def _pred(self, logprc, sampling_period: int) -> np.ndarray:
        logret = np.diff(logprc)

        logret_preavg, kernel = self.preaveraging(logret, sampling_period)
        rv = np.square(logret_preavg).mean()
        noise_adjustment = np.square(logret).mean(
        ) * 0.5 * (np.square(np.diff(kernel)).sum())

        return rv - noise_adjustment

    def preaveraging(self, logret, window):
        # closer to true price return
        assert 0 < window < len(logret)
        kernel = np.minimum(np.linspace(0, 1, window+2),
                            np.linspace(1, 0, window+2))[1:-1]
        kernel /= np.sqrt((np.square(kernel).sum()))  # psi = 1
        logret_preavg = np.convolve(logret, kernel, mode='valid')
        return logret_preavg, kernel

    def true_price(self, logprc, window):
        kernel = np.ones(max(1,window//2)) 
        kernel /= len(kernel)
        return np.convolve(logprc, kernel, mode='valid')

    def fit(self) -> dict:
        logprc = self.logprc.copy()
        logret = np.diff(logprc)
        windows = range(1, 20, 2)
        iteration = 30  # TODO : increase
        scores = dict()

        information_measure = Filtration().information_amount
        for w in windows:
            logret_avg = self.preaveraging(logret, w)[0]
            # info_amt = information_measure(logret_avg[::max(1,w//2)], n=iteration)
            info_amt = information_measure(logret_avg, n=iteration)
            rv = np.square(np.diff(self.true_price(logprc, window=w))).mean()
            prv = self._pred(logprc, w)
            score = prv/info_amt # FIXME : no mathematical resoning. need to think
            print(
                f"window = {w}\t information amount = {info_amt:.2e}\t prv/rv = {prv/rv:.2e}\t rv = {rv:.2e}\t prv = {prv:.2e} \t score = {score:.2e}")
            scores[w] = score
        optimal_param = {"window": max(scores, key=scores.get)}
        self.optimal_param = optimal_param
        self.scores = scores
        return optimal_param


class ADF():
    # adfuller 기반 window 찾기
    pass

class TRV():
    pass


class Fracdiff:
    # Stationary 해질때까지 차분하여 Fourier/Pade transform

    def comb(self, n, k):
        return scipy.special.gamma(n+1)/scipy.special.gamma(k+1)/scipy.special.gamma(n-k+1)

    def fracdiff_kernel(self, order=2.5, tau=1e-2):
        max_k = 1000
        coeffs = []
        for k in range(max_k):
            coeff = (-1)**k * self.comb(order, k)
            coeffs.append(coeff)
            if abs(coeff) < tau:
                break
        return np.array(coeffs)

    def fracdiff(self, x: np.array, order: float, tau=1e-2):
        kernel = self.fracdiff_kernel(order, tau)
        return np.convolve(kernel, x)[:-len(kernel)+1]

    # ## Stationary test over various fracdiff dimension
    # # df_pvalues = pd.Series({order:st.adfuller(fracdiff(x_shift_cumsum, order))[1] for order in np.arange(0,1.5,0.01)})
    # # df_pvalues.plot(marker=".", grid=True)
    # # plt.gca().set_xticks(df_pvalues.index, minor=True)
    # # plt.axhline(0.05, color="orange")
    # # plt.axvline(1.28, color="gray")
