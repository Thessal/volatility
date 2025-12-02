import numpy as np
from collections import Counter
import pandas as pd 

# These will let us use R packages:
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
# Load the packages
try:
    base = importr('base')
    ks = importr('ks')
except:
    print("Failed to load R packages")


class Evaluation:

    def __init__(self, logprc, K = 60*10):
        assert logprc.ndim == 1
        self.logprc = logprc.copy()
        self.K = K

    def filtration_information(self) -> float:
        # time fltration의 information 측정. KLD 기반.

        # Return of sampled log price
        logret = np.diff(self.logprc[::self.K])
        logret = logret / np.std(logret)
        most_frequent, _ = Counter(logret).most_common(n=1)[0]
        logret = logret[(logret != most_frequent) & (
            np.abs(logret) < 10)]  # remove extreme points
        x = logret.copy()

        # Conditional probability density (noise + information)
        d_cond = self.conditional_probability(x[:-1], x[1:])

        # Unconditional prob density (time structure broken, prior noise)
        d_uncond = self.unconditional_probability(x)

        kld = np.mean(np.maximum(d_cond, 1e-6) *
                      np.log(np.maximum(d_cond, 1e-6) / np.maximum(d_uncond, 1e-6)))
        # kld = np.mean(np.maximum(d_uncond, 1e-6) *
        #               np.log(np.maximum(d_cond, 1e-6) / np.maximum(d_uncond, 1e-6)))
        # KLD here means information amount of timd dependency
        return float(kld)

    def unconditional_probability(self, x0):
        x = ro.FloatVector(x0)
        result = ks.kde(x=x, gridsize=ro.FloatVector(
            [1000]), xmin=ro.FloatVector([-10]), xmax=ro.FloatVector([10]))
        r_x = np.array(result.rx2('eval.points'))
        r_density = np.array(result.rx2('estimate'))
        # pd.Series(r_density, index=r_x).plot()
        # pd.Series(x0).plot.hist(bins=300)
        density = np.outer(r_density, r_density)

        density = density.ravel()
        density = np.abs(density)
        density /= np.sum(density)
        return density

    def conditional_probability(self, x0, x1):
        # # Takens embedding
        # x0 = x[:-1]
        # x1 = x[1:]
        assert np.isfinite(x0).all()
        assert np.isfinite(x1).all()

        # Conditional expectation is a projection to known information set, in L2 space.
        # That is Ito calculus
        _, _, density = self.copula(x0, x1)

        # Trivial normalization
        density = np.abs(density.ravel())
        density /= np.sum(density)
        return density

    def copula(self, x, y):
        # joint distribution estimation using KDE
        xy = np.stack([x, y]).copy()  # .T
        xy_flat = ro.FloatVector(xy.flatten())
        mat = base.matrix(xy_flat, ncol=2)
        assert len(mat) > 0
        assert np.mean(np.abs(np.array(mat) - xy.T)) < 1e-6

        # The `H` argument (bandwidth matrix) is determined automatically by default.
        result = ks.kde(x=mat, gridsize=ro.FloatVector([1000, 1000]), xmin=ro.FloatVector(
            [-10, -10]), xmax=ro.FloatVector([10, 10]))
        r_x = np.array(result.rx2('eval.points').rx2(1))
        r_y = np.array(result.rx2('eval.points').rx2(2))
        r_density = np.array(result.rx2('estimate'))

        return r_x, r_y, r_density
    

    def fractal_dimension(self) -> float:
        # self-similarity dimension, box-counting method with Takens embedding
        x = self.logprc.copy()
        df_point_cloud = pd.DataFrame(np.stack( [np.linspace(0, 1, len(x)) , ( x.max() - x ) / ( x.max() - x.min() )] ).T)
        df_box_counting = pd.Series({ 
            np.log(grid_count): np.log(len(df_point_cloud.multiply(grid_count-1e-6).apply(np.floor).drop_duplicates()))
            for grid_count in range(10,100)
        }) # {log(1/grid size) : log(box count)}
        dimension = ((df_box_counting.diff()) / (df_box_counting.index.diff())).mean()
        return dimension # 1 if linear, 1.82 if gaussian, 2 if uniform random


# class Minkowski(InformationMeasure):
#     # Minkowski measure
#     pass

