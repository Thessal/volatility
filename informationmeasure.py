import numpy as np 

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


class InformationMeasure:
    def information_amount(self, x: np.ndarray, **kwargs) -> float:
        raise NotImplementedError
    
class Filtration(InformationMeasure):
    # 단순히 persistency를 measure하는 것일수도 있어서 주의 필요
    def information_amount(self, x: np.ndarray, n = 100) -> float:
        assert x.ndim == 1
        x = x.copy()
        
        # Original density (noise + information)
        d_orig = self.conditional_probability(x)

        # Shuffled (Filtration structure broken, noise only)
        results = []
        for _ in range(n):
            np.random.shuffle(x)
            d = self.conditional_probability(x)
            
            # KL-divergence (Information loss by breaking time dependency)
            # kld = np.mean(d * np.log(np.maximum(d_orig,1e-6) / np.maximum(d, 1e-6)))
            kld = np.mean(np.maximum(d_orig,1e-6) * np.log(np.maximum(d_orig,1e-6) / np.maximum(d, 1e-6)))
            results.append(kld)
        return float(np.mean(results))

    def conditional_probability(self, x):
        # Takens embedding
        x0 = x[:-1]
        x1 = x[1:]
        
        # Conditional expectation is a projection to known information set, in L2 space.
        # That is Ito calculus
        _, _, density = self.copula(x0, x1)
        return density
            
    def copula(self, x, y):
        # joint distribution estimation using KDE
        xy = np.stack([x, y]).T
        xy_flat = ro.FloatVector(xy.flatten())
        mat = base.matrix(xy_flat, ncol=2)
            
        # The `H` argument (bandwidth matrix) is determined automatically by default.
        result = ks.kde(x=mat)
        r_x = np.array(result.rx2('eval.points').rx2(1))
        r_y = np.array(result.rx2('eval.points').rx2(2))
        r_density = np.array(result.rx2('estimate'))

        return r_x, r_y, r_density



class Minkowski(InformationMeasure):
    #Minkowski measure
    pass

class Boxcounting(InformationMeasure):
    #Self similarity dimension using Box-counting method
    pass

class Fracdiff(InformationMeasure):
    # Projection to deterministic component
    # Stationarity test로 할지, 아니면 ito calculus 공부를 할지 고민중.
    # https://pubs.aip.org/aip/cha/article/35/2/023156/3336634/A-study-of-anomalous-stochastic-processes-via
    pass