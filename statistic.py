"""
statistic.py
------------
Mirrors the Java Statistic class from Choreography.

Provides descriptive statistics (mean, SD, median, quartiles, min, max)
plus the statistical distribution functions used internally:
  - Normal CDF / inverse CDF
  - Chi-squared, F, and t CDFs / inverse CDFs
  - Gamma, incomplete-Beta functions
  - Robust (outlier-trimming) computation
"""

from __future__ import annotations
import math
import numpy as np
from scipy import stats as _scipy_stats
from typing import Optional


# ---------------------------------------------------------------------------
# Constants (mirrors Statistic.java constants)
# ---------------------------------------------------------------------------
QUARTILE_TO_SD: float = 0.7413          # IQR / (2*z_0.75) ≈ IQR/2.698
HALFWIDTH_TO_SD: float = 1.4826         # 1 / (sqrt(2)*erfInv(0.5))
OVER_SQRT_2: float = 1.0 / math.sqrt(2)
OVER_SQRT_2_PI: float = 1.0 / math.sqrt(2 * math.pi)
SQRT_2: float = math.sqrt(2)
LN_TWO_PI: float = math.log(2 * math.pi)


# ---------------------------------------------------------------------------
# Low-level statistical functions (mirrors static methods in Statistic.java)
# ---------------------------------------------------------------------------

def erf(x: float) -> float:
    return math.erf(x)


def erfc(x: float) -> float:
    return math.erfc(x)


def erf_inv(x: float) -> float:
    """Inverse error function (mirrors Statistic.erfInv)."""
    return float(_scipy_stats.norm.ppf((x + 1) / 2) * OVER_SQRT_2)


def erfc_inv(x: float) -> float:
    """Inverse complementary error function."""
    return erf_inv(1.0 - x)


def cdf_normal(x: float) -> float:
    """Standard normal CDF Φ(x)."""
    return float(_scipy_stats.norm.cdf(x))


def icdf_normal(p: float) -> float:
    """Inverse standard normal CDF (probit)."""
    return float(_scipy_stats.norm.ppf(p))


def invnormcdf_tail(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Tail probability: P(X > x) = p → return x."""
    return float(_scipy_stats.norm.isf(p, loc=mu, scale=sigma))


def lngamma(x: float) -> float:
    return math.lgamma(x)


def gamma_func(x: float) -> float:
    return math.gamma(x)


def regular_lower_incomplete_gamma(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x)."""
    from scipy.special import gammainc
    return float(gammainc(a, x))


def regular_upper_incomplete_gamma(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x)."""
    from scipy.special import gammaincc
    return float(gammaincc(a, x))


def cdf_chi_sq(df: int, x: float) -> float:
    """χ² CDF with *df* degrees of freedom evaluated at *x*."""
    return float(_scipy_stats.chi2.cdf(x, df))


def cdf_not_chi_sq(df: int, x: float) -> float:
    """Survival function of χ² (1 − CDF)."""
    return float(_scipy_stats.chi2.sf(x, df))


def beta_func(a: int, b: int) -> float:
    from scipy.special import beta as _beta
    return float(_beta(a, b))


def incomplete_beta(x: float, a: int, b: int) -> float:
    """Incomplete beta function I_x(a, b) (regularised)."""
    from scipy.special import betainc
    return float(betainc(a, b, x))


def inc_beta_reg(x: float, a: int, b: int) -> float:
    return incomplete_beta(x, a, b)


def cdf_f_stat(f: float, df1: int, df2: int) -> float:
    return float(_scipy_stats.f.cdf(f, df1, df2))


def icdf_f_stat(p: float, df1: int, df2: int,
                tol: float = 1e-8) -> float:
    return float(_scipy_stats.f.ppf(p, df1, df2))


def cdf_t_stat(t: float, df: int) -> float:
    return float(_scipy_stats.t.cdf(t, df))


def icdf_t_stat(p: float, df: int, tol: float = 1e-8) -> float:
    return float(_scipy_stats.t.ppf(p, df))


# ---------------------------------------------------------------------------
# Statistic class
# ---------------------------------------------------------------------------

class Statistic:
    """
    Descriptive statistics for a 1-D float array.

    Attributes (all public, matching Statistic.java field names)
    -----------------------------------------------------------
    maximum, minimum, average, deviation : float
    median, first_quartile, last_quartile : float
    jitter : float  – robust spread estimate (scaled IQR)
    n : int         – number of valid (non-NaN) observations
    """

    # Class-level constants (accessible as Statistic.QUARTILE_TO_SD etc.)
    quartile_to_sd: float = QUARTILE_TO_SD
    halfwidth_to_sd: float = HALFWIDTH_TO_SD

    def __init__(self, data: Optional[np.ndarray] = None,
                 start: int = 0, end: Optional[int] = None):
        self.maximum: float = 0.0
        self.minimum: float = 0.0
        self.average: float = 0.0
        self.deviation: float = 0.0
        self.median: float = 0.0
        self.first_quartile: float = 0.0
        self.last_quartile: float = 0.0
        self.jitter: float = 0.0
        self.n: int = 0

        if data is not None:
            self.compute(data, start, end)

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def zero(self) -> None:
        """Reset all fields to zero."""
        self.maximum = self.minimum = self.average = self.deviation = 0.0
        self.median = self.first_quartile = self.last_quartile = 0.0
        self.jitter = 0.0
        self.n = 0

    def clone(self, other: "Statistic") -> "Statistic":
        """Copy fields from *other* into self; return self."""
        self.maximum = other.maximum
        self.minimum = other.minimum
        self.average = other.average
        self.deviation = other.deviation
        self.median = other.median
        self.first_quartile = other.first_quartile
        self.last_quartile = other.last_quartile
        self.jitter = other.jitter
        self.n = other.n
        return self

    def compute(self, data: np.ndarray,
                start: int = 0, end: Optional[int] = None) -> None:
        """
        Compute all statistics on data[start:end].
        NaN values are ignored (mirrors Java behaviour for special sentinel).
        """
        if end is None:
            end = len(data)
        arr = np.asarray(data[start:end], dtype=float)
        arr = arr[np.isfinite(arr)]
        self.n = len(arr)
        if self.n == 0:
            self.zero()
            return
        self.maximum = float(np.max(arr))
        self.minimum = float(np.min(arr))
        self.average = float(np.mean(arr))
        self.deviation = float(np.std(arr, ddof=1)) if self.n > 1 else 0.0
        self.median = float(np.median(arr))
        self.first_quartile = float(np.percentile(arr, 25))
        self.last_quartile = float(np.percentile(arr, 75))
        iqr = self.last_quartile - self.first_quartile
        self.jitter = iqr * QUARTILE_TO_SD

    def robust_compute(self, data: np.ndarray,
                       threshold: float = 2.5,
                       start: int = 0,
                       end: Optional[int] = None) -> int:
        """
        Compute statistics after iteratively discarding outliers.

        Points more than *threshold* standard deviations from the mean
        are removed iteratively until convergence (mirrors Statistic.robustCompute).

        Returns the number of retained observations.
        """
        if end is None:
            end = len(data)
        arr = np.asarray(data[start:end], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            self.zero()
            return 0

        for _ in range(100):   # cap iterations
            if len(arr) < 2:
                break
            mu = np.mean(arr)
            sd = np.std(arr, ddof=1)
            if sd == 0:
                break
            mask = np.abs(arr - mu) <= threshold * sd
            if mask.sum() == len(arr):
                break
            arr = arr[mask]

        self.compute(arr)
        return self.n

    def approximately_incorporate(self, other: "Statistic") -> None:
        """
        Merge another Statistic into self using a weighted-mean approximation
        (mirrors Statistic.approximatelyIncorporate).
        """
        if other.n == 0:
            return
        if self.n == 0:
            self.clone(other)
            return
        w1, w2 = float(self.n), float(other.n)
        total = w1 + w2
        new_mean = (w1 * self.average + w2 * other.average) / total
        # Pooled variance (exact when both means known)
        var1 = self.deviation ** 2
        var2 = other.deviation ** 2
        pooled_var = (
            (w1 * (var1 + (self.average - new_mean) ** 2)
             + w2 * (var2 + (other.average - new_mean) ** 2)) / total
        )
        self.average = new_mean
        self.deviation = math.sqrt(pooled_var)
        self.maximum = max(self.maximum, other.maximum)
        self.minimum = min(self.minimum, other.minimum)
        # Approximate: take weighted median
        self.median = (w1 * self.median + w2 * other.median) / total
        self.first_quartile = (
            w1 * self.first_quartile + w2 * other.first_quartile
        ) / total
        self.last_quartile = (
            w1 * self.last_quartile + w2 * other.last_quartile
        ) / total
        iqr = self.last_quartile - self.first_quartile
        self.jitter = iqr * QUARTILE_TO_SD
        self.n = int(total)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"Statistic(n={self.n}, mean={self.average:.4g}, "
                f"sd={self.deviation:.4g}, "
                f"median={self.median:.4g}, "
                f"[{self.minimum:.4g}, {self.maximum:.4g}])")

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": self.average,
            "sd": self.deviation,
            "median": self.median,
            "q1": self.first_quartile,
            "q3": self.last_quartile,
            "min": self.minimum,
            "max": self.maximum,
            "jitter": self.jitter,
        }

    # ------------------------------------------------------------------
    # Static distribution methods (forwarding to module-level functions)
    # ------------------------------------------------------------------

    @staticmethod
    def invnormcdf_tail(p: float, mu: float = 0.0,
                        sigma: float = 1.0) -> float:
        return invnormcdf_tail(p, mu, sigma)

    @staticmethod
    def icdf_normal(p: float) -> float:
        return icdf_normal(p)

    @staticmethod
    def erf(x: float) -> float:
        return erf(x)

    @staticmethod
    def erfc(x: float) -> float:
        return erfc(x)

    @staticmethod
    def erf_inv(x: float) -> float:
        return erf_inv(x)

    @staticmethod
    def erfc_inv(x: float) -> float:
        return erfc_inv(x)

    @staticmethod
    def cdf_normal(x: float) -> float:
        return cdf_normal(x)

    @staticmethod
    def lngamma(x: float) -> float:
        return lngamma(x)

    @staticmethod
    def gamma(x: float) -> float:
        return gamma_func(x)

    @staticmethod
    def cdf_chi_sq(df: int, x: float) -> float:
        return cdf_chi_sq(df, x)

    @staticmethod
    def cdf_not_chi_sq(df: int, x: float) -> float:
        return cdf_not_chi_sq(df, x)

    @staticmethod
    def cdf_f_stat(f: float, df1: int, df2: int) -> float:
        return cdf_f_stat(f, df1, df2)

    @staticmethod
    def icdf_f_stat(p: float, df1: int, df2: int,
                    tol: float = 1e-8) -> float:
        return icdf_f_stat(p, df1, df2, tol)

    @staticmethod
    def cdf_t_stat(t: float, df: int) -> float:
        return cdf_t_stat(t, df)

    @staticmethod
    def icdf_t_stat(p: float, df: int, tol: float = 1e-8) -> float:
        return icdf_t_stat(p, df, tol)

    @staticmethod
    def inc_beta_reg(x: float, a: int, b: int) -> float:
        return inc_beta_reg(x, a, b)

    @staticmethod
    def beta(a: int, b: int) -> float:
        return beta_func(a, b)

    @staticmethod
    def incomplete_beta(x: float, a: int, b: int) -> float:
        return incomplete_beta(x, a, b)
