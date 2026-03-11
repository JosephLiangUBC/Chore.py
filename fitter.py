"""
fitter.py
---------
Mirrors the Java Fitter class and its inner classes:
  Fitter.LineFitter     → LineFitter
  Fitter.CircleFitter   → CircleFitter
  Fitter.SpotFitter     → SpotFitter
  Fitter.EigenFinder    → EigenFinder
  Fitter.PolynomialRootFinder → PolynomialRootFinder

The Fitter accumulates (x, y) point pairs incrementally (via addL / addC)
and then dispatches to the appropriate sub-fitter.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Parameter result containers
# ---------------------------------------------------------------------------

@dataclass
class LinearParameters:
    """Result of a line fit: y = slope*x + intercept."""
    slope: float = 0.0
    intercept: float = 0.0
    r_squared: float = 0.0


@dataclass
class CircularParameters:
    """Result of a circle fit: centre (cx, cy) and radius r."""
    cx: float = 0.0
    cy: float = 0.0
    r: float = 0.0
    residual: float = 0.0


@dataclass
class SpotParameters:
    """Result of a 2-D Gaussian / spot fit: centre (cx, cy), amplitude, sigma."""
    cx: float = 0.0
    cy: float = 0.0
    amplitude: float = 0.0
    sigma: float = 1.0
    residual: float = 0.0


# ---------------------------------------------------------------------------
# Sub-fitters
# ---------------------------------------------------------------------------

class LineFitter:
    """
    Ordinary least-squares line fitter using accumulated sums.
    Mirrors Fitter$LineFitter.
    """

    def fit(self, fitter: "Fitter") -> LinearParameters:
        """Fit y = slope*x + intercept from fitter's accumulated sums."""
        n = fitter.n
        if n < 2:
            return LinearParameters()
        sx = fitter.Sx
        sy = fitter.Sy
        sxx = fitter.Sxx
        sxy = fitter.Sxy
        syy = fitter.Syy
        denom = n * sxx - sx * sx
        if denom == 0:
            return LinearParameters()
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        # R²
        ss_tot = syy - sy * sy / n
        ss_res = syy - slope * sxy - intercept * sy
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
        return LinearParameters(slope=slope, intercept=intercept,
                                r_squared=max(0.0, r2))


class CircleFitter:
    """
    Algebraic circle fitter (Pratt / Coope method) using accumulated sums.
    Mirrors Fitter$CircleFitter.
    """

    def fit(self, fitter: "Fitter") -> CircularParameters:
        """
        Fit a circle to the accumulated (x, y) points.
        Uses the algebraic method: minimise ‖Ax - b‖ subject to the
        algebraic circle constraint (mirrors Java's implementation).
        """
        n = fitter.n
        if n < 3:
            return CircularParameters()
        # Collect raw sums (centred for numerical stability)
        Ox, Oy = fitter.Ox, fitter.Oy
        sx  = fitter.Sx  - n * Ox
        sy  = fitter.Sy  - n * Oy
        sxx = fitter.Sxx - 2 * Ox * fitter.Sx + n * Ox * Ox
        syy = fitter.Syy - 2 * Oy * fitter.Sy + n * Oy * Oy
        sxy = fitter.Sxy - Ox * fitter.Sy - Oy * fitter.Sx + n * Ox * Oy
        sxz = fitter.Sxz - Ox * (fitter.Sxx + fitter.Syy) \
              - Oy * fitter.Sxy + Ox * (fitter.Sx * Ox + fitter.Sy * Oy) \
              # Approximate (full Szz not stored separately)
        # Rebuild with a simple normal-equation approach (Coope):
        # Build matrix M and vector b from (x_i, y_i, z_i=x²+y²)
        # [Σx², Σxy, Σx] [a]   [Σxz]
        # [Σxy, Σy², Σy] [b] = [Σyz]
        # [Σx,  Σy,  n ] [c]   [Σz ]
        # where circle: x²+y² = a*x + b*y + c → centre=(a/2,b/2), r²=(a²+b²)/4+c
        sx_  = fitter.Sx
        sy_  = fitter.Sy
        sxx_ = fitter.Sxx
        syy_ = fitter.Syy
        sxy_ = fitter.Sxy
        sxz_ = fitter.Sxz
        syz_ = fitter.Syz
        szz_ = fitter.Szz
        A = np.array([
            [sxx_, sxy_, sx_],
            [sxy_, syy_, sy_],
            [sx_,  sy_,  float(n)],
        ])
        b = np.array([sxz_, syz_, szz_ - sxx_ - syy_])
        # Solve for [a, b, c] where circle = x²+y² - ax - by - c = 0
        try:
            # Use (x²+y²) = a*x + b*y + c  →  centre = (a/2, b/2)
            # Rewrite as standard normal equations
            b2 = np.array([sxz_, syz_, szz_])
            # Actually: the equation is (x-cx)²+(y-cy)²=r²
            # Expanding: x²+y² - 2cx*x - 2cy*y + (cx²+cy²-r²)=0
            # Let A=-2cx, B=-2cy, C=cx²+cy²-r²
            # Σ(x²+y²)*x = A*Σx² + B*Σxy + C*Σx
            sol = np.linalg.lstsq(A, b2, rcond=None)[0]
            cx = -sol[0] / 2.0 + (fitter.Ox if fitter.automove else 0)
            cy = -sol[1] / 2.0 + (fitter.Oy if fitter.automove else 0)
            r = math.sqrt(max(0.0, sol[0] ** 2 / 4 + sol[1] ** 2 / 4 - sol[2]))
        except np.linalg.LinAlgError:
            return CircularParameters()
        return CircularParameters(cx=cx, cy=cy, r=r)


class SpotFitter:
    """
    2-D weighted centroid (spot) fitter.
    Mirrors Fitter$SpotFitter – fits centre of a bright region using
    intensity-weighted moments (treating z = intensity).
    """

    def fit(self, fitter: "Fitter") -> SpotParameters:
        n = fitter.n
        if n == 0:
            return SpotParameters()
        # Weighted centroid using Szz as total weight proxy
        # Sxz = Σ x_i * z_i, Syz = Σ y_i * z_i, Szz = Σ z_i
        total_z = fitter.Szz
        if total_z == 0:
            return SpotParameters()
        cx = fitter.Sxz / total_z
        cy = fitter.Syz / total_z
        # Spread (intensity-weighted variance → sigma)
        var = ((fitter.Sxx - 2 * cx * fitter.Sx + n * cx * cx)
               + (fitter.Syy - 2 * cy * fitter.Sy + n * cy * cy)) / n
        sigma = math.sqrt(max(0.0, var))
        return SpotParameters(cx=cx, cy=cy, amplitude=total_z / n, sigma=sigma)


class EigenFinder:
    """
    2×2 covariance eigenvector finder.
    Mirrors Fitter$EigenFinder – returns principal axes of an (x, y)
    scatter via the analytic 2×2 eigen-decomposition.
    """

    def principal_axes(self, fitter: "Fitter") -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Returns (v1, v2, lambda1, lambda2) where v1 is the major axis
        (unit vector) and lambda1 >= lambda2 are eigenvalues.
        """
        n = fitter.n
        if n < 2:
            return (np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0, 0.0)
        mx = fitter.Sx / n
        my = fitter.Sy / n
        cxx = fitter.Sxx / n - mx * mx
        cyy = fitter.Syy / n - my * my
        cxy = fitter.Sxy / n - mx * my
        # Characteristic polynomial: λ² - (cxx+cyy)λ + (cxx*cyy - cxy²) = 0
        tr = cxx + cyy
        det = cxx * cyy - cxy * cxy
        disc = max(0.0, tr * tr / 4 - det)
        sq = math.sqrt(disc)
        l1 = tr / 2 + sq
        l2 = tr / 2 - sq
        # Major eigenvector
        if abs(cxy) > 1e-12:
            v1 = np.array([cxy, l1 - cxx])
        else:
            v1 = np.array([1.0, 0.0]) if cxx >= cyy else np.array([0.0, 1.0])
        norm = np.linalg.norm(v1)
        if norm > 0:
            v1 /= norm
        v2 = np.array([-v1[1], v1[0]])
        return v1, v2, l1, l2


class PolynomialRootFinder:
    """
    Finds real roots of a polynomial using numpy.
    Mirrors Fitter$PolynomialRootFinder.
    """

    @staticmethod
    def real_roots(coefficients: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        """
        Return the real roots of the polynomial whose coefficients are given
        in *coefficients* (highest degree first, numpy convention).
        """
        roots = np.roots(coefficients)
        real_roots = roots[np.abs(roots.imag) < tol].real
        return np.sort(real_roots)

    @staticmethod
    def trisect_cosine(x: float) -> float:
        """
        Solve cos(3θ) = x for θ in [0, π/3] — mirrors Fitter.trisectCosine.
        Used in the algebraic circle fitter's discriminant.
        """
        # cos(3θ) = x → θ = arccos(x)/3
        x = max(-1.0, min(1.0, x))
        return math.acos(x) / 3.0


# ---------------------------------------------------------------------------
# Main Fitter class
# ---------------------------------------------------------------------------

class Fitter:
    """
    Incremental moment accumulator and dispatcher for geometric fitting.

    Mirrors the Java Fitter class.  Points are added with addL (linear)
    or addC (centred after automove shift), and then one of the sub-fitters
    is invoked.

    Attributes
    ----------
    Ox, Oy : float   – accumulated origin shift (used by automove)
    Sx, Sy : float   – Σx, Σy
    Sxx, Syy, Sxy    – Σx², Σy², Σxy
    Sxz, Syz, Szz    – Σx·z, Σy·z, Σz  (z = x²+y² by default for circles)
    n      : int     – number of points added
    automove : bool  – if True, keep origin centred on the point cloud
    """

    def __init__(self, other: Optional["Fitter"] = None):
        if other is not None:
            self._copy_from(other)
        else:
            self.reset()
        self.automove: bool = True
        self._eigen = EigenFinder()
        self.polyrf = PolynomialRootFinder()
        self.spot = SpotFitter()
        self.line = LineFitter()
        self.circ = CircleFitter()

    def reset(self) -> "Fitter":
        self.Ox = self.Oy = 0.0
        self.Sx = self.Sy = 0.0
        self.Sxx = self.Syy = self.Sxy = 0.0
        self.Sxz = self.Syz = self.Szz = 0.0
        self.n = 0
        return self

    def _copy_from(self, other: "Fitter") -> None:
        self.Ox = other.Ox; self.Oy = other.Oy
        self.Sx = other.Sx; self.Sy = other.Sy
        self.Sxx = other.Sxx; self.Syy = other.Syy; self.Sxy = other.Sxy
        self.Sxz = other.Sxz; self.Syz = other.Syz; self.Szz = other.Szz
        self.n = other.n

    def join(self, other: "Fitter") -> "Fitter":
        """Merge another Fitter's sums into self; return self."""
        self.Sx += other.Sx; self.Sy += other.Sy
        self.Sxx += other.Sxx; self.Syy += other.Syy; self.Sxy += other.Sxy
        self.Sxz += other.Sxz; self.Syz += other.Syz; self.Szz += other.Szz
        self.n += other.n
        return self

    def shift_zero(self, centre: bool = True) -> None:
        """Shift origin to the current centroid (if centre=True)."""
        if centre and self.n > 0:
            mx = self.Sx / self.n
            my = self.Sy / self.n
            self.move_by(-mx, -my)

    def move_by(self, dx: float, dy: float) -> None:
        """Translate the accumulated points by (dx, dy)."""
        self.Sxy += dx * self.Sy + dy * self.Sx + self.n * dx * dy
        self.Sxz += dx * (self.Sxx + self.Syy) + 2 * dy * self.Sxy \
                    + self.n * dx * (dx * dx + dy * dy)  # approximate
        self.Syz += dy * (self.Sxx + self.Syy) + 2 * dx * self.Sxy \
                    + self.n * dy * (dx * dx + dy * dy)
        self.Sxx += 2 * dx * self.Sx + self.n * dx * dx
        self.Syy += 2 * dy * self.Sy + self.n * dy * dy
        self.Sx  += self.n * dx
        self.Sy  += self.n * dy
        self.Ox  -= dx
        self.Oy  -= dy

    # ------------------------------------------------------------------
    # Point accumulation
    # ------------------------------------------------------------------

    def addL(self, x: float, y: float) -> None:
        """Add a point (x, y) using linear (non-centred) accumulation."""
        self.Sx += x; self.Sy += y
        self.Sxx += x * x; self.Syy += y * y; self.Sxy += x * y
        z = x * x + y * y
        self.Sxz += x * z; self.Syz += y * z; self.Szz += z
        self.n += 1

    def subL(self, x: float, y: float) -> None:
        """Remove a previously added point."""
        self.Sx -= x; self.Sy -= y
        self.Sxx -= x * x; self.Syy -= y * y; self.Sxy -= x * y
        z = x * x + y * y
        self.Sxz -= x * z; self.Syz -= y * z; self.Szz -= z
        self.n -= 1

    def addC(self, x: float, y: float) -> None:
        """Add a point with optional automove centring."""
        if self.automove and self.n == 0:
            self.Ox, self.Oy = x, y
        cx, cy = x - self.Ox, y - self.Oy
        self.addL(cx, cy)

    def subC(self, x: float, y: float) -> None:
        cx, cy = x - self.Ox, y - self.Oy
        self.subL(cx, cy)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def add_points(self, xy: np.ndarray) -> None:
        """Add all rows of an (N, 2) array."""
        for row in xy:
            self.addL(float(row[0]), float(row[1]))

    @classmethod
    def from_points(cls, xy: np.ndarray) -> "Fitter":
        f = cls()
        f.add_points(xy)
        return f

    # ------------------------------------------------------------------
    # Fitting dispatch
    # ------------------------------------------------------------------

    def fit_line(self) -> LinearParameters:
        return self.line.fit(self)

    def fit_circle(self) -> CircularParameters:
        return self.circ.fit(self)

    def fit_spot(self) -> SpotParameters:
        return self.spot.fit(self)

    def principal_axes(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        return self._eigen.principal_axes(self)

    # ------------------------------------------------------------------
    # Convenience: fit from a numpy array directly
    # ------------------------------------------------------------------

    @staticmethod
    def line_fit(xy: np.ndarray) -> LinearParameters:
        f = Fitter.from_points(xy)
        return f.fit_line()

    @staticmethod
    def circle_fit(xy: np.ndarray) -> CircularParameters:
        f = Fitter.from_points(xy)
        return f.fit_circle()

    @staticmethod
    def eigen_axes(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        f = Fitter.from_points(xy)
        return f.principal_axes()

    # ------------------------------------------------------------------
    # Noise / position error estimate (mirrors Choreography usage)
    # ------------------------------------------------------------------

    def position_noise(self) -> float:
        """
        Estimate positional noise as the RMS residual from the best-fit line.
        Used by Dance.positionNoiseEstimate().
        """
        if self.n < 2:
            return 0.0
        lp = self.fit_line()
        # Residual variance = Syy - slope*Sxy - intercept*Sy
        resid_var = (self.Syy - lp.slope * self.Sxy
                     - lp.intercept * self.Sy)
        return math.sqrt(max(0.0, resid_var / self.n))
