"""
measures/eigenspine.py
----------------------
Mirrors Eigenspine.java – PCA decomposition of spine postures.

NIPALS algorithm: iteratively finds principal components of the
(frames × spine_points*2) data matrix.

Provides:
  - Eigenspine class  – plugin-style PCA analysis
  - nipals()          – standalone NIPALS PCA function
  - project_spine()   – project a spine onto pre-computed PCs
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..dance import Dance


# ---------------------------------------------------------------------------
# NIPALS PCA (mirrors Eigenspine.doNIPALS)
# ---------------------------------------------------------------------------

def nipals(X: np.ndarray,
           n_components: int = 3,
           centre: bool = True,
           tol: float = 1e-6,
           max_iter: int = 500
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Non-linear Iterative Partial Least Squares (NIPALS) PCA.

    Mirrors Eigenspine.doNIPALS.

    Parameters
    ----------
    X            : (n_samples, n_features) data matrix
    n_components : number of PCs to extract
    centre       : subtract column means before decomposition
    tol          : convergence tolerance for loading vectors
    max_iter     : maximum iterations per component

    Returns
    -------
    scores      : (n_samples, n_components) score matrix T
    loadings    : (n_components, n_features) loading matrix P
    explained   : (n_components,) fraction of variance explained by each PC
    mean        : (n_features,) column means (or zeros if centre=False)
    """
    X = np.array(X, dtype=float)
    mean = X.mean(axis=0) if centre else np.zeros(X.shape[1])
    E = X - mean

    total_var = np.sum(E ** 2)

    scores   = np.zeros((X.shape[0], n_components))
    loadings = np.zeros((n_components, X.shape[1]))
    explained = np.zeros(n_components)

    for k in range(n_components):
        # Initial guess: column with maximum variance
        var_cols = np.var(E, axis=0)
        t = E[:, np.argmax(var_cols)].copy()
        t_old = np.zeros_like(t)

        for _ in range(max_iter):
            # Loading: p = E'*t / (t'*t)
            tt = np.dot(t, t)
            if tt < 1e-14:
                break
            p = E.T @ t / tt
            p /= np.linalg.norm(p) + 1e-14   # normalise
            # Score: t = E*p
            t = E @ p
            if np.linalg.norm(t - t_old) < tol * np.linalg.norm(t):
                break
            t_old = t.copy()

        scores[:, k]   = t
        loadings[k, :] = p
        # Deflate
        E -= np.outer(t, p)
        explained[k] = np.sum(t ** 2) * np.sum(p ** 2) / (total_var + 1e-14)

    return scores, loadings, explained, mean


# ---------------------------------------------------------------------------
# EigenSpine result
# ---------------------------------------------------------------------------

@dataclass
class EigenSpineResult:
    """Per-frame PC scores for one Dance."""
    dance_id:   int
    scores:     np.ndarray   # (n_frames, n_components)
    times:      np.ndarray   # (n_frames,)
    explained:  np.ndarray   # (n_components,)


# ---------------------------------------------------------------------------
# Eigenspine class (plugin)
# ---------------------------------------------------------------------------

class Eigenspine:
    """
    PCA decomposition of spine shape across all frames of a dataset.

    Mirrors Eigenspine.java.

    Usage
    -----
    >>> es = Eigenspine(n_components=3, n_spine_points=11)
    >>> es.fit(dances)            # compute PCs from all spines
    >>> scores = es.transform(dance)   # project one worm onto PCs
    >>> df = es.to_dataframe()    # all scores as DataFrame
    """

    EXTENSION = ".eig"

    def __init__(self,
                 n_components:    int   = 3,
                 n_spine_points:  int   = 11,
                 centre:          bool  = True,
                 normalise_length: bool = True):
        self.n_components    = n_components
        self.n_spine_points  = n_spine_points
        self.centre          = centre
        self.normalise_length = normalise_length

        # Fit outputs
        self.mean_:      Optional[np.ndarray] = None
        self.idev_:      Optional[np.ndarray] = None
        self.loadings_:  Optional[np.ndarray] = None
        self.explained_: Optional[np.ndarray] = None

        self._results: Dict[int, EigenSpineResult] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, dances: Dict[int, "Dance"]) -> "Eigenspine":
        """
        Compute principal components from all available spines.

        The data matrix has shape (total_frames_with_spine, n_spine_points*2).
        Each row is a flattened, length-normalised spine.
        """
        matrix_rows = []
        for d in dances.values():
            for sp in d.spine:
                if sp is not None and sp.size() >= 2:
                    row = self._flatten_spine(sp)
                    if row is not None:
                        matrix_rows.append(row)

        if not matrix_rows:
            raise ValueError("No valid spines found for Eigenspine fitting.")

        X = np.array(matrix_rows, dtype=float)
        scores, loadings, explained, mean = nipals(
            X, self.n_components, self.centre)

        self.mean_      = mean
        self.loadings_  = loadings
        self.explained_ = explained
        # Inverse std devs for whitening (optional)
        self.idev_ = np.ones(X.shape[1])
        std = X.std(axis=0)
        mask = std > 0
        self.idev_[mask] = 1.0 / std[mask]

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, dance: "Dance") -> EigenSpineResult:
        """Project dance spines onto the fitted PCs."""
        if self.mean_ is None:
            raise RuntimeError("Eigenspine must be fitted before transform.")

        n = dance.n_frames
        scores = np.full((n, self.n_components), np.nan, dtype=np.float32)

        for i, sp in enumerate(dance.spine):
            if sp is None:
                continue
            row = self._flatten_spine(sp)
            if row is None:
                continue
            centred = row - self.mean_
            scores[i] = self.loadings_ @ centred

        result = EigenSpineResult(
            dance_id=dance.ID,
            scores=scores,
            times=dance.times.copy(),
            explained=self.explained_.copy()
        )
        self._results[dance.ID] = result
        return result

    def fit_transform(self, dances: Dict[int, "Dance"]
                      ) -> Dict[int, EigenSpineResult]:
        self.fit(dances)
        return {wid: self.transform(d) for wid, d in dances.items()}

    # ------------------------------------------------------------------
    # Plugin quantifier interface
    # ------------------------------------------------------------------

    def quantifier_count(self) -> int:
        return self.n_components

    def quantifier_title(self, i: int) -> str:
        return f"PC{i + 1}"

    def compute_dancer_quantity(self, dance: "Dance", index: int) -> np.ndarray:
        """Return time-course of PC score *index* for *dance*."""
        if dance.ID not in self._results:
            self.transform(dance)
        return self._results[dance.ID].scores[:, index]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_dataframe(self, include_nan: bool = False):
        """
        Return a DataFrame with columns [worm_id, time, PC1, PC2, …].
        """
        import pandas as pd
        rows = []
        for wid, res in self._results.items():
            for i, t in enumerate(res.times):
                row = {'worm_id': wid, 'time': float(t)}
                for k in range(self.n_components):
                    row[f"PC{k + 1}"] = float(res.scores[i, k])
                if include_nan or not any(
                    math.isnan(res.scores[i, k])
                    for k in range(self.n_components)
                ):
                    rows.append(row)
        return pd.DataFrame(rows)

    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of total variance explained by each PC."""
        if self.explained_ is None:
            return np.array([])
        return self.explained_.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flatten_spine(self, sp) -> Optional[np.ndarray]:
        """Resample spine to n_spine_points, normalise length, flatten."""
        from ..utils import resample_polyline, cumulative_arc_length
        pts = sp.points
        if len(pts) < 2:
            return None
        resampled = resample_polyline(pts, self.n_spine_points)
        if self.normalise_length:
            arc = cumulative_arc_length(resampled)
            total = arc[-1]
            if total > 0:
                resampled = resampled / total
            else:
                return None
        # Translate so first point is at origin
        resampled = resampled - resampled[0]
        return resampled.flatten()
