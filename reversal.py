"""
measures/reversal.py
--------------------
Mirrors MeasureReversal.java – detailed reversal detection and
characterisation for C. elegans locomotion data.

A reversal is a continuous backward-locomotion episode.  This module
provides:

  - detect_reversals(dance, ...)   – raw reversal event list
  - Reversal                        – structured reversal record
  - MeasureReversal                – plugin-style class matching the Java API
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .dance import Dance


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Reversal:
    """One reversal episode.  Mirrors MeasureReversal$Reversal."""
    start_index: int       # index in Dance.times
    end_index:   int       # exclusive
    start_time:  float     # seconds
    end_time:    float     # seconds

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def n_frames(self) -> int:
        return self.end_index - self.start_index


@dataclass
class ReversalResult:
    """
    Per-worm reversal analysis result.  Mirrors MeasureReversal$Result.
    """
    dance_id:     int
    reversals:    List[Reversal] = field(default_factory=list)

    # Summary statistics
    count:              int   = 0
    mean_duration:      float = 0.0
    sd_duration:        float = 0.0
    mean_distance:      float = 0.0
    mean_speed:         float = 0.0
    inter_reversal_interval: float = 0.0   # mean time between reversals
    rate:               float = 0.0        # reversals / second


# ---------------------------------------------------------------------------
# Core reversal detection
# ---------------------------------------------------------------------------

def _signed_speed(dance: "Dance",
                  mm_per_pixel: float = 1.0,
                  window: float = 0.5) -> np.ndarray:
    """Return signed speed array (negative = backward)."""
    return dance.quantityIsSpeed(mm_per_pixel, window, signed=True)


def detect_reversals(dance:          "Dance",
                     mm_per_pixel:   float = 1.0,
                     window:         float = 0.5,
                     min_duration:   float = 0.2,
                     min_distance:   float = 0.01,
                     require_fwd:    bool  = False,
                     time_range:     Optional[Tuple[float, float]] = None
                     ) -> List[Reversal]:
    """
    Identify backward locomotion (reversal) episodes in *dance*.

    Parameters
    ----------
    dance          : Dance object
    mm_per_pixel   : spatial calibration
    window         : speed-smoothing window (s)
    min_duration   : minimum reversal duration to retain (s)
    min_distance   : minimum displacement during reversal to retain (mm)
    require_fwd    : if True, only count reversals followed by forward motion
    time_range     : (t_start, t_end) to restrict analysis

    Returns
    -------
    List of Reversal objects, sorted by start_time.
    """
    if len(dance.times) < 2:
        return []

    sp = _signed_speed(dance, mm_per_pixel, window)
    times = dance.times.astype(float)

    # Optional time mask
    if time_range is not None:
        mask = (times >= time_range[0]) & (times <= time_range[1])
        sp = np.where(mask, sp, np.nan)

    reversals: List[Reversal] = []
    n = len(sp)
    in_rev = False
    start_i = 0

    for i in range(n):
        s = sp[i]
        if math.isnan(s):
            if in_rev:
                _commit_reversal(reversals, dance, start_i, i, times,
                                 mm_per_pixel, min_duration, min_distance)
                in_rev = False
            continue

        if not in_rev and s < 0.0:
            in_rev = True
            start_i = i
        elif in_rev and s >= 0.0:
            _commit_reversal(reversals, dance, start_i, i, times,
                             mm_per_pixel, min_duration, min_distance)
            in_rev = False

    if in_rev:
        _commit_reversal(reversals, dance, start_i, n, times,
                         mm_per_pixel, min_duration, min_distance)

    # Optionally require forward motion after reversal
    if require_fwd and len(reversals) > 1:
        keep = []
        for i, rev in enumerate(reversals[:-1]):
            next_start = reversals[i + 1].start_time
            # Check mean speed in the gap
            gap_mask = ((times >= rev.end_time) &
                        (times < next_start))
            if np.any(gap_mask):
                gap_sp = float(np.nanmean(sp[gap_mask]))
                if gap_sp > 0:
                    keep.append(rev)
            else:
                keep.append(rev)
        # Always keep last
        keep.append(reversals[-1])
        reversals = keep

    return reversals


def _commit_reversal(reversals: List[Reversal],
                     dance: "Dance",
                     start_i: int, end_i: int,
                     times: np.ndarray,
                     mm_per_pixel: float,
                     min_duration: float,
                     min_distance: float) -> None:
    if end_i <= start_i:
        return
    t0 = float(times[start_i])
    t1 = float(times[min(end_i, len(times) - 1)])
    dur = t1 - t0
    if dur < min_duration:
        return
    # Distance
    seg = dance.centroid[start_i:end_i]
    if len(seg) > 1:
        disp = float(np.sum(
            np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1]))
        )) * mm_per_pixel
    else:
        disp = 0.0
    if disp < min_distance:
        return
    reversals.append(Reversal(start_i, end_i, t0, t1))


# ---------------------------------------------------------------------------
# MeasureReversal class (plugin interface)
# ---------------------------------------------------------------------------

class MeasureReversal:
    """
    Mirrors MeasureReversal.java – a CustomComputation plugin for
    detailed reversal analysis.

    Usage
    -----
    >>> mr = MeasureReversal(mm_per_pixel=0.025, window=0.5,
    ...                      min_duration=0.2, min_distance=0.01)
    >>> results = mr.compute_all(dances)      # dict: id → ReversalResult
    >>> df = mr.to_dataframe(results)
    """

    EXTENSION = ".rev"

    def __init__(self,
                 mm_per_pixel:   float = 1.0,
                 window:         float = 0.5,
                 min_duration:   float = 0.2,
                 min_distance:   float = 0.01,
                 require_fwd:    bool  = False,
                 time_range:     Optional[Tuple[float, float]] = None,
                 separate_files: bool  = False):
        self.mm_per_pixel   = mm_per_pixel
        self.window         = window
        self.min_duration   = min_duration
        self.min_distance   = min_distance
        self.require_fwd    = require_fwd
        self.time_range     = time_range
        self.separate_files = separate_files
        self.results:       Dict[int, ReversalResult] = {}
        self._reversal_map: Dict[int, List[Reversal]] = {}

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def validate_dancer(self, dance: "Dance") -> bool:
        """Return True if dance has enough data to analyse."""
        return dance.n_frames >= 5 and dance.duration() > 0.1

    def compute_dancer(self, dance: "Dance") -> ReversalResult:
        """Compute reversals for one Dance and return a ReversalResult."""
        revs = detect_reversals(
            dance,
            mm_per_pixel=self.mm_per_pixel,
            window=self.window,
            min_duration=self.min_duration,
            min_distance=self.min_distance,
            require_fwd=self.require_fwd,
            time_range=self.time_range,
        )
        self._reversal_map[dance.ID] = revs
        result = self._summarise(dance, revs)
        self.results[dance.ID] = result
        return result

    def compute_all(self, dances: Dict[int, "Dance"]
                    ) -> Dict[int, ReversalResult]:
        """
        Compute reversals for all dances.

        Parameters
        ----------
        dances : mapping of worm_id → Dance

        Returns
        -------
        dict mapping worm_id → ReversalResult
        """
        for d in dances.values():
            if self.validate_dancer(d):
                self.compute_dancer(d)
        return self.results

    # ------------------------------------------------------------------
    # Quantifier interface (mirrors CustomComputation)
    # ------------------------------------------------------------------

    def quantifier_count(self) -> int:
        """Number of scalar quantities available per dancer."""
        return 6

    def quantifier_title(self, index: int) -> str:
        titles = ["reversal_count", "mean_duration", "sd_duration",
                  "mean_speed", "inter_reversal_interval", "reversal_rate"]
        return titles[index]

    def compute_dancer_quantity(self, dance: "Dance", index: int) -> float:
        """Return scalar quantity *index* for *dance*."""
        if dance.ID not in self.results:
            self.compute_dancer(dance)
        r = self.results[dance.ID]
        vals = [r.count, r.mean_duration, r.sd_duration,
                r.mean_speed, r.inter_reversal_interval, r.rate]
        return float(vals[index])

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def get_reversal_timecourse(self, dance: "Dance") -> np.ndarray:
        """
        Return a (N,) array: +1 during reversal, 0 otherwise.
        Matches Dance time axis.
        """
        q = np.zeros(dance.n_frames, dtype=np.float32)
        revs = self._reversal_map.get(dance.ID, [])
        for rev in revs:
            q[rev.start_index:rev.end_index] = 1.0
        return q

    def get_reversal_durations(self, dance: "Dance") -> np.ndarray:
        """Return array of reversal durations (seconds)."""
        revs = self._reversal_map.get(dance.ID, [])
        return np.array([r.duration for r in revs], dtype=np.float32)

    def to_dataframe(self, results: Optional[Dict[int, ReversalResult]] = None):
        """Convert results to a pandas DataFrame (one row per worm)."""
        import pandas as pd
        if results is None:
            results = self.results
        rows = []
        for wid, r in results.items():
            rows.append({
                'worm_id':           wid,
                'reversal_count':    r.count,
                'mean_duration_s':   r.mean_duration,
                'sd_duration_s':     r.sd_duration,
                'mean_speed_mm_s':   r.mean_speed,
                'IRI_s':             r.inter_reversal_interval,
                'rate_per_s':        r.rate,
            })
        return pd.DataFrame(rows)

    def to_events_dataframe(self):
        """
        Return a DataFrame with one row per reversal event (all worms).
        """
        import pandas as pd
        rows = []
        for wid, revs in self._reversal_map.items():
            for rev in revs:
                rows.append({
                    'worm_id':    wid,
                    'start_time': rev.start_time,
                    'end_time':   rev.end_time,
                    'duration':   rev.duration,
                    'n_frames':   rev.n_frames,
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _summarise(self, dance: "Dance",
                   revs: List[Reversal]) -> ReversalResult:
        r = ReversalResult(dance_id=dance.ID, reversals=revs)
        r.count = len(revs)
        if r.count == 0:
            return r
        durs = np.array([rv.duration for rv in revs])
        r.mean_duration = float(np.mean(durs))
        r.sd_duration   = float(np.std(durs, ddof=1)) if len(durs) > 1 else 0.0

        sp_arr = np.array([
            dance.path_length(rv.start_index, rv.end_index)
            * self.mm_per_pixel / max(rv.duration, 1e-6)
            for rv in revs
        ])
        r.mean_speed = float(np.mean(sp_arr))

        if r.count > 1:
            starts = np.array([rv.start_time for rv in revs])
            ends   = np.array([rv.end_time   for rv in revs[:-1]])
            r.inter_reversal_interval = float(np.mean(starts[1:] - ends))
        else:
            r.inter_reversal_interval = 0.0

        total_t = dance.duration()
        r.rate = r.count / total_t if total_t > 0 else 0.0
        return r
