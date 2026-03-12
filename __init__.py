"""
chore
=====
Python re-implementation of the Choreography (Chore) C. elegans
behavioural analysis library.

Original Java implementation by Rex Kerr (kerr@caltech.edu) and the
Schafer / Rankin labs.

Quick-start
-----------
>>> from chore import Choreography
>>> c = Choreography("path/to/mwt_directory", mm_per_pixel=0.025)
>>> c.load()
>>> print(c)                        # Choreography(n_worms=…)
>>> speeds = c.get_quantity("speed")   # dict: worm_id → np.ndarray
>>> df = c.to_dataframe(["speed", "length", "curve"])
>>> mr = c.run_reversal_analysis()
>>> print(mr.to_dataframe())

Module layout
-------------
chore.choreography   Choreography – main orchestrator
chore.dance          Dance        – per-worm track data object
chore.statistic      Statistic    – descriptive statistics + distributions
chore.fitter         Fitter       – geometric fitting (line / circle / eigen)
chore.spine_outline  SpineData, OutlineData, Fractionator, QuadRanger
chore.io             File I/O (read_summary, read_blob_file, load_directory)
chore.utils          Vec2F/I/D/S, angle helpers, resampling
chore.measures       Analysis plugins:
    .reversal        MeasureReversal  – backward-locomotion detection
    .omega           MeasureOmega     – omega-turn detection
    .eigenspine      Eigenspine       – NIPALS spine PCA
    .curvaceous      Curvaceous       – detailed curvature analysis
    .spatial         Flux, MeasureRadii, Respine, Reoutline,
                     Extract, SpinesForward
"""

__version__ = "1.0.0"
__author__  = "Python port based on Rex Kerr's Choreography (Java)"

# ---------------------------------------------------------------------------
# Top-level exports
# ---------------------------------------------------------------------------

from .choreography   import Choreography, OUTPUT_CODES, STAT_CODES
from .dance          import Dance, ReversalEvent, StyleSegment
from .statistic      import (Statistic,
                              cdf_normal, icdf_normal,
                              cdf_t_stat, icdf_t_stat,
                              cdf_f_stat, icdf_f_stat,
                              cdf_chi_sq, cdf_not_chi_sq,
                              erf, erfc, erf_inv, erfc_inv,
                              lngamma, gamma_func,
                              beta_func, incomplete_beta)
from .fitter         import (Fitter, LineFitter, CircleFitter,
                              SpotFitter, EigenFinder,
                              LinearParameters, CircularParameters,
                              SpotParameters)
from .spine_outline  import (SpineData, OutlineData,
                              Fractionator, QuadRanger)
from .io             import (load_directory, read_summary,
                              read_blob_file, write_dat_file,
                              write_summary_stats)
from .utils          import (Vec2F, Vec2D, Vec2I, Vec2S,
                              signed_angle_v2f, signed_angle_v2s,
                              cumulative_arc_length, resample_polyline,
                              wrap_angle, angle_diff,
                              smooth_array, gaussian_smooth,
                              clip, ring_left, ring_right)

# from .measures import (
#     MeasureReversal, detect_reversals, Reversal, ReversalResult,
#     MeasureOmega, OmegaBend,
#     Eigenspine, nipals, EigenSpineResult,
#     Curvaceous,
#     Flux, FluxShape, FluxEvent,
#     MeasureRadii, RadiiProfile,
#     Respine, Reoutline, Extract, SpinesForward,
# )

from .reversal import MeasureReversal, detect_reversals, Reversal, ReversalResult
from .omega import MeasureOmega, OmegaBend
from .eigenspine import Eigenspine, nipals, EigenSpineResult
from .curvaceous import Curvaceous
from .spatial import MeasureRadii, RadiiProfile, Flux, FluxShape, FluxEvent, Respine, Reoutline, Extract, SpinesForward

from .datamap import (
    DataMapper,
    # Color mappers
    ColorMapper, RainbowMapper, SunsetMapper, SpatterMapper,
    # Backgrounds
    Backgrounder, BlackBackgrounder, GreenGrounder, WhiteGrounder,
    ImageGrounder, DimImageGrounder,
    # Dot painters
    DotPainter, CirclePainter, SpotPainter,
    IdentityPainter, FramePainter, ValuePainter, LinePainter,
    # Value sources
    ValueSource, ValueIdentity, ValueValue,
    # View
    ViewRequest,
)

__all__ = [
    # Core
    "Choreography", "OUTPUT_CODES", "STAT_CODES",
    "Dance", "ReversalEvent", "StyleSegment",
    "Statistic",
    "Fitter", "LineFitter", "CircleFitter", "SpotFitter", "EigenFinder",
    "LinearParameters", "CircularParameters", "SpotParameters",
    "SpineData", "OutlineData", "Fractionator", "QuadRanger",
    # I/O
    "load_directory", "read_summary", "read_blob_file",
    "write_dat_file", "write_summary_stats",
    # Utils
    "Vec2F", "Vec2D", "Vec2I", "Vec2S",
    "signed_angle_v2f", "signed_angle_v2s",
    "cumulative_arc_length", "resample_polyline",
    "wrap_angle", "angle_diff",
    "smooth_array", "gaussian_smooth",
    "clip", "ring_left", "ring_right",
    # Statistics functions
    "cdf_normal", "icdf_normal",
    "cdf_t_stat", "icdf_t_stat",
    "cdf_f_stat", "icdf_f_stat",
    "cdf_chi_sq", "cdf_not_chi_sq",
    "erf", "erfc", "erf_inv", "erfc_inv",
    "lngamma", "gamma_func", "beta_func", "incomplete_beta",
    # Measures
    "MeasureReversal", "detect_reversals", "Reversal", "ReversalResult",
    "MeasureOmega", "OmegaBend",
    "Eigenspine", "nipals", "EigenSpineResult",
    "Curvaceous",
    "Flux", "FluxShape", "FluxEvent",
    "MeasureRadii", "RadiiProfile",
    "Respine", "Reoutline", "Extract", "SpinesForward",
    # DataMapper (--map)
    "DataMapper",
    "ColorMapper", "RainbowMapper", "SunsetMapper", "SpatterMapper",
    "Backgrounder", "BlackBackgrounder", "GreenGrounder", "WhiteGrounder",
    "ImageGrounder", "DimImageGrounder",
    "DotPainter", "CirclePainter", "SpotPainter",
    "IdentityPainter", "FramePainter", "ValuePainter", "LinePainter",
    "ValueSource", "ValueIdentity", "ValueValue",
    "ViewRequest",
]
