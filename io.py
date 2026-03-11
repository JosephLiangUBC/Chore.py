"""
io.py
-----
File I/O for Multi-Worm Tracker (MWT) data files.

Supported formats
-----------------
*.summary  – plain-text per-frame summary of all blobs
*.blob     – per-worm file with frame-by-frame morphology and optional
             spine / outline data (text-based, read by Dance.readInputStream)
*.zip      – ZIP archive containing the above files (Choreography also
             reads from ZIP)

Public API
----------
read_summary(path)          → dict mapping worm_id → Dance
read_blob_file(path)        → Dance
load_directory(path, ...)   → Choreography (fully loaded)
"""

from __future__ import annotations
import os
import re
import zipfile
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, IO

from .dance import Dance
from .spine_outline import SpineData, OutlineData
from .utils import Vec2F


# ---------------------------------------------------------------------------
# Constants / parsing helpers
# ---------------------------------------------------------------------------

# Default MWT pixel-to-mm conversion (Choreography default is 0.025 mm/px)
DEFAULT_MM_PER_PIXEL: float = 0.025

# Minimum number of tokens expected on a summary line
SUMMARY_MIN_TOKENS = 11


def _open_file(path: Union[str, Path],
               zipf: Optional[zipfile.ZipFile] = None,
               zip_entry: Optional[str] = None
               ) -> IO:
    """Open a file either from disk or from within a ZIP archive."""
    if zipf is not None and zip_entry is not None:
        import io
        return io.TextIOWrapper(zipf.open(zip_entry), encoding='utf-8',
                                errors='replace')
    return open(path, 'r', encoding='utf-8', errors='replace')


# ---------------------------------------------------------------------------
# .summary file reader
# ---------------------------------------------------------------------------

def read_summary(path: Union[str, Path]) -> Dict[int, Dance]:
    """
    Parse an MWT .summary file and return a dict mapping worm ID → Dance.

    .summary format (whitespace-delimited, one line per worm per frame)::

        <id>  <frame>  <time>  <x>  <y>  <area>  <pob>  <perim>
            <orient>  <major>  <minor>  [<eccentricity>]

    Fields
    ------
    id          : worm identifier (integer)
    frame       : frame number
    time        : time in seconds
    x, y        : centroid in pixels
    area        : body area in pixels²
    pob         : pixels on border (used for quality filtering)
    perim       : perimeter in pixels
    orient      : major-axis orientation in radians
    major       : major half-axis (pixels) → body length ≈ 2*major
    minor       : minor half-axis (pixels) → body width ≈ 2*minor
    eccentricity: (optional) shape eccentricity

    Lines beginning with ``#`` or ``%`` are treated as comments/headers.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    # temporary storage: worm_id → lists of per-frame values
    raw: Dict[int, Dict[str, list]] = {}

    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip('\n')
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            tokens = line.split()
            if len(tokens) < SUMMARY_MIN_TOKENS:
                warnings.warn(f"Line {lineno}: too few tokens ({len(tokens)}), "
                              f"skipping.")
                continue
            try:
                wid     = int(tokens[0])
                frame   = int(tokens[1])
                t       = float(tokens[2])
                x       = float(tokens[3])
                y       = float(tokens[4])
                area    = float(tokens[5])
                # tokens[6] = pixels on border (pob)
                # tokens[7] = perimeter
                orient  = float(tokens[8])
                major   = float(tokens[9])
                minor   = float(tokens[10])
            except (ValueError, IndexError) as e:
                warnings.warn(f"Line {lineno}: parse error ({e}), skipping.")
                continue

            if wid not in raw:
                raw[wid] = {
                    'frames': [], 'times': [], 'area': [],
                    'cx': [], 'cy': [], 'orient': [],
                    'major': [], 'minor': [],
                }
            d = raw[wid]
            d['frames'].append(frame)
            d['times'].append(t)
            d['area'].append(area)
            d['cx'].append(x)
            d['cy'].append(y)
            d['orient'].append(orient)
            d['major'].append(major)
            d['minor'].append(minor)

    # Build Dance objects
    dances: Dict[int, Dance] = {}
    for wid, d in raw.items():
        dance = Dance(worm_id=wid)
        n = len(d['frames'])
        dance.frames   = np.array(d['frames'], dtype=np.int32)
        dance.times    = np.array(d['times'],  dtype=np.float32)
        dance.area     = np.array(d['area'],   dtype=np.float32)
        cx = np.array(d['cx'], dtype=np.float32)
        cy = np.array(d['cy'], dtype=np.float32)
        dance.centroid = np.column_stack([cx, cy])
        orient = np.array(d['orient'], dtype=np.float32)
        dance.bearing  = np.column_stack([
            np.cos(orient), np.sin(orient)
        ]).astype(np.float32)
        major = np.array(d['major'], dtype=np.float32)
        minor = np.array(d['minor'], dtype=np.float32)
        dance.extent   = np.column_stack([major, minor]).astype(np.float32)
        # Initialise empty spine / outline lists
        dance.spine   = [None] * n
        dance.outline = [None] * n
        dances[wid] = dance

    return dances


# ---------------------------------------------------------------------------
# Per-worm .blob file reader
# ---------------------------------------------------------------------------

def read_blob_file(path: Union[str, Path],
                   worm_id: Optional[int] = None
                   ) -> Dance:
    """
    Parse a single MWT .blob file and return a Dance object.

    .blob format
    ------------
    Each frame block starts with a header line::

        % <frame_index> <time>

    Followed by one data line per frame containing::

        <cx> <cy> <area> <pob> <perimeter> <orient> <major> <minor>
            [<spine_n> <sx0> <sy0> <w0> <sx1> <sy1> <w1> … ]
            [! <outline_n> <ox0> <oy0> <ox1> <oy1> … ]

    Where spine data is prefixed by the count, and outline by ``!``.
    Lines starting with ``#`` are comments.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Blob file not found: {path}")

    if worm_id is None:
        # Try to infer from filename: e.g. 20100601_120000_00001.blob → 1
        m = re.search(r'_(\d+)\.blob$', path.name)
        worm_id = int(m.group(1)) if m else -1

    dance = Dance(worm_id=worm_id)

    frames_list:  List[int]          = []
    times_list:   List[float]        = []
    area_list:    List[float]        = []
    cx_list:      List[float]        = []
    cy_list:      List[float]        = []
    orient_list:  List[float]        = []
    major_list:   List[float]        = []
    minor_list:   List[float]        = []
    spine_list:   List[Optional[SpineData]]   = []
    outline_list: List[Optional[OutlineData]] = []

    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        current_frame = -1
        current_time  = 0.0
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('%'):
                # Frame header
                parts = line.split()
                try:
                    current_frame = int(parts[1])
                    current_time  = float(parts[2]) if len(parts) > 2 else 0.0
                except (ValueError, IndexError):
                    pass
                continue

            # Data line
            tokens = line.split()
            if len(tokens) < 8:
                continue
            try:
                cx    = float(tokens[0])
                cy    = float(tokens[1])
                area  = float(tokens[2])
                # tokens[3] = pob, tokens[4] = perimeter
                orient = float(tokens[5])
                major  = float(tokens[6])
                minor  = float(tokens[7])
            except (ValueError, IndexError):
                continue

            frames_list.append(current_frame)
            times_list.append(current_time)
            area_list.append(area)
            cx_list.append(cx)
            cy_list.append(cy)
            orient_list.append(orient)
            major_list.append(major)
            minor_list.append(minor)

            # Optional spine
            sp_data:  Optional[SpineData]   = None
            out_data: Optional[OutlineData] = None

            idx = 8
            if idx < len(tokens):
                try:
                    sp_n = int(tokens[idx]); idx += 1
                    if sp_n > 0 and idx + sp_n * 3 <= len(tokens):
                        sp_pts = np.zeros((sp_n, 2), dtype=np.float32)
                        sp_ws  = np.zeros(sp_n, dtype=np.float32)
                        for k in range(sp_n):
                            sp_pts[k, 0] = float(tokens[idx]);     idx += 1
                            sp_pts[k, 1] = float(tokens[idx]);     idx += 1
                            sp_ws[k]     = float(tokens[idx]);     idx += 1
                        sp_data = SpineData(sp_pts, sp_ws)
                except (ValueError, IndexError):
                    pass

            # Optional outline (prefixed by '!')
            if idx < len(tokens) and tokens[idx] == '!':
                idx += 1
                try:
                    out_n = int(tokens[idx]); idx += 1
                    if out_n > 0 and idx + out_n * 2 <= len(tokens):
                        out_pts = np.zeros((out_n, 2), dtype=np.float32)
                        for k in range(out_n):
                            out_pts[k, 0] = float(tokens[idx]); idx += 1
                            out_pts[k, 1] = float(tokens[idx]); idx += 1
                        out_data = OutlineData(out_pts)
                except (ValueError, IndexError):
                    pass

            spine_list.append(sp_data)
            outline_list.append(out_data)

    n = len(frames_list)
    if n == 0:
        return dance

    dance.frames   = np.array(frames_list, dtype=np.int32)
    dance.times    = np.array(times_list,  dtype=np.float32)
    dance.area     = np.array(area_list,   dtype=np.float32)
    cx = np.array(cx_list, dtype=np.float32)
    cy = np.array(cy_list, dtype=np.float32)
    dance.centroid = np.column_stack([cx, cy])
    orient = np.array(orient_list, dtype=np.float32)
    dance.bearing  = np.column_stack([
        np.cos(orient), np.sin(orient)
    ]).astype(np.float32)
    major = np.array(major_list, dtype=np.float32)
    minor = np.array(minor_list, dtype=np.float32)
    dance.extent   = np.column_stack([major, minor]).astype(np.float32)
    dance.spine    = spine_list
    dance.outline  = outline_list

    return dance


# ---------------------------------------------------------------------------
# Directory / ZIP loader
# ---------------------------------------------------------------------------

def find_mwt_files(directory: Union[str, Path]
                   ) -> Tuple[Optional[Path], List[Path]]:
    """
    Locate the .summary file and all .blob files in an MWT output directory.

    Returns (summary_path, blob_paths).
    """
    directory = Path(directory)

    # Support ZIP archives
    if directory.suffix.lower() == '.zip':
        with zipfile.ZipFile(directory) as zf:
            names = zf.namelist()
        summary = next((n for n in names if n.endswith('.summary')), None)
        blobs   = [n for n in names if n.endswith('.blob')]
        return (directory / summary if summary else None,
                [directory / b for b in blobs])

    summary_candidates = list(directory.glob('*.summary'))
    if not summary_candidates:
        warnings.warn(f"No .summary file found in {directory}")
        summary_path = None
    else:
        if len(summary_candidates) > 1:
            warnings.warn(f"Multiple .summary files found; using "
                          f"{summary_candidates[0]}")
        summary_path = summary_candidates[0]

    blob_paths = sorted(directory.glob('*.blob'))
    return summary_path, blob_paths


def load_directory(directory: Union[str, Path],
                   mm_per_pixel: float = DEFAULT_MM_PER_PIXEL,
                   min_time: float = 0.0,
                   max_time: float = float('inf'),
                   load_blobs: bool = True,
                   compute_statistics: bool = True,
                   quiet: bool = False
                   ) -> Dict[int, Dance]:
    """
    Load a complete MWT output directory (or ZIP archive).

    Parameters
    ----------
    directory         : path to the MWT output folder or .zip
    mm_per_pixel      : spatial calibration (mm per pixel)
    min_time, max_time: time window filter (seconds)
    load_blobs        : if True, also read individual .blob files for
                        spine/outline data
    compute_statistics: if True, call dance.calc_basic_statistics() on each
    quiet             : suppress progress messages

    Returns
    -------
    dict mapping worm_id → Dance
    """
    directory = Path(directory)
    summary_path, blob_paths = find_mwt_files(directory)

    dances: Dict[int, Dance] = {}

    # Primary source: .summary file
    if summary_path is not None:
        if not quiet:
            print(f"Reading summary: {summary_path}")
        dances = read_summary(summary_path)

    # Augment with spine/outline data from .blob files
    if load_blobs and blob_paths:
        if not quiet:
            print(f"Reading {len(blob_paths)} blob file(s)…")
        for bp in blob_paths:
            try:
                d = read_blob_file(bp)
                if d.ID in dances:
                    # Merge spine/outline into existing Dance
                    existing = dances[d.ID]
                    _merge_spine_outline(existing, d)
                else:
                    dances[d.ID] = d
            except Exception as e:
                warnings.warn(f"Error reading {bp}: {e}")

    # Time filtering
    if min_time > 0 or max_time < float('inf'):
        dances = {
            wid: _trim_time(d, min_time, max_time)
            for wid, d in dances.items()
            if len(d.times) > 0
        }

    # Attach mm_per_pixel for downstream use
    for d in dances.values():
        d._mm_per_pixel = mm_per_pixel

    # Basic statistics
    if compute_statistics:
        for d in dances.values():
            d.calc_basic_statistics()
            d.find_segmentation()

    if not quiet:
        print(f"Loaded {len(dances)} worms.")

    return dances


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _merge_spine_outline(target: Dance, source: Dance) -> None:
    """Copy spine/outline from *source* into *target* by matching frame index."""
    if len(source.frames) == 0:
        return
    frame_to_idx = {int(f): i for i, f in enumerate(target.frames)}
    for i, frame in enumerate(source.frames):
        if int(frame) in frame_to_idx:
            ti = frame_to_idx[int(frame)]
            if ti < len(target.spine) and i < len(source.spine):
                if source.spine[i] is not None:
                    target.spine[ti] = source.spine[i]
            if ti < len(target.outline) and i < len(source.outline):
                if source.outline[i] is not None:
                    target.outline[ti] = source.outline[i]


def _trim_time(d: Dance, t0: float, t1: float) -> Dance:
    """Return a new Dance with frames restricted to [t0, t1]."""
    mask = (d.times >= t0) & (d.times <= t1)
    if not np.any(mask):
        return d
    nd = Dance(worm_id=d.ID)
    nd.times    = d.times[mask]
    nd.frames   = d.frames[mask]
    nd.area     = d.area[mask]
    nd.centroid = d.centroid[mask]
    nd.bearing  = d.bearing[mask]
    nd.extent   = d.extent[mask]
    if len(d.circles):
        nd.circles = d.circles[mask]
    indices = np.where(mask)[0]
    nd.spine   = [d.spine[i]   for i in indices if i < len(d.spine)]
    nd.outline = [d.outline[i] for i in indices if i < len(d.outline)]
    return nd


# ---------------------------------------------------------------------------
# Output: write derived data to text files (mirrors Choreography output)
# ---------------------------------------------------------------------------

def write_dat_file(path: Union[str, Path],
                   dance: Dance,
                   quantity: np.ndarray,
                   stat: str = "%",
                   header: bool = True) -> None:
    """
    Write a single derived quantity to a Choreography-style .dat file.

    Parameters
    ----------
    path     : output file path
    dance    : the Dance whose data is being written
    quantity : 1-D array of the computed metric
    stat     : statistic specifier ("%" = mean, "^" = sd, etc.)
    header   : if True, write a comment header line
    """
    path = Path(path)
    with open(path, 'w') as fh:
        if header:
            fh.write(f"# worm_id={dance.ID} stat={stat}\n")
        for t, v in zip(dance.times, quantity):
            fh.write(f"{t:.4f}\t{v:.6g}\n")


def write_summary_stats(path: Union[str, Path],
                        dances: Dict[int, Dance],
                        quantities: Dict[str, np.ndarray],
                        header: bool = True) -> None:
    """
    Write a tab-delimited summary statistics file (one row per worm).

    Parameters
    ----------
    path       : output path
    dances     : worm id → Dance mapping
    quantities : metric_name → array (same order as dances.values())
    """
    import pandas as pd
    rows = []
    for wid, d in dances.items():
        row = {'worm_id': wid,
               'duration': d.duration(),
               'n_frames': d.n_frames}
        for metric, arr in quantities.items():
            finite = arr[np.isfinite(arr)]
            if len(finite):
                row[f"{metric}_mean"] = float(np.mean(finite))
                row[f"{metric}_sd"]   = float(np.std(finite, ddof=1)) \
                    if len(finite) > 1 else 0.0
                row[f"{metric}_median"] = float(np.median(finite))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, sep='\t', index=False)
