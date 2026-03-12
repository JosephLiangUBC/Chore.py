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
read_summary(path)          → dict mapping worm_id → Dance, or pandas.DataFrame
read_blob_file(path)        → Dance
load_directory(path, ...)   → Choreography (fully loaded)
"""

from __future__ import annotations
import os
import re
import zipfile
import warnings
import math
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


def _is_zip_path(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() == '.zip'


def _parse_float_token(token: str) -> float:
    """
    Parse floats including legacy nonstandard tokens occasionally emitted by MWT.

    Examples seen in the wild include ``1.#INF``, ``-1.#INF``, ``1.#QNAN``,
    and ``-1.#IO``. Python's ``float()`` does not accept these forms.
    """
    try:
        return float(token)
    except ValueError:
        upper = token.strip().upper()
        if not upper:
            raise
        sign = -1.0 if upper.startswith("-") else 1.0
        unsigned = upper[1:] if upper[0] in "+-" else upper
        if "#INF" in unsigned:
            return math.copysign(float("inf"), sign)
        if "#IND" in unsigned or "#IO" in unsigned or "#NAN" in unsigned:
            return float("nan")
        raise


# ---------------------------------------------------------------------------
# .summary file reader
# ---------------------------------------------------------------------------

def read_summary(path: Union[str, Path],
                 zipf: Optional[zipfile.ZipFile] = None,
                 zip_entry: Optional[str] = None):
    """
    Parse an MWT ``.summary`` file.

    Returns:
    - a frame-level pandas.DataFrame for the real MWT summary format
    - or a dict mapping worm_id → Dance for the older per-worm summary format
    """
    path = Path(path)

    if zipf is None and zip_entry is None and _is_zip_path(path):
        with zipfile.ZipFile(path) as local_zipf:
            names = [name for name in local_zipf.namelist() if name.endswith('.summary')]
            if not names:
                raise FileNotFoundError(f"No .summary file found in ZIP archive: {path}")
            blob_entries = [name for name in local_zipf.namelist()
                            if name.endswith('.blob') or name.endswith('.blobs')]
            if blob_entries:
                return _read_blob_collection(path, zipf=local_zipf,
                                             blob_entries=sorted(blob_entries))
            return read_summary(path, zipf=local_zipf, zip_entry=sorted(names)[0])

    if zipf is None and zip_entry is None and not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    first_tokens: Optional[List[str]] = None
    with _open_file(path, zipf=zipf, zip_entry=zip_entry) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            first_tokens = line.split()
            break

    if not first_tokens:
        return {}

    try:
        int(first_tokens[0])
        float(first_tokens[1])
        second_is_int = False
        try:
            int(first_tokens[1])
            second_is_int = True
        except ValueError:
            second_is_int = False
        if not second_is_int:
            if len(first_tokens) >= 15:
                return _read_summary_frame_table(path, zipf=zipf, zip_entry=zip_entry)
            return {}
    except ValueError:
        return {}

    # Legacy per-worm summary support.
    raw: Dict[int, Dict[str, list]] = {}

    with _open_file(path, zipf=zipf, zip_entry=zip_entry) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip('\n')
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            tokens = line.split()
            if len(tokens) < SUMMARY_MIN_TOKENS:
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


def _read_summary_frame_table(path: Union[str, Path],
                              zipf: Optional[zipfile.ZipFile] = None,
                              zip_entry: Optional[str] = None):
    """
    Parse an MWT frame-level ``.summary`` file into a pandas DataFrame.

    The returned table has one row per processed image.
    """
    import pandas as pd

    path = Path(path)
    if zipf is None and zip_entry is None and _is_zip_path(path):
        with zipfile.ZipFile(path) as local_zipf:
            names = [name for name in local_zipf.namelist() if name.endswith('.summary')]
            if not names:
                raise FileNotFoundError(f"No .summary file found in ZIP archive: {path}")
            return _read_summary_frame_table(path, zipf=local_zipf, zip_entry=sorted(names)[0])

    if zipf is None and zip_entry is None and not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    rows = []
    with _open_file(path, zipf=zipf, zip_entry=zip_entry) as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            main_text, events_text, ancestry_text, blobref_text = _split_summary_sections(line)
            tokens = main_text.split()
            if len(tokens) < 15:
                warnings.warn(f"Summary line {lineno}: expected >=15 columns, got {len(tokens)}")
                continue
            try:
                row = {
                    "image_number": int(tokens[0]),
                    "time_s": _parse_float_token(tokens[1]),
                    "n_objects": int(tokens[2]),
                    "n_summary_objects": int(tokens[3]),
                    "mean_duration_s": _parse_float_token(tokens[4]),
                    "mean_speed_px_s": _parse_float_token(tokens[5]),
                    "mean_angular_speed_rad_s": _parse_float_token(tokens[6]),
                    "mean_length_px": _parse_float_token(tokens[7]),
                    "mean_rel_length": _parse_float_token(tokens[8]),
                    "mean_width_px": _parse_float_token(tokens[9]),
                    "mean_rel_width": _parse_float_token(tokens[10]),
                    "mean_aspect": _parse_float_token(tokens[11]),
                    "mean_rel_aspect": _parse_float_token(tokens[12]),
                    "mean_end_wiggle_rad": _parse_float_token(tokens[13]),
                    "mean_area_px": _parse_float_token(tokens[14]),
                    "event_flags": _parse_summary_event_flags(events_text),
                    "lineage_pairs": _parse_summary_pairs(ancestry_text),
                    "blob_refs": _parse_summary_blobrefs(blobref_text),
                }
            except ValueError as e:
                warnings.warn(f"Summary line {lineno}: parse error ({e}), skipping.")
                continue
            rows.append(row)

    return pd.DataFrame(rows)


def _split_summary_sections(line: str) -> Tuple[str, str, str, str]:
    """
    Split a summary line into base columns and optional %, %%, %%% sections.
    """
    blobref_text = ""
    ancestry_text = ""
    events_text = ""

    if " %%% " in line:
        line, _, blobref_text = line.partition(" %%% ")
    if " %% " in line:
        line, _, ancestry_text = line.partition(" %% ")
    if " % " in line:
        line, _, events_text = line.partition(" % ")

    return line.strip(), events_text.strip(), ancestry_text.strip(), blobref_text.strip()


def _parse_summary_event_flags(text: str) -> List[int]:
    if not text:
        return []
    flags = []
    for tok in text.split():
        try:
            flags.append(int(tok, 16))
        except ValueError:
            continue
    return flags


def _parse_summary_pairs(text: str) -> List[Tuple[int, int]]:
    if not text:
        return []
    vals = []
    for tok in text.split():
        try:
            vals.append(int(tok))
        except ValueError:
            continue
    return [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]


def _parse_summary_blobrefs(text: str) -> List[Tuple[int, int, int]]:
    if not text:
        return []
    toks = text.split()
    refs = []
    i = 0
    while i + 1 < len(toks):
        try:
            obj_id = int(toks[i])
            file_and_offset = toks[i + 1]
            if "." in file_and_offset:
                file_no_s, offset_s = file_and_offset.split(".", 1)
                file_no = int(file_no_s)
                offset = int(offset_s) if offset_s else 0
            else:
                file_no = int(file_and_offset)
                offset = 0
            refs.append((obj_id, file_no, offset))
        except ValueError:
            pass
        i += 2
    return refs


# ---------------------------------------------------------------------------
# Per-worm .blob file reader
# ---------------------------------------------------------------------------

def read_blob_file(path: Union[str, Path],
                   worm_id: Optional[int] = None,
                   zipf: Optional[zipfile.ZipFile] = None,
                   zip_entry: Optional[str] = None) -> Dance:
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
    blob_name = Path(zip_entry).name if zip_entry is not None else path.name

    if zipf is None and zip_entry is None and _is_zip_path(path):
        with zipfile.ZipFile(path) as local_zipf:
            names = [name for name in local_zipf.namelist()
                     if name.endswith('.blob') or name.endswith('.blobs')]
            if not names:
                raise FileNotFoundError(f"No .blob/.blobs file found in ZIP archive: {path}")
            if len(names) > 1:
                raise ValueError(
                    f"ZIP archive contains multiple .blob/.blobs files; specify zip_entry explicitly: {path}"
                )
            return read_blob_file(path, worm_id=worm_id, zipf=local_zipf, zip_entry=names[0])

    if zipf is None and zip_entry is None and not path.exists():
        raise FileNotFoundError(f"Blob file not found: {path}")

    if blob_name.endswith('.blobs'):
        dances = _read_multiworm_blobs_file(path, zipf=zipf, zip_entry=zip_entry)
        if worm_id is None:
            if len(dances) != 1:
                raise ValueError(
                    f"{blob_name} contains {len(dances)} worms; specify worm_id or use load_directory()."
                )
            return next(iter(dances.values()))
        if worm_id not in dances:
            raise KeyError(f"Worm {worm_id} not found in {blob_name}")
        return dances[worm_id]

    if worm_id is None:
        # Try to infer from filename: e.g. 20100601_120000_00001.blob → 1
        m = re.search(r'_(\d+)\.blob$', blob_name)
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

    with _open_file(path, zipf=zipf, zip_entry=zip_entry) as fh:
        lines = fh.readlines()

    has_legacy_headers = any(
        line.lstrip().startswith('%') and len(line.split()) >= 3
        for line in lines
    )

    if has_legacy_headers:
        current_frame = -1
        current_time = 0.0
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('%'):
                parts = line.split()
                try:
                    current_frame = int(parts[1])
                    current_time = float(parts[2]) if len(parts) > 2 else 0.0
                except (ValueError, IndexError):
                    pass
                continue

            tokens = line.split()
            if len(tokens) < 8:
                continue
            try:
                cx = float(tokens[0])
                cy = float(tokens[1])
                area = float(tokens[2])
                orient = float(tokens[5])
                major = float(tokens[6])
                minor = float(tokens[7])
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
            spine_list.append(None)
            outline_list.append(None)
    else:
        for raw_line in lines:
            line = raw_line.strip()
            rec = _parse_blob_record(line)
            if rec is None:
                continue

            frames_list.append(rec['frame'])
            times_list.append(rec['time'])
            area_list.append(rec['area'])
            cx_list.append(rec['cx'])
            cy_list.append(rec['cy'])
            orient_list.append(float(np.arctan2(rec['bearing'][1], rec['bearing'][0])))
            major_list.append(float(rec['length']) * 0.5)
            minor_list.append(float(rec['width']) * 0.5)
            spine_list.append(rec['spine'])
            outline_list.append(rec['outline'])

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


def _new_dance_raw() -> Dict[str, list]:
    return {
        'frames': [], 'times': [], 'area': [],
        'cx': [], 'cy': [], 'orient': [],
        'major': [], 'minor': [],
        'spine': [], 'outline': [],
    }


def _decode_blob_outline(start_x: int, start_y: int,
                         n_points: int, encoded: str) -> Optional[OutlineData]:
    """Decode the compact 4-connected outline payload used after ``%%``."""
    if n_points <= 0:
        return None
    encoded = encoded.strip()
    x, y = start_x, start_y
    pts = [(x, y)]
    steps_needed = max(0, n_points - 1)
    steps_done = 0
    for ch in encoded:
        v = ord(ch) - 48
        if not (0 <= v < 64):
            return None
        for shift in (0, 2, 4):
            step = (v >> shift) & 0b11
            if step == 0:
                x -= 1
            elif step == 1:
                x += 1
            elif step == 2:
                y -= 1
            else:
                y += 1
            pts.append((x, y))
            steps_done += 1
            if steps_done >= steps_needed:
                return OutlineData(np.asarray(pts, dtype=np.float32), absolute=True)
    return OutlineData(np.asarray(pts, dtype=np.float32), absolute=True)


def _parse_blob_record(line: str) -> Optional[Dict[str, object]]:
    """
    Parse one blob/blobs record line according to the MWT blob format.
    """
    line = line.strip()
    if not line or line.startswith('#') or line.startswith('%'):
        return None

    main_and_spine, has_outline_sep, outline_text = line.partition(' %% ')
    main_text, has_spine_sep, spine_text = main_and_spine.partition(' % ')
    tokens = main_text.split()
    if len(tokens) < 10:
        return None

    try:
        frame = int(tokens[0])
        t = float(tokens[1])
        cx = float(tokens[2])
        cy = float(tokens[3])
        area = float(tokens[4])
        bx = float(tokens[5])
        by = float(tokens[6])
        orth_sd = float(tokens[7])
        length = float(tokens[8])
        width = float(tokens[9])
    except ValueError:
        return None

    bearing = np.array([bx, by], dtype=np.float32)
    norm = float(np.hypot(bearing[0], bearing[1]))
    if norm > 0:
        bearing /= norm

    spine = _parse_relative_spine(spine_text) if has_spine_sep else None

    outline = None
    if has_outline_sep:
        parts = outline_text.split(maxsplit=3)
        if len(parts) >= 4:
            try:
                out_x = int(parts[0])
                out_y = int(parts[1])
                out_n = int(parts[2])
            except ValueError:
                out_x = out_y = out_n = 0
            if out_n > 0:
                outline = _decode_blob_outline(out_x, out_y, out_n, parts[3])

    return {
        'frame': frame,
        'time': t,
        'cx': cx,
        'cy': cy,
        'area': area,
        'bearing': bearing,
        'orth_sd': orth_sd,
        'length': length,
        'width': width,
        'spine': spine,
        'outline': outline,
    }


def _finalize_dances(raw: Dict[int, Dict[str, list]]) -> Dict[int, Dance]:
    dances: Dict[int, Dance] = {}
    for wid, d in raw.items():
        dance = Dance(worm_id=wid)
        n = len(d['frames'])
        dance.frames = np.array(d['frames'], dtype=np.int32)
        dance.times = np.array(d['times'], dtype=np.float32)
        dance.area = np.array(d['area'], dtype=np.float32)
        dance.centroid = np.column_stack([
            np.array(d['cx'], dtype=np.float32),
            np.array(d['cy'], dtype=np.float32),
        ])
        orient = np.array(d['orient'], dtype=np.float32)
        dance.bearing = np.column_stack([np.cos(orient), np.sin(orient)]).astype(np.float32)
        dance.extent = np.column_stack([
            np.array(d['major'], dtype=np.float32),
            np.array(d['minor'], dtype=np.float32),
        ]).astype(np.float32)
        dance.spine = list(d['spine'])
        dance.outline = list(d['outline'])
        if len(dance.spine) < n:
            dance.spine.extend([None] * (n - len(dance.spine)))
        if len(dance.outline) < n:
            dance.outline.extend([None] * (n - len(dance.outline)))
        dances[wid] = dance
    return dances


def _parse_relative_spine(spine_text: str) -> Optional[SpineData]:
    if not spine_text:
        return None
    tokens = spine_text.split()
    if len(tokens) < 4 or len(tokens) % 2 != 0:
        return None
    pts = np.zeros((len(tokens) // 2, 2), dtype=np.float32)
    widths = np.zeros(len(tokens) // 2, dtype=np.float32)
    for i in range(0, len(tokens), 2):
        j = i // 2
        pts[j, 0] = float(tokens[i])
        pts[j, 1] = float(tokens[i + 1])
    return SpineData(pts, widths, absolute=False)


def _read_multiworm_blobs_file(path: Union[str, Path],
                               zipf: Optional[zipfile.ZipFile] = None,
                               zip_entry: Optional[str] = None) -> Dict[int, Dance]:
    raw: Dict[int, Dict[str, list]] = {}
    current_worm_id: Optional[int] = None

    with _open_file(path, zipf=zipf, zip_entry=zip_entry) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('%') and not line.startswith('%%'):
                parts = line.split()
                if len(parts) == 2:
                    try:
                        current_worm_id = int(parts[1])
                        raw.setdefault(current_worm_id, _new_dance_raw())
                        continue
                    except ValueError:
                        pass
            if current_worm_id is None:
                continue
            rec = _parse_blob_record(line)
            if rec is None:
                continue

            d = raw[current_worm_id]
            d['frames'].append(rec['frame'])
            d['times'].append(rec['time'])
            d['cx'].append(rec['cx'])
            d['cy'].append(rec['cy'])
            d['area'].append(rec['area'])
            d['orient'].append(float(np.arctan2(rec['bearing'][1], rec['bearing'][0])))
            d['major'].append(float(rec['length']) * 0.5)
            d['minor'].append(float(rec['width']) * 0.5)
            d['spine'].append(rec['spine'])
            d['outline'].append(rec['outline'])

    return _finalize_dances(raw)


def _merge_dances(target: Dict[int, Dance], source: Dict[int, Dance]) -> None:
    for wid, dance in source.items():
        if wid in target:
            _merge_spine_outline(target[wid], dance)
        else:
            target[wid] = dance


def _read_blob_collection(path: Union[str, Path],
                          zipf: Optional[zipfile.ZipFile] = None,
                          blob_entries: Optional[List[Union[str, Path]]] = None) -> Dict[int, Dance]:
    dances: Dict[int, Dance] = {}
    if blob_entries is None:
        return dances
    for entry in blob_entries:
        entry_name = str(entry)
        if entry_name.endswith('.blobs'):
            _merge_dances(
                dances,
                _read_multiworm_blobs_file(path, zipf=zipf, zip_entry=entry_name),
            )
        else:
            dance = read_blob_file(path, zipf=zipf, zip_entry=entry_name)
            dances[dance.ID] = dance
    return dances


# ---------------------------------------------------------------------------
# Directory / ZIP loader
# ---------------------------------------------------------------------------

def find_mwt_files(directory: Union[str, Path]
                   ) -> Tuple[Optional[Union[Path, str]], List[Union[Path, str]]]:
    """
    Locate the .summary file and all .blob/.blobs files in an MWT output directory.

    Returns (summary_path, blob_paths).
    """
    directory = Path(directory)

    # Support ZIP archives
    if directory.suffix.lower() == '.zip':
        with zipfile.ZipFile(directory) as zf:
            names = sorted(
                name for name in zf.namelist()
                if not name.endswith('/')
            )
        summary = next((name for name in names if name.endswith('.summary')), None)
        blobs = [name for name in names if name.endswith('.blob') or name.endswith('.blobs')]
        return summary, blobs

    summary_candidates = list(directory.glob('*.summary'))
    if not summary_candidates:
        warnings.warn(f"No .summary file found in {directory}")
        summary_path = None
    else:
        if len(summary_candidates) > 1:
            warnings.warn(f"Multiple .summary files found; using "
                          f"{summary_candidates[0]}")
        summary_path = summary_candidates[0]

    blob_paths = sorted(list(directory.glob('*.blob')) + list(directory.glob('*.blobs')))
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
    is_zip = _is_zip_path(directory)

    dances: Dict[int, Dance] = {}

    if is_zip:
        with zipfile.ZipFile(directory) as zf:
            if summary_path is not None:
                if not quiet:
                    print(f"Reading summary from ZIP: {summary_path}")
                summary_data = read_summary(directory, zipf=zf, zip_entry=str(summary_path))
                dances = summary_data if isinstance(summary_data, dict) else {}

            if load_blobs and blob_paths:
                if not quiet:
                    print(f"Reading {len(blob_paths)} blob file(s) from ZIP…")
                try:
                    _merge_dances(
                        dances,
                        _read_blob_collection(directory, zipf=zf, blob_entries=blob_paths),
                    )
                except Exception as e:
                    warnings.warn(f"Error reading blob collection from {directory}: {e}")
    else:
        # Primary source: .summary file
        if summary_path is not None:
            if not quiet:
                print(f"Reading summary: {summary_path}")
            summary_data = read_summary(summary_path)
            dances = summary_data if isinstance(summary_data, dict) else {}

        # Augment with spine/outline data from .blob files
        if load_blobs and blob_paths:
            if not quiet:
                print(f"Reading {len(blob_paths)} blob file(s)…")
            for bp in blob_paths:
                try:
                    if str(bp).endswith('.blobs'):
                        _merge_dances(dances, _read_multiworm_blobs_file(bp))
                    else:
                        d = read_blob_file(bp)
                        if d.ID in dances:
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
