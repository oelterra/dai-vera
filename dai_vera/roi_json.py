"""
roi_json.py
-----------
Python translation of MATLAB saveRoiAsJson / generateRoiJson.
No GUI imports.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional


def save_roi_as_json(
    roi_bundle: dict,
    output_dir: Optional[str] = None,
    filename:   Optional[str] = None,
) -> str:
    """
    Serialise the full ROI bundle (pre + post) to JSON.

    Matches MATLAB saveRoiAsJson:
        - Default output dir: ~/tmp  (MATLAB used C:\\tmp)
        - Default filename:   timestamped  e.g. roi_20260311_143022.json
        - Converts any numpy arrays to plain lists.

    Parameters
    ----------
    roi_bundle : dict with keys 'preRoiObject' and 'postRoiObject'
    output_dir : override output directory
    filename   : override filename (without extension)

    Returns
    -------
    Absolute path of the written file.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "tmp")
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"roi_{timestamp}"

    path = os.path.join(output_dir, f"{filename}.json")

    # make JSON-serialisable (numpy arrays → lists, numpy scalars → float/int)
    safe = _make_serialisable(roi_bundle)

    with open(path, "w") as f:
        json.dump(safe, f, indent=2)

    return path


def _make_serialisable(obj):
    """Recursively convert numpy types to plain Python types."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj