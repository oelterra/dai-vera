from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Sequence


def _format_process_error(command: Sequence[str], result: subprocess.CompletedProcess[str]) -> RuntimeError:
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    details = stderr or stdout or "Unknown error."
    command_text = " ".join(command)
    return RuntimeError(f"Failed to run coronary segmentation.\nCommand: {command_text}\nDetails: {details}")


def run_coronary_segmentation(cta_folder_path: str, output_dir: str | None = None) -> str:
    if not cta_folder_path or not os.path.isdir(cta_folder_path):
        raise RuntimeError("CTA folder not found. Load a valid CTA folder before running segmentation.")

    resolved_output_dir = output_dir or os.path.join(cta_folder_path, "seg_output")
    os.makedirs(resolved_output_dir, exist_ok=True)

    direct_cli = shutil.which("TotalSegmentator")
    if direct_cli:
        command = [direct_cli, "-i", cta_folder_path, "-o", resolved_output_dir, "-ta", "coronary_arteries"]
    else:
        command = [sys.executable, "-m", "totalsegmentator", "-i", cta_folder_path, "-o", resolved_output_dir, "-ta", "coronary_arteries"]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "TotalSegmentator was not found. Install it and make sure it is available in your current Python environment or PATH."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to run coronary segmentation: {exc}") from exc

    if result.returncode != 0:
        raise _format_process_error(command, result)

    mask_path = os.path.join(resolved_output_dir, "coronary_arteries.nii.gz")
    if not os.path.isfile(mask_path):
        raise RuntimeError("Coronary segmentation finished, but coronary_arteries.nii.gz was not found in the output folder.")

    return mask_path
