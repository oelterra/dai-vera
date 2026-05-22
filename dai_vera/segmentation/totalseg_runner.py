from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence

import nibabel as nib
import numpy as np


def _format_process_error(command: Sequence[str], result: subprocess.CompletedProcess[str]) -> RuntimeError:
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    details = stderr or stdout or "Unknown error."
    command_text = " ".join(command)
    return RuntimeError(f"Failed to run coronary segmentation.\nCommand: {command_text}\nDetails: {details}")


def _resolve_base_command() -> list[str]:
    direct_cli = shutil.which("TotalSegmentator")
    if direct_cli:
        return [direct_cli]
    return [sys.executable, "-m", "totalsegmentator"]


def _run_totalseg_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
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


def _should_retry_with_nifti(result: subprocess.CompletedProcess[str]) -> bool:
    details = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
    retry_markers = (
        "dicom2nifti",
        "indexerror: list index out of range",
        "are_imaging_dicoms",
        "dicom_series_to_nifti",
        "no dicom",
    )
    return any(marker in details for marker in retry_markers)


def _write_cta_volume_nifti(cta_volume: dict, output_path: str) -> None:
    pixels = np.asarray(cta_volume.get("pixels"))
    if pixels.ndim != 4:
        raise RuntimeError(f"CTA volume is expected to be 4D, got shape {pixels.shape}.")
    if pixels.shape[0] < 1:
        raise RuntimeError("CTA volume does not contain any time points to export.")

    volume_3d = np.asarray(pixels[0], dtype=np.float32)
    nifti_data = np.transpose(volume_3d, (1, 2, 0))
    nib.save(nib.Nifti1Image(nifti_data, affine=np.eye(4, dtype=np.float32)), output_path)


def run_coronary_segmentation(
    cta_folder_path: str,
    output_dir: str | None = None,
    cta_volume: dict | None = None,
) -> str:
    if not cta_folder_path or not os.path.isdir(cta_folder_path):
        raise RuntimeError("CTA folder not found. Load a valid CTA folder before running segmentation.")

    resolved_output_dir = output_dir or os.path.join(cta_folder_path, "seg_output")
    os.makedirs(resolved_output_dir, exist_ok=True)
    base_command = _resolve_base_command()
    command = [*base_command, "-i", cta_folder_path, "-o", resolved_output_dir, "-ta", "coronary_arteries"]

    result = _run_totalseg_command(command)
    if result.returncode != 0 and cta_volume is not None and _should_retry_with_nifti(result):
        with tempfile.TemporaryDirectory(prefix="dai_vera_cta_") as temp_dir:
            nifti_input_path = os.path.join(temp_dir, "cta_volume.nii.gz")
            _write_cta_volume_nifti(cta_volume, nifti_input_path)
            fallback_command = [*base_command, "-i", nifti_input_path, "-o", resolved_output_dir, "-ta", "coronary_arteries"]
            fallback_result = _run_totalseg_command(fallback_command)
            if fallback_result.returncode != 0:
                raise _format_process_error(fallback_command, fallback_result)
            result = fallback_result

    if result.returncode != 0:
        raise _format_process_error(command, result)

    mask_path = os.path.join(resolved_output_dir, "coronary_arteries.nii.gz")
    if not os.path.isfile(mask_path):
        raise RuntimeError("Coronary segmentation finished, but coronary_arteries.nii.gz was not found in the output folder.")

    return mask_path
