from __future__ import annotations

import os

import nibabel as nib
import numpy as np


def load_nifti_mask(mask_path: str) -> np.ndarray:
    if not mask_path or not os.path.isfile(mask_path):
        raise RuntimeError("Segmentation mask file was not found.")

    try:
        nifti = nib.load(mask_path)
        return np.asarray(nifti.get_fdata())
    except Exception as exc:
        raise RuntimeError(f"Failed to load coronary mask: {exc}") from exc


def align_mask_to_cta_volume(mask: np.ndarray, cta_shape: tuple[int, ...]) -> np.ndarray:
    aligned = np.asarray(mask)
    aligned = np.squeeze(aligned)

    if len(cta_shape) != 4:
        raise RuntimeError("CTA volume shape is invalid for coronary mask alignment.")

    _, z_slices, height, width = cta_shape
    expected_shape = (z_slices, height, width)

    if aligned.ndim == 4:
        if aligned.shape[0] == 1:
            aligned = aligned[0]
        elif aligned.shape[-1] == 1:
            aligned = aligned[..., 0]

    if aligned.shape == expected_shape:
        return aligned

    if aligned.shape == (height, width, z_slices):
        return np.transpose(aligned, (2, 0, 1))

    raise RuntimeError(
        f"Coronary mask shape mismatch. Expected {expected_shape} or {(height, width, z_slices)}, got {aligned.shape}."
    )
