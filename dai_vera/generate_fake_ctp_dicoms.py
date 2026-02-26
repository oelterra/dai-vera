import os
import math
import uuid
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    generate_uid,
    CTImageStorage,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _dt_to_dicoms(dt: datetime) -> Tuple[str, str]:
    # DICOM DA: YYYYMMDD, TM: HHMMSS.frac (we’ll keep HHMMSS)
    da = dt.strftime("%Y%m%d")
    tm = dt.strftime("%H%M%S")
    return da, tm


def generate_fake_ctp(
    out_root: str,
    times: int = 20,
    slices: int = 24,
    shape: Tuple[int, int] = (256, 256),
    z_spacing_mm: float = 5.0,
    time_step_ms: int = 1000,
    seed: int = 7,
) -> str:
    """
    Generates a fake CT perfusion dataset:
      out_root/
        CTP/
          time_000/
            slice_000.dcm
            slice_001.dcm
            ...
          time_001/
            ...
    """
    rng = np.random.default_rng(seed)

    out_dir = os.path.join(out_root, "CTP")
    _ensure_dir(out_dir)

    # Shared IDs across whole “study”
    study_uid = generate_uid()
    frame_of_ref_uid = generate_uid()

    # Patient/study metadata (dummy)
    patient_id = "DAIVERA_TEST"
    patient_name = "DAIVERA^SYNTHETIC"
    accession = "000000"
    study_desc = "Synthetic CTP Study"
    series_desc_base = "Synthetic CTP Timepoint"

    base_dt = datetime.now()

    rows, cols = shape
    pixel_spacing = [0.8, 0.8]  # mm

    # We'll create 1 series per timepoint (common for dynamic series)
    for t in range(times):
        series_uid = generate_uid()
        series_number = 100 + t

        tp_dir = os.path.join(out_dir, f"time_{t:03d}")
        _ensure_dir(tp_dir)

        # Time tags
        trigger_time = float(t * time_step_ms)  # ms
        series_dt = base_dt + timedelta(milliseconds=t * time_step_ms)
        study_date, series_time = _dt_to_dicoms(series_dt)

        for z in range(slices):
            # File meta
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = CTImageStorage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            file_meta.ImplementationClassUID = generate_uid()

            # Create dataset
            ds = FileDataset(
                filename_or_obj="",
                dataset={},
                file_meta=file_meta,
                preamble=b"\0" * 128,
            )

            # --- Required-ish DICOM fields ---
            ds.SOPClassUID = CTImageStorage
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.FrameOfReferenceUID = frame_of_ref_uid

            ds.PatientName = patient_name
            ds.PatientID = patient_id
            ds.AccessionNumber = accession

            ds.StudyDescription = study_desc
            ds.SeriesDescription = f"{series_desc_base} {t:03d}"

            ds.Modality = "CT"
            ds.StudyDate = study_date
            ds.StudyTime = series_time
            ds.SeriesDate = study_date
            ds.SeriesTime = series_time
            ds.ContentDate = study_date
            ds.ContentTime = series_time

            ds.SeriesNumber = series_number
            ds.InstanceNumber = z + 1  # within series

            # --- Geometry / sorting helpers ---
            ds.ImagePositionPatient = [0.0, 0.0, float(z) * z_spacing_mm]
            ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            ds.PixelSpacing = pixel_spacing
            ds.SliceThickness = z_spacing_mm

            # --- Perfusion-ish time tags your loader can use ---
            ds.TemporalPositionIdentifier = int(t + 1)  # 1-based
            ds.TriggerTime = trigger_time  # ms
            ds.AcquisitionTime = series_time  # HHMMSS
            ds.ContentTime = series_time

            # --- Pixel data ---
            ds.Rows = rows
            ds.Columns = cols
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1  # signed

            # Make a fake “perfusion signal” that changes over time
            # Baseline noise + a time-varying Gaussian "contrast bolus" center
            yy, xx = np.mgrid[0:rows, 0:cols]
            cx = cols * 0.5 + math.sin(t / 3) * 10 + rng.normal(0, 0.5)
            cy = rows * 0.5 + math.cos(t / 4) * 10 + rng.normal(0, 0.5)
            sigma = 25.0
            bolus = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))

            # time amplitude: rises then falls
            amp = 800.0 * math.exp(-((t - times * 0.35) ** 2) / (2 * (times * 0.12) ** 2))
            base = -1000.0  # like HU air baseline
            tissue = 40.0   # soft tissue HU-ish
            noise = rng.normal(0, 12, size=(rows, cols))

            img = base + tissue + amp * bolus + noise

            # add slight slice-dependent bias
            img += (z - slices / 2) * 0.8

            img_i16 = np.clip(img, -2048, 3071).astype(np.int16)

            # Rescale to HU is already in values; keep slope/intercept standard
            ds.RescaleIntercept = 0
            ds.RescaleSlope = 1
            ds.PixelData = img_i16.tobytes()

            # Write file
            out_path = os.path.join(tp_dir, f"slice_{z:03d}.dcm")
            ds.save_as(out_path, write_like_original=False)

    return out_dir


if __name__ == "__main__":
    # Change this to where you want the data generated
    output_root = os.path.abspath("./synthetic_dicom_output")
    out_folder = generate_fake_ctp(
        out_root=output_root,
        times=20,
        slices=24,
        shape=(256, 256),
        z_spacing_mm=5.0,
        time_step_ms=1000,
        seed=7,
    )
    print("✅ Synthetic CTP DICOMs generated here:")
    print(out_folder)