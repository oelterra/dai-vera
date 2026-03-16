import os
import math
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
            ...
          time_001/
            ...
    """
    rng = np.random.default_rng(seed)

    out_dir = os.path.join(out_root, "CTP")
    _ensure_dir(out_dir)

    study_uid = generate_uid()
    frame_of_ref_uid = generate_uid()

    patient_id = "DAIVERA_TEST"
    patient_name = "DAIVERA^SYNTHETIC"
    accession = "000000"
    study_desc = "Synthetic CTP Study"
    series_desc_base = "Synthetic CTP Timepoint"

    base_dt = datetime.now()

    rows, cols = shape
    pixel_spacing = [0.8, 0.8]

    for t in range(times):
        series_uid = generate_uid()
        series_number = 100 + t

        tp_dir = os.path.join(out_dir, f"time_{t:03d}")
        _ensure_dir(tp_dir)

        trigger_time = float(t * time_step_ms)
        series_dt = base_dt + timedelta(milliseconds=t * time_step_ms)
        study_date, series_time = _dt_to_dicoms(series_dt)

        for z in range(slices):
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = CTImageStorage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            file_meta.ImplementationClassUID = generate_uid()

            ds = FileDataset(
                filename_or_obj="",
                dataset={},
                file_meta=file_meta,
                preamble=b"\0" * 128,
            )

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
            ds.InstanceNumber = z + 1

            ds.ImagePositionPatient = [0.0, 0.0, float(z) * z_spacing_mm]
            ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            ds.PixelSpacing = pixel_spacing
            ds.SliceThickness = z_spacing_mm
            ds.SpacingBetweenSlices = z_spacing_mm

            ds.TemporalPositionIdentifier = int(t + 1)
            ds.TriggerTime = trigger_time
            ds.AcquisitionTime = series_time
            ds.ContentTime = series_time

            ds.Rows = rows
            ds.Columns = cols
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1

            yy, xx = np.mgrid[0:rows, 0:cols]
            cx = cols * 0.5 + math.sin(t / 3) * 10 + rng.normal(0, 0.5)
            cy = rows * 0.5 + math.cos(t / 4) * 10 + rng.normal(0, 0.5)
            sigma = 25.0
            bolus = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))

            amp = 800.0 * math.exp(-((t - times * 0.35) ** 2) / (2 * (times * 0.12) ** 2))
            base = -1000.0
            tissue = 40.0
            noise = rng.normal(0, 12, size=(rows, cols))

            img = base + tissue + amp * bolus + noise
            img += (z - slices / 2) * 0.8

            img_i16 = np.clip(img, -2048, 3071).astype(np.int16)

            ds.RescaleIntercept = 0
            ds.RescaleSlope = 1
            ds.PixelData = img_i16.tobytes()

            out_path = os.path.join(tp_dir, f"slice_{z:03d}.dcm")
            ds.save_as(out_path, write_like_original=False)

    return out_dir


def generate_fake_cta(
    out_root: str,
    slices: int = 180,
    shape: Tuple[int, int] = (512, 512),
    z_spacing_mm: float = 1.0,
    seed: int = 11,
) -> str:
    """
    Generates a fake CTA dataset:
      out_root/
        CTA/
          slice_000.dcm
          slice_001.dcm
          ...

    CTA is slice-only:
      - no timepoints
      - one series
    """
    rng = np.random.default_rng(seed)

    out_dir = os.path.join(out_root, "CTA")
    _ensure_dir(out_dir)

    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_of_ref_uid = generate_uid()

    patient_id = "DAIVERA_TEST"
    patient_name = "DAIVERA^SYNTHETIC"
    accession = "000001"
    study_desc = "Synthetic CTA Study"
    series_desc = "Synthetic CTA Arterial Phase"

    base_dt = datetime.now()
    study_date, series_time = _dt_to_dicoms(base_dt)

    rows, cols = shape
    pixel_spacing = [0.6, 0.6]

    yy, xx = np.mgrid[0:rows, 0:cols]

    main_cx = cols * 0.52
    main_cy = rows * 0.48

    for z in range(slices):
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        ds = FileDataset(
            filename_or_obj="",
            dataset={},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )

        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_of_ref_uid

        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.AccessionNumber = accession

        ds.StudyDescription = study_desc
        ds.SeriesDescription = series_desc

        ds.Modality = "CT"
        ds.StudyDate = study_date
        ds.StudyTime = series_time
        ds.SeriesDate = study_date
        ds.SeriesTime = series_time
        ds.ContentDate = study_date
        ds.ContentTime = series_time

        ds.SeriesNumber = 200
        ds.InstanceNumber = z + 1

        ds.ImagePositionPatient = [0.0, 0.0, float(z) * z_spacing_mm]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = pixel_spacing
        ds.SliceThickness = z_spacing_mm
        ds.SpacingBetweenSlices = z_spacing_mm

        ds.Rows = rows
        ds.Columns = cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1

        img = rng.normal(35, 10, size=(rows, cols))

        body_rx = cols * 0.34
        body_ry = rows * 0.40
        body = (((xx - cols / 2) / body_rx) ** 2 + ((yy - rows / 2) / body_ry) ** 2) <= 1.0
        img[~body] = -1000

        spine = ((xx - cols * 0.50) ** 2 + (yy - rows * 0.68) ** 2) <= (rows * 0.05) ** 2
        img[spine] = 700 + rng.normal(0, 20, size=spine.sum())

        cx = main_cx + 18 * math.sin(z / 18.0)
        cy = main_cy + 10 * math.cos(z / 23.0)

        vessel_r = 9 + 2 * math.sin(z / 20.0)
        vessel = ((xx - cx) ** 2 + (yy - cy) ** 2) <= vessel_r**2
        img[vessel] = 320 + rng.normal(0, 18, size=vessel.sum())

        branch_cx = cx + 30 + 6 * math.sin(z / 15.0)
        branch_cy = cy - 20 + 4 * math.cos(z / 17.0)
        branch_r = 5 + 1.2 * math.cos(z / 22.0)
        branch = ((xx - branch_cx) ** 2 + (yy - branch_cy) ** 2) <= branch_r**2
        img[branch] = 280 + rng.normal(0, 16, size=branch.sum())

        if slices * 0.40 <= z <= slices * 0.60:
            stenosis_r = max(3.0, vessel_r * 0.45)
            stenosis = ((xx - cx) ** 2 + (yy - cy) ** 2) <= stenosis_r**2
            img[vessel] = 70 + rng.normal(0, 12, size=vessel.sum())
            img[stenosis] = 320 + rng.normal(0, 15, size=stenosis.sum())

        img += (z - slices / 2) * 0.15

        img_i16 = np.clip(img, -1024, 3071).astype(np.int16)

        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        ds.PixelData = img_i16.tobytes()

        out_path = os.path.join(out_dir, f"slice_{z:03d}.dcm")
        ds.save_as(out_path, write_like_original=False)

    return out_dir


if __name__ == "__main__":
    output_root = os.path.abspath("./synthetic_dicom_output")

    ctp_out = generate_fake_ctp(
        out_root=output_root,
        times=20,
        slices=24,
        shape=(256, 256),
        z_spacing_mm=5.0,
        time_step_ms=1000,
        seed=7,
    )

    cta_out = generate_fake_cta(
        out_root=output_root,
        slices=180,
        shape=(512, 512),
        z_spacing_mm=1.0,
        seed=11,
    )

    print("✅ Synthetic CTP DICOMs generated here:")
    print(ctp_out)
    print()
    print("✅ Synthetic CTA DICOMs generated here:")
    print(cta_out)