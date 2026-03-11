from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Dict, Any
from dai_vera.roi import ROI


@dataclass
class AppState:
    # ---------------- Upload Paths ----------------
    ctp_folder: str = ""
    cta_folder: str = ""

    # ---------------- Loaded Volumes ----------------
    ctp_volume: Optional[Dict[str, Any]] = None
    cta_volume: Optional[Dict[str, Any]] = None

    # ---------------- Vendor ----------------
    ctp_vendor: str = ""
    cta_vendor: str = ""

    # ---------------- Slice / Time Positions ----------------
    ctp_slice: int = 12
    ctp_time: int = 50

    cta_slice: int = 12
    cta_time: int = 1

    # ---------------- Image Windowing ----------------
    ctp_length: float = 0.50
    ctp_width: float = 0.50

    cta_length: float = 0.50
    cta_width: float = 0.50

    # Optional alias for UI wording
    ctp_level: float = 0.50
    cta_level: float = 0.50

    # ---------------- Imaging Data ----------------
    ctp_image_4d: Optional[np.ndarray] = None
    ctp_time_points: Optional[np.ndarray] = None

    # ---------------- ROI State ----------------
    pre_roi: Optional[ROI] = None

    # ---------------- Slice Thickness ----------------
    ctp_slice_thickness_mm: float = 1.0
    cta_slice_thickness_mm: float = 1.0

    # ---------------- Translation Mapping ----------------
    translations: Dict[int, int] = field(default_factory=dict)

    selected_ctp_slice: Optional[int] = None
    selected_cta_slice: Optional[int] = None

    def set_pre_lesion(self, x: int, y: int, z: int) -> ROI:

        if self.ctp_image_4d is None:
            raise RuntimeError("CTP image data not loaded")

        if self.ctp_time_points is None:
            num_t = self.ctp_image_4d.shape[1]
            self.ctp_time_points = np.arange(num_t)

        z = int(np.clip(z, 0, self.ctp_image_4d.shape[0] - 1))
        y = int(np.clip(y, 0, self.ctp_image_4d.shape[2] - 1))
        x = int(np.clip(x, 0, self.ctp_image_4d.shape[3] - 1))

        curve = self.ctp_image_4d[z, :, y, x]

        radius_px = 2
        area_px = np.pi * radius_px**2

        self.pre_roi = ROI(
            x=x,
            y=y,
            z=z,
            time_points=self.ctp_time_points,
            curve=curve,
            radius=radius_px,
            area=area_px,
        )

        return self.pre_roi