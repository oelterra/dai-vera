from dataclasses import dataclass
import numpy as np
from typing import Optional, Dict, Any
from dai_vera.roi import ROI


@dataclass
class AppState:
    # Uploads
    ctp_folder: str = ""
    cta_folder: str = ""

    ctp_volume: Optional[Dict[str, Any]] = None
    cta_volume: Optional[Dict[str, Any]] = None

    ctp_vendor: str = ""
    cta_vendor: str = ""

    # Shared slice/time (persist across pages)
    ctp_slice: int = 12
    ctp_time: int = 50
    cta_slice: int = 12
    cta_time: int = 50

    # Image options
    ctp_length: float = 0.50
    ctp_width: float = 0.50
    cta_length: float = 0.50
    cta_width: float = 0.50

    # imaging data
    ctp_image_4d: Optional[np.ndarray] = None
    ctp_time_points: Optional[np.ndarray] = None
    
    # ROI state
    pre_roi: Optional[ROI] = None

    def set_pre_lesion(self, x: int, y: int, z: int) -> ROI:
        """
        Sets a pre-lesion ROI at (x, y, z) and extracts the time–intensity curve.
        """

        if self.ctp_image_4d is None:
            raise RuntimeError("CTP image data not loaded")

        # Initialize time points if needed
        if self.ctp_time_points is None:
            num_t = self.ctp_image_4d.shape[1]
            self.ctp_time_points = np.arange(num_t)

        # Clamp indices defensively
        z = int(np.clip(z, 0, self.ctp_image_4d.shape[0] - 1))
        y = int(np.clip(y, 0, self.ctp_image_4d.shape[2] - 1))
        x = int(np.clip(x, 0, self.ctp_image_4d.shape[3] - 1))

        # Extract time–intensity curve
        curve = self.ctp_image_4d[:, z, y, x]

        # Placeholder segmentation (replace later)
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
