from dataclasses import dataclass


@dataclass
class AppState:
    # Uploads
    ctp_folder: str = ""
    cta_folder: str = ""

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
