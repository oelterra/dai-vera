import customtkinter as ctk
import tkinter as tk
from pathlib import Path
from PIL import Image

from dai_vera.gui.theme import THEME
from dai_vera.gui.components.navigation import SidebarNav, PAGES
from dai_vera.gui.state import AppState

from dai_vera.gui.pages.import_ct import ImportCTPage
from dai_vera.gui.pages.curves_roi import CurvesROIPage
from dai_vera.gui.pages.vessel_analysis import VesselAnalysisPage
from dai_vera.gui.pages.ffr_results import FFRResultsPage


PAGE_CLASSES = {
    "import_ct": ImportCTPage,
    "curves_roi": CurvesROIPage,
    "vessel_analysis": VesselAnalysisPage,
    "ffr_results": FFRResultsPage,
}


class DAIVeraApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.withdraw()
        self.title("DAI Vera")
        self.geometry("1600x900")
        self.minsize(1200, 675)
        self._app_icon = None
        self._splash_image = None
        self._splash_window = None
        self.nav = None
        self.content = None

        ctk.set_appearance_mode("dark")
        self.configure(fg_color=THEME["bg"])
        self.wm_aspect(16, 9, 24, 13)
        self._set_app_icon()

        # shared state across pages
        self.app_state = AppState()

        self.current_key = "import_ct"
        self.page_instance = None

        # fullscreen toggles
        self._is_fullscreen = False

        def toggle_fullscreen(_event=None):
            self._is_fullscreen = not self._is_fullscreen
            self.attributes("-fullscreen", self._is_fullscreen)

        def exit_fullscreen(_event=None):
            self._is_fullscreen = False
            self.attributes("-fullscreen", False)

        self.bind("<F11>", toggle_fullscreen)
        self.bind("<Escape>", exit_fullscreen)

        self._show_startup_splash()

    def show_page(self, key: str):
        if self.page_instance is not None:
            self.page_instance.destroy()

        self.current_key = key
        page_cls = PAGE_CLASSES[key]
        self.page_instance = page_cls(self.content, app_state=self.app_state)
        self.page_instance.pack(fill="both", expand=True)

        self.nav.refresh()

    def navigate(self, key: str):
        self.show_page(key)

    def go_next(self):
        keys = [k for _, k in PAGES]
        i = keys.index(self.current_key)
        nxt = keys[min(i + 1, len(keys) - 1)]
        self.show_page(nxt)

    def go_back(self):
        keys = [k for _, k in PAGES]
        i = keys.index(self.current_key)
        prev = keys[max(i - 1, 0)]
        self.show_page(prev)

    def _set_app_icon(self):
        logo_path = Path(__file__).resolve().parents[1] / "assets" / "logo.png"
        if not logo_path.exists():
            return
        try:
            self._app_icon = tk.PhotoImage(file=str(logo_path))
            self.iconphoto(True, self._app_icon)
        except Exception:
            pass

    def _show_startup_splash(self):
        splash_path = Path(__file__).resolve().parents[1] / "assets" / "load.png"
        if not splash_path.exists():
            self._build_main_shell()
            return

        try:
            pil_image = Image.open(splash_path)
            width, height = pil_image.size
            max_width = 520
            scale = min(1.0, max_width / max(1, width))
            splash_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            self._splash_image = ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=splash_size,
            )
            splash_width, splash_height = splash_size
        except Exception:
            splash_width, splash_height = 420, 220

        self._splash_window = ctk.CTkToplevel(self)
        self._splash_window.title("DAI Vera")
        self._splash_window.overrideredirect(True)
        self._splash_window.configure(fg_color=THEME["bg"])
        self._splash_window.attributes("-topmost", True)

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        pos_x = max(0, (screen_w - splash_width) // 2)
        pos_y = max(0, (screen_h - splash_height) // 2)
        self._splash_window.geometry(f"{splash_width}x{splash_height}+{pos_x}+{pos_y}")

        splash_frame = ctk.CTkFrame(self._splash_window, fg_color=THEME["bg"], corner_radius=18)
        splash_frame.pack(fill="both", expand=True)

        if self._splash_image is not None:
            ctk.CTkLabel(splash_frame, text="", image=self._splash_image).pack(fill="both", expand=True)
        else:
            ctk.CTkLabel(
                splash_frame,
                text="DAI Vera",
                font=("Helvetica", 30, "bold"),
                text_color=THEME["text"],
            ).pack(fill="both", expand=True)

        self.after(3000, self._build_main_shell)

    def _build_main_shell(self):
        if self._splash_window is not None:
            self._splash_window.destroy()
            self._splash_window = None

        self.grid_columnconfigure(0, weight=0, minsize=280)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.nav = SidebarNav(
            self,
            on_navigate=self.navigate,
            on_back=self.go_back,
            on_next=self.go_next,
            get_current_key=lambda: self.current_key,
        )
        self.nav.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)

        self.content = ctk.CTkFrame(self, fg_color=THEME["bg"])
        self.content.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=10)

        self.deiconify()
        self.show_page(self.current_key)
