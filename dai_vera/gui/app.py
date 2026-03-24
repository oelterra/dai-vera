import customtkinter as ctk

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

        self.title("DAI Vera")
        self.geometry("1600x900")
        self.minsize(1200, 675)

        ctk.set_appearance_mode("dark")
        self.configure(fg_color=THEME["bg"])
        self.wm_aspect(16, 9, 24, 13)

        # shared state across pages
        self.app_state = AppState()

        self.current_key = "import_ct"
        self.page_instance = None

        # responsive workstation shell
        self.grid_columnconfigure(0, weight=0, minsize=280)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.nav = SidebarNav(
            self,
            on_navigate=self.navigate,
            on_next=self.go_next,
            get_current_key=lambda: self.current_key,
        )
        self.nav.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)

        self.content = ctk.CTkFrame(self, fg_color=THEME["bg"])
        self.content.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=10)

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

        self.show_page(self.current_key)

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
