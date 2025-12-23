import customtkinter as ctk

from dai_vera.gui.theme import THEME
from dai_vera.gui.components.navigation import TopNav, PAGES

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
        self.geometry("1200x700")
        self.minsize(1100, 650)

        ctk.set_appearance_mode("dark")
        self.configure(fg_color=THEME["bg"])

        self.current_key = "import_ct"
        self.page_instance = None

        self.nav = TopNav(
            self,
            on_navigate=self.navigate,
            on_next=self.go_next,
            get_current_key=lambda: self.current_key,
        )
        self.nav.pack(fill="x", padx=14, pady=(14, 8))

        self.content = ctk.CTkFrame(self, fg_color=THEME["bg"])
        self.content.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        self.show_page(self.current_key)

    def show_page(self, key: str):
        if self.page_instance is not None:
            self.page_instance.destroy()

        self.current_key = key
        page_cls = PAGE_CLASSES[key]
        self.page_instance = page_cls(self.content)
        self.page_instance.pack(fill="both", expand=True)

        self.nav.refresh()

    def navigate(self, key: str):
        self.show_page(key)

    def go_next(self):
        keys = [k for _, k in PAGES]
        i = keys.index(self.current_key)
        nxt = keys[min(i + 1, len(keys) - 1)]
        self.show_page(nxt)
