import customtkinter as ctk
from dai_vera.gui.theme import THEME, FONTS

class CurvesROIPage(ctk.CTkFrame):
    key = "curves_roi"
    title = "Curves and ROI Screen"

    def __init__(self, master):
        super().__init__(master, fg_color=THEME["bg"])
        ctk.CTkLabel(self, text=self.title, font=FONTS["title"], text_color=THEME["text"]).pack(anchor="w", padx=16, pady=12)
