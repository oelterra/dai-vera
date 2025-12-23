import customtkinter as ctk
from dai_vera.gui.theme import THEME, FONTS

class FFRResultsPage(ctk.CTkFrame):
    key = "ffr_results"
    title = "FFR Results"

    def __init__(self, master):
        super().__init__(master, fg_color=THEME["bg"])
        ctk.CTkLabel(self, text=self.title, font=FONTS["title"], text_color=THEME["text"]).pack(anchor="w", padx=16, pady=12)
