# dai_vera/gui/components/navigation.py
import customtkinter as ctk
from dai_vera.gui.theme import THEME, FONTS

PAGES = [
    ("Import CT and Prep", "import_ct"),
    ("Curves and ROI", "curves_roi"),
    ("Vessel Analysis", "vessel_analysis"),
    ("FFR Results", "ffr_results"),
]


class TopNav(ctk.CTkFrame):
    def __init__(self, master, on_navigate, on_next, get_current_key):
        super().__init__(master, fg_color=THEME["panel"], corner_radius=12)

        self.on_navigate = on_navigate
        self.on_next = on_next
        self.get_current_key = get_current_key
        self.tab_buttons = {}

        # left: logo + name
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.pack(side="left", padx=12, pady=10)

        ctk.CTkLabel(left, text="DAI", text_color=THEME["accent"], font=FONTS["h1"]).pack(side="left")
        ctk.CTkLabel(left, text="Vera", text_color=THEME["text"], font=FONTS["h1"]).pack(side="left", padx=(6, 0))

        # middle: tabs
        mid = ctk.CTkFrame(self, fg_color="transparent")
        mid.pack(side="left", padx=20, pady=10)

        for label, key in PAGES:
            btn = ctk.CTkButton(
                mid,
                text=label,
                height=30,
                fg_color=THEME["panel_2"],
                hover_color=THEME["border"],
                text_color=THEME["text"],
                corner_radius=10,
                command=lambda k=key: self.on_navigate(k),
            )
            btn.pack(side="left", padx=6)
            self.tab_buttons[key] = btn

        # right: next button
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.pack(side="right", padx=12, pady=10)

        self.next_btn = ctk.CTkButton(
            right,
            text="Next",
            height=32,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            corner_radius=10,
            command=self.on_next,
        )
        self.next_btn.pack()

        self.refresh()

    def refresh(self):
        current = self.get_current_key()
        for key, btn in self.tab_buttons.items():
            if key == current:
                btn.configure(fg_color=THEME["accent"], text_color="black")
            else:
                btn.configure(fg_color=THEME["panel_2"], text_color=THEME["text"])
