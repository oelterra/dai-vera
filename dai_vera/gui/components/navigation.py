import customtkinter as ctk

from dai_vera.gui.theme import THEME, FONTS

PAGES = [
    ("Import CT and Prep", "import_ct"),
    ("Curves and ROI", "curves_roi"),
    ("Vessel Analysis", "vessel_analysis"),
    ("FFR Results", "ffr_results"),
]


class SidebarNav(ctk.CTkFrame):
    def __init__(self, master, on_navigate, on_back, on_next, get_current_key):
        super().__init__(
            master,
            fg_color=THEME["panel"],
            corner_radius=0,
            border_width=0,
            width=280,
        )

        self.on_navigate = on_navigate
        self.on_back = on_back
        self.on_next = on_next
        self.get_current_key = get_current_key

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        brand = ctk.CTkFrame(self, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=22, pady=(22, 14))
        brand.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            brand,
            text="DAI Vera",
            font=("Helvetica", 28, "bold"),
            text_color=THEME["text"],
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            brand,
            text="CT workstation workflow",
            font=FONTS["body"],
            text_color=THEME["muted"],
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        nav_body = ctk.CTkFrame(self, fg_color="transparent")
        nav_body.grid(row=1, column=0, sticky="nsew", padx=16, pady=8)
        nav_body.grid_columnconfigure(0, weight=1)

        self.step_buttons = {}
        for idx, (label, key) in enumerate(PAGES, start=1):
            btn = ctk.CTkButton(
                nav_body,
                text=label,
                anchor="w",
                height=60,
                corner_radius=14,
                fg_color=THEME["panel_2"],
                hover_color=THEME["panel_3"],
                border_width=1,
                border_color=THEME["border"],
                text_color=THEME["text"],
                font=FONTS["h2"],
                command=lambda page_key=key: self.on_navigate(page_key),
            )
            btn.grid(row=idx - 1, column=0, sticky="ew", pady=(0, 10))
            self.step_buttons[key] = btn

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=16, pady=(12, 18))
        footer.grid_columnconfigure(0, weight=1)

        self.back_btn = ctk.CTkButton(
            footer,
            text="Back",
            height=40,
            corner_radius=12,
            fg_color=THEME["panel_2"],
            hover_color=THEME["panel_3"],
            text_color=THEME["text"],
            font=FONTS["body"],
            command=self.on_back,
        )
        self.back_btn.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.next_btn = ctk.CTkButton(
            footer,
            text="Next Step",
            height=44,
            corner_radius=12,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            font=FONTS["h2"],
            command=self.on_next,
        )
        self.next_btn.grid(row=1, column=0, sticky="ew")

        self.refresh()

    def refresh(self):
        current = self.get_current_key()
        keys = [key for _, key in PAGES]
        is_first = current == keys[0]
        is_last = current == keys[-1]
        self.back_btn.configure(state="disabled" if is_first else "normal")
        self.next_btn.configure(state="disabled" if is_last else "normal")

        for key, btn in self.step_buttons.items():
            if key == current:
                btn.configure(
                    fg_color=THEME["accent"],
                    hover_color=THEME["accent_2"],
                    border_color=THEME["accent"],
                    text_color="black",
                )
            else:
                btn.configure(
                    fg_color=THEME["panel_2"],
                    hover_color=THEME["panel_3"],
                    border_color=THEME["border"],
                    text_color=THEME["text"],
                )
