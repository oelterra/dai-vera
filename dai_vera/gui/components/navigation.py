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
        super().__init__(
            master,
            fg_color=THEME["panel"],
            corner_radius=18,
            border_width=0,
        )

        self.on_navigate = on_navigate
        self.on_next = on_next
        self.get_current_key = get_current_key

        # Layout: [title] [steps (expands)] [next]
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)

        # ---------------- LEFT: TITLE ONLY ----------------
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=0, column=0, sticky="w", padx=14, pady=12)

        ctk.CTkLabel(left, text="DAI", text_color=THEME["accent"], font=FONTS["h2"]).pack(side="left")
        ctk.CTkLabel(left, text="Vera", text_color=THEME["text"], font=FONTS["h2"]).pack(side="left", padx=(6, 0))

        # ---------------- CENTER: STEP BUTTONS (RESIZABLE) ----------------
        center = ctk.CTkFrame(self, fg_color="transparent")
        center.grid(row=0, column=1, sticky="ew", padx=10, pady=10)
        center.grid_columnconfigure(0, weight=1)

        # darker + thinner border, and it stretches with the window
        self.steps_wrap = ctk.CTkFrame(
            center,
            fg_color=THEME["panel_2"],
            corner_radius=12,
            border_width=1,                 # thinner
            border_color=THEME["black"],   # darker outline
        )
        self.steps_wrap.grid(row=0, column=0, sticky="ew")
        self.steps_wrap.grid_columnconfigure(0, weight=1)

        # inner frame stretches too
        self.steps_inner = ctk.CTkFrame(self.steps_wrap, fg_color="transparent")
        self.steps_inner.grid(row=0, column=0, sticky="ew", padx=14, pady=10)

        for i in range(len(PAGES)):
            self.steps_inner.grid_columnconfigure(i, weight=1)

        self.step_buttons = {}
        for i, (label, key) in enumerate(PAGES):
            b = ctk.CTkButton(
                self.steps_inner,
                text=label,
                height=32,  # slightly thinner
                corner_radius=12,
                fg_color=THEME["panel_3"],
                hover_color=THEME["border"],
                text_color=THEME["text"],
                font=FONTS["body"],
                command=lambda k=key: self.on_navigate(k),
            )
            # more spacing between tabs + expands evenly
            b.grid(row=0, column=i, padx=10, sticky="ew")
            self.step_buttons[key] = b

        # ---------------- RIGHT: NEXT ----------------
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=0, column=2, sticky="e", padx=14, pady=12)

        self.next_btn = ctk.CTkButton(
            right,
            text="Next",
            height=36,
            width=140,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            corner_radius=12,
            command=self.on_next,
        )
        self.next_btn.pack()

        self.refresh()

    def refresh(self):
        current = self.get_current_key()
        for key, btn in self.step_buttons.items():
            if key == current:
                btn.configure(fg_color=THEME["accent"], text_color="black")
            else:
                btn.configure(fg_color=THEME["panel_3"], text_color=THEME["text"])
