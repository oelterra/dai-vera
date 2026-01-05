import customtkinter as ctk

from dai_vera.gui.theme import THEME, FONTS


class VesselAnalysisPage(ctk.CTkFrame):
    key = "vessel_analysis"

    def __init__(self, master, app_state):
        super().__init__(master, fg_color=THEME["bg"])
        self.state = app_state

        # ---------- responsive 2-column layout ----------
        self.grid_columnconfigure(0, weight=1, uniform="half")
        self.grid_columnconfigure(1, weight=1, uniform="half")
        self.grid_rowconfigure(0, weight=1)

        # ---------- LEFT ----------
        left = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)

        # Input Parameters box (top)
        params = ctk.CTkFrame(left, fg_color=THEME["panel_2"], corner_radius=16)
        params.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 12))
        params.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(params, text="Input Parameters", font=FONTS["h1"]).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 10)
        )

        ctk.CTkLabel(params, text="Pre Lesion Lumen Radius (cm)", font=FONTS["body"]).grid(
            row=1, column=0, sticky="w", padx=14, pady=(0, 10)
        )
        self.entry_pre_radius = ctk.CTkEntry(
            params,
            height=34,
            fg_color=THEME["input_bg"],
            border_color=THEME["border"],
            text_color=THEME["text"],
        )
        self.entry_pre_radius.grid(row=1, column=1, sticky="ew", padx=(12, 14), pady=(0, 10))

        ctk.CTkLabel(params, text="Pos Lesion Lumen Radius (cm)", font=FONTS["body"]).grid(
            row=2, column=0, sticky="w", padx=14, pady=(0, 14)
        )
        self.entry_post_radius = ctk.CTkEntry(
            params,
            height=34,
            fg_color=THEME["input_bg"],
            border_color=THEME["border"],
            text_color=THEME["text"],
        )
        self.entry_post_radius.grid(row=2, column=1, sticky="ew", padx=(12, 14), pady=(0, 14))

        # Buttons (stack)
        btns = ctk.CTkFrame(left, fg_color="transparent")
        btns.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        btns.grid_columnconfigure(0, weight=1)

        self.selected_view = ctk.StringVar(value="stenosis")

        self.btn_mark_stenosis = ctk.CTkButton(
            btns,
            text="Mark Stenosis",
            height=78,
            corner_radius=16,
            font=FONTS["h2"],
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            command=lambda: self._select_view("stenosis"),
        )
        self.btn_mark_stenosis.grid(row=0, column=0, sticky="ew", pady=(0, 14))

        self.btn_mark_branch = ctk.CTkButton(
            btns,
            text="Mark Branch",
            height=78,
            corner_radius=16,
            font=FONTS["h2"],
            fg_color=THEME["panel_2"],
            hover_color=THEME["border_2"],
            text_color=THEME["text"],
            command=lambda: self._select_view("branch"),
        )
        self.btn_mark_branch.grid(row=1, column=0, sticky="ew", pady=(0, 14))

        self.btn_mark_breakers = ctk.CTkButton(
            btns,
            text="Mark Breakers",
            height=78,
            corner_radius=16,
            font=FONTS["h2"],
            fg_color=THEME["panel_2"],
            hover_color=THEME["border_2"],
            text_color=THEME["text"],
            command=lambda: self._select_view("breakers"),
        )
        self.btn_mark_breakers.grid(row=2, column=0, sticky="ew")

        # ---------- RIGHT ----------
        right = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        # Title updates based on selected button
        self.right_title = ctk.CTkLabel(right, text="Mark Stenosis", font=FONTS["h1"])
        self.right_title.grid(row=0, column=0, sticky="w", padx=18, pady=(18, 10))

        # Empty content area for now
        self.right_body = ctk.CTkFrame(right, fg_color=THEME["panel_2"], corner_radius=16)
        self.right_body.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        self._apply_button_styles()

    def _select_view(self, view_key: str):
        self.selected_view.set(view_key)
        if view_key == "stenosis":
            self.right_title.configure(text="Mark Stenosis")
        elif view_key == "branch":
            self.right_title.configure(text="Mark Branch")
        else:
            self.right_title.configure(text="Mark Breakers")
        self._apply_button_styles()

    def _apply_button_styles(self):
        active = self.selected_view.get()

        def set_active(btn: ctk.CTkButton, is_active: bool):
            if is_active:
                btn.configure(fg_color=THEME["accent"], hover_color=THEME["accent_2"], text_color="black")
            else:
                btn.configure(fg_color=THEME["panel_2"], hover_color=THEME["border_2"], text_color=THEME["text"])

        set_active(self.btn_mark_stenosis, active == "stenosis")
        set_active(self.btn_mark_branch, active == "branch")
        set_active(self.btn_mark_breakers, active == "breakers")
