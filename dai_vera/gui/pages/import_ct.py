import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog

from dai_vera.gui.theme import THEME, FONTS


class ImportCTPage(ctk.CTkFrame):
    key = "import_ct"

    def __init__(self, master, app_state):
        super().__init__(master, fg_color=THEME["bg"])

        # ✅ shared app state
        self.state = app_state

        # Root grid: two halves always
        self.grid_columnconfigure(0, weight=1, uniform="half")
        self.grid_columnconfigure(1, weight=1, uniform="half")
        self.grid_rowconfigure(0, weight=1)

        # Left half (images)
        self.left_panel = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(1, weight=1)

        # Right half (parameters container)
        self.right_panel = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1)

        # ✅ Make the right side scrollable (so nothing gets cut off on small windows)
        self.right_scroll = ctk.CTkScrollableFrame(
            self.right_panel,
            fg_color="transparent",
            corner_radius=0,
        )
        self.right_scroll.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.right_scroll.grid_columnconfigure(0, weight=1)

        # --- state vars (init from shared state) ---
        self.ctp_folder_path = ctk.StringVar(value=getattr(self.state, "ctp_folder", ""))
        self.cta_folder_path = ctk.StringVar(value=getattr(self.state, "cta_folder", ""))

        self.ctp_slice_index = ctk.IntVar(value=int(getattr(self.state, "ctp_slice", 12)))
        self.ctp_time_index = ctk.IntVar(value=int(getattr(self.state, "ctp_time", 50)))

        self.cta_slice_index = ctk.IntVar(value=int(getattr(self.state, "cta_slice", 12)))
        self.cta_time_index = ctk.IntVar(value=int(getattr(self.state, "cta_time", 50)))

        # Build left panels
        self._build_image_panel_ctp(row=0)
        self._build_image_panel_cta(row=1)

        # Build right parameters (now inside scrollable frame)
        self._build_parameters_panel()

    def _pick_folder(self, title: str) -> str:
        root = self.winfo_toplevel()
        return filedialog.askdirectory(parent=root, title=title, mustexist=True)

    def _select_folder_for(self, kind: str):
        path = self._pick_folder(f"Select {kind} folder")
        if not path:
            return

        if kind == "CTP":
            self.ctp_folder_path.set(path)
            self.state.ctp_folder = path  # ✅ persist
        else:
            self.cta_folder_path.set(path)
            self.state.cta_folder = path  # ✅ persist

    # ---------------- Left: Image Panels ----------------
    def _build_image_panel_ctp(self, row: int):
        self.ctp_panel = self._build_image_panel(
            parent=self.left_panel,
            title="CTP Images",
            kind="CTP",
            folder_var=self.ctp_folder_path,
            slice_var=self.ctp_slice_index,
            time_var=self.ctp_time_index,
            row=row,
        )

    def _build_image_panel_cta(self, row: int):
        self.cta_panel = self._build_image_panel(
            parent=self.left_panel,
            title="CTA Images",
            kind="CTA",
            folder_var=self.cta_folder_path,
            slice_var=self.cta_slice_index,
            time_var=self.cta_time_index,
            row=row,
        )

    def _build_image_panel(self, parent, title, kind, folder_var, slice_var, time_var, row: int):
        panel = ctk.CTkFrame(parent, fg_color=THEME["panel_2"], corner_radius=16)
        panel.grid(row=row, column=0, sticky="nsew", padx=14, pady=(14, 8) if row == 0 else (8, 14))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_columnconfigure(1, weight=0)
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_rowconfigure(2, weight=0)

        # Title
        ctk.CTkLabel(panel, text=title, font=FONTS["h1"], text_color=THEME["text"]).grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 6)
        )

        # Content area: (canvas) + slice slider
        content = ctk.CTkFrame(panel, fg_color="transparent")
        content.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=14, pady=(0, 10))
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=0)
        content.grid_rowconfigure(0, weight=1)

        # Canvas container
        canvas_wrap = ctk.CTkFrame(content, fg_color=THEME["panel_3"], corner_radius=14)
        canvas_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        canvas_wrap.grid_rowconfigure(0, weight=1)
        canvas_wrap.grid_columnconfigure(0, weight=1)

        tk_canvas = tk.Canvas(canvas_wrap, bg=THEME["panel_3"], highlightthickness=0, bd=0)
        tk_canvas.grid(row=0, column=0, sticky="nsew")

        inner = tk.Frame(tk_canvas, bg=THEME["panel_3"])
        inner_id = tk_canvas.create_window((0, 0), window=inner, anchor="nw")

        # Dashed upload zone (single click)
        upload_canvas = tk.Canvas(
            inner,
            width=520,
            height=220,
            bg=THEME["panel_3"],
            highlightthickness=0,
            bd=0,
        )
        upload_canvas.pack(padx=14, pady=14)

        def redraw_dashed(_=None):
            upload_canvas.delete("all")
            w = max(10, upload_canvas.winfo_width())
            h = max(10, upload_canvas.winfo_height())
            pad = 14
            upload_canvas.create_rectangle(
                pad, pad, w - pad, h - pad,
                outline=THEME["muted"],
                width=2,
                dash=(6, 6)
            )
            upload_canvas.create_text(
                w // 2, h // 2,
                text=f"Upload {kind} Image",
                fill=THEME["text"],
                font=("Helvetica", 13)
            )

        upload_canvas.bind("<Configure>", redraw_dashed)
        redraw_dashed()
        upload_canvas.bind("<Button-1>", lambda e: self._select_folder_for(kind))

        # Keep scrollregion correct
        def on_inner_configure(_evt=None):
            tk_canvas.configure(scrollregion=tk_canvas.bbox("all"))
            tk_canvas.itemconfigure(inner_id, width=tk_canvas.winfo_width())

        inner.bind("<Configure>", on_inner_configure)

        # Slice slider (right)
        slice_col = ctk.CTkFrame(content, fg_color="transparent")
        slice_col.grid(row=0, column=1, sticky="ns")

        ctk.CTkLabel(slice_col, text="Slice", text_color=THEME["muted"], font=FONTS["small"]).pack(pady=(8, 6))

        slice_slider = ctk.CTkSlider(
            slice_col,
            from_=1, to=100,
            number_of_steps=99,
            orientation="vertical",
            variable=slice_var,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
            height=200,
        )
        slice_slider.pack(padx=6, pady=(0, 6), fill="y")

        slice_val = ctk.CTkLabel(slice_col, text=str(slice_var.get()), text_color=THEME["text"], font=FONTS["small"])
        slice_val.pack(pady=(0, 8))

        def on_slice_change(_v=None):
            v = int(slice_var.get())
            slice_val.configure(text=str(v))
            if kind == "CTP":
                self.state.ctp_slice = v
            else:
                self.state.cta_slice = v

        slice_slider.configure(command=on_slice_change)
        on_slice_change()

        # Time slider (bottom)
        bottom = ctk.CTkFrame(panel, fg_color="transparent")
        bottom.grid(row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 14))
        bottom.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(bottom, text="Time Points", text_color=THEME["muted"], font=FONTS["small"]).grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )

        time_slider = ctk.CTkSlider(
            bottom,
            from_=1, to=100,
            number_of_steps=99,
            variable=time_var,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
        )
        time_slider.grid(row=0, column=1, sticky="ew")

        time_val = ctk.CTkLabel(bottom, text=str(time_var.get()), text_color=THEME["text"], font=FONTS["small"])
        time_val.grid(row=0, column=2, sticky="e", padx=(10, 0))

        def on_time_change(_v=None):
            v = int(time_var.get())
            time_val.configure(text=str(v))
            if kind == "CTP":
                self.state.ctp_time = v
            else:
                self.state.cta_time = v

        time_slider.configure(command=on_time_change)
        on_time_change()

        return panel

    # ---------------- Right: Parameters ----------------
    def _build_parameters_panel(self):
        # ✅ Put everything inside the scrollable frame
        wrap = ctk.CTkFrame(self.right_scroll, fg_color=THEME["panel_2"], corner_radius=16)
        wrap.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        wrap.grid_columnconfigure(0, weight=1)

        # --- Image Options ---
        ctk.CTkLabel(wrap, text="Image Options", font=FONTS["title"], text_color=THEME["text"]).grid(
            row=0, column=0, sticky="w", padx=14, pady=(14, 10)
        )

        self.image_options_box = ctk.CTkFrame(wrap, fg_color=THEME["panel_3"], corner_radius=14)
        self.image_options_box.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))
        self.image_options_box.grid_columnconfigure(1, weight=1)

        # ✅ init from state
        self.ctp_contrast_length = ctk.DoubleVar(value=float(getattr(self.state, "ctp_length", 0.50)))
        self.ctp_contrast_width = ctk.DoubleVar(value=float(getattr(self.state, "ctp_width", 0.50)))
        self.cta_contrast_length = ctk.DoubleVar(value=float(getattr(self.state, "cta_length", 0.50)))
        self.cta_contrast_width = ctk.DoubleVar(value=float(getattr(self.state, "cta_width", 0.50)))

        r = 0
        ctk.CTkLabel(self.image_options_box, text="CTP Images", font=FONTS["h2"], text_color=THEME["text"]).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 6)
        )
        r += 1
        r, self.slider_ctp_len = self._slider_row(self.image_options_box, "Length", self.ctp_contrast_length, r)
        r, self.slider_ctp_wid = self._slider_row(self.image_options_box, "Width", self.ctp_contrast_width, r)

        ctk.CTkLabel(self.image_options_box, text="CTA Images", font=FONTS["h2"], text_color=THEME["text"]).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 6)
        )
        r += 1
        r, self.slider_cta_len = self._slider_row(self.image_options_box, "Length", self.cta_contrast_length, r)
        r, self.slider_cta_wid = self._slider_row(self.image_options_box, "Width", self.cta_contrast_width, r)

        # ✅ persist image option sliders into state
        self.slider_ctp_len.configure(command=lambda _v=None: setattr(self.state, "ctp_length", float(self.ctp_contrast_length.get())))
        self.slider_ctp_wid.configure(command=lambda _v=None: setattr(self.state, "ctp_width", float(self.ctp_contrast_width.get())))
        self.slider_cta_len.configure(command=lambda _v=None: setattr(self.state, "cta_length", float(self.cta_contrast_length.get())))
        self.slider_cta_wid.configure(command=lambda _v=None: setattr(self.state, "cta_width", float(self.cta_contrast_width.get())))

        # ✅ NEW: Set Translations section (under sliders)
        ctk.CTkLabel(self.image_options_box, text="Set Translations", font=FONTS["h2"], text_color=THEME["text"]).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=14, pady=(16, 8)
        )
        r += 1

        self.translations_row = ctk.CTkFrame(self.image_options_box, fg_color="transparent")
        self.translations_row.grid(row=r, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 14))
        self.translations_row.grid_columnconfigure(0, weight=1)
        self.translations_row.grid_columnconfigure(1, weight=1)
        self.translations_row.grid_columnconfigure(2, weight=1)

        self.btn_add_translation = ctk.CTkButton(
            self.translations_row,
            text="Add Translation",
            height=46,
            corner_radius=12,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            command=self._on_add_translation,
        )
        self.btn_add_translation.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.btn_select_ctp_slice = ctk.CTkButton(
            self.translations_row,
            text="Select CTP Slice",
            height=46,
            corner_radius=12,
            fg_color=THEME["panel_2"],
            hover_color=THEME["border_2"],
            text_color=THEME["text"],
            command=self._on_select_ctp_slice,
        )
        self.btn_select_ctp_slice.grid(row=0, column=1, sticky="ew", padx=8)

        self.btn_select_cta_slice = ctk.CTkButton(
            self.translations_row,
            text="Select CTA Slice",
            height=46,
            corner_radius=12,
            fg_color=THEME["panel_2"],
            hover_color=THEME["border_2"],
            text_color=THEME["text"],
            command=self._on_select_cta_slice,
        )
        self.btn_select_cta_slice.grid(row=0, column=2, sticky="ew", padx=(8, 0))

        # --- Input Parameters ---
        ctk.CTkLabel(wrap, text="Input Parameters", font=FONTS["title"], text_color=THEME["text"]).grid(
            row=2, column=0, sticky="w", padx=14, pady=(10, 10)
        )

        self.input_params_box = ctk.CTkFrame(wrap, fg_color=THEME["panel_3"], corner_radius=14)
        self.input_params_box.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 12))
        self.input_params_box.grid_columnconfigure(0, weight=1)
        self.input_params_box.grid_columnconfigure(1, weight=1)

        self.param_coronary_artery = ctk.StringVar(value="Left Main")
        self.param_coronary_dominance = ctk.StringVar(value="Right")
        self.param_rest_stress = ctk.StringVar(value="Rest")

        self.param_xray_kv = ctk.StringVar(value="")
        self.param_contrast_concentration = ctk.StringVar(value="")
        self.param_contrast_volume_ml = ctk.StringVar(value="")
        self.param_abp_mmhg = ctk.StringVar(value="")

        row = 0
        row = self._form_dropdown(self.input_params_box, "Coronary Artery", self.param_coronary_artery,
                                  ["Left Main", "Right Coronary Artery", "LAD", "LCx"], row)
        row = self._form_dropdown(self.input_params_box, "Coronary Dominance", self.param_coronary_dominance,
                                  ["Right", "Left"], row)
        row = self._form_dropdown(self.input_params_box, "Rest or Stress Condition", self.param_rest_stress,
                                  ["Rest", "Stress"], row)
        row = self._form_entry(self.input_params_box, "X-ray Tube Voltage (kV)", self.param_xray_kv, row)
        row = self._form_entry(self.input_params_box, "Contrast Concentration", self.param_contrast_concentration, row)
        row = self._form_entry(self.input_params_box, "Contrast Volume (mL)", self.param_contrast_volume_ml, row)
        row = self._form_entry(self.input_params_box, "Arterial Blood Pressure (mmHg)", self.param_abp_mmhg, row)

        # Buttons
        self.bottom_buttons = ctk.CTkFrame(wrap, fg_color="transparent")
        self.bottom_buttons.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 14))
        self.bottom_buttons.grid_columnconfigure(0, weight=1)
        self.bottom_buttons.grid_columnconfigure(1, weight=1)

        self.btn_clear = ctk.CTkButton(
            self.bottom_buttons, text="Clear", height=40, corner_radius=12,
            fg_color=THEME["accent"], hover_color=THEME["accent_2"], text_color="black",
            command=self._on_clear
        )
        self.btn_clear.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.btn_compute = ctk.CTkButton(
            self.bottom_buttons, text="Compute", height=40, corner_radius=12,
            fg_color=THEME["accent"], hover_color=THEME["accent_2"], text_color="black",
            command=self._on_compute
        )
        self.btn_compute.grid(row=0, column=1, sticky="ew", padx=(8, 0))

    def _slider_row(self, parent, label, var, row):
        parent.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(parent, text=label, text_color=THEME["muted"], font=FONTS["body"]).grid(
            row=row, column=0, sticky="w", padx=14, pady=(6, 6)
        )
        line = ctk.CTkFrame(parent, fg_color="transparent")
        line.grid(row=row, column=1, sticky="ew", padx=(0, 14), pady=(6, 6))
        line.grid_columnconfigure(0, weight=1)

        slider = ctk.CTkSlider(
            line, from_=0.0, to=1.0, variable=var,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
        )
        slider.grid(row=0, column=0, sticky="ew")
        return row + 1, slider

    def _form_dropdown(self, parent, label, var, options, row):
        ctk.CTkLabel(parent, text=label, text_color=THEME["text"], font=FONTS["body"]).grid(
            row=row, column=0, sticky="w", padx=14, pady=(10, 0)
        )
        dd = ctk.CTkOptionMenu(
            parent,
            values=options,
            variable=var,
            fg_color=THEME["input_bg"],
            button_color=THEME["border"],
            button_hover_color=THEME["border_2"],
            text_color=THEME["text"],
            dropdown_fg_color=THEME["panel_2"],
            dropdown_text_color=THEME["text"],
            dropdown_hover_color=THEME["border"],
            height=36,
        )
        dd.grid(row=row, column=1, sticky="ew", padx=14, pady=(10, 0))
        return row + 1

    def _form_entry(self, parent, label, var, row):
        ctk.CTkLabel(parent, text=label, text_color=THEME["text"], font=FONTS["body"]).grid(
            row=row, column=0, sticky="w", padx=14, pady=(10, 0)
        )
        e = ctk.CTkEntry(
            parent,
            textvariable=var,
            height=36,
            fg_color=THEME["input_bg"],
            border_color=THEME["input_border"],
            text_color=THEME["text"],
        )
        e.grid(row=row, column=1, sticky="ew", padx=14, pady=(10, 0))
        return row + 1

    # ---------------- Actions ----------------
    def _on_clear(self):
        self.ctp_folder_path.set("")
        self.cta_folder_path.set("")
        self.state.ctp_folder = ""
        self.state.cta_folder = ""

    def _on_compute(self):
        print("Compute clicked")

    def _on_add_translation(self):
        print("Add Translation clicked")

    def _on_select_ctp_slice(self):
        print("Select CTP Slice clicked")

    def _on_select_cta_slice(self):
        print("Select CTA Slice clicked")
