import os
import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from typing import Optional

try:
    import pydicom
except ImportError:
    pydicom = None

from dai_vera.gui.theme import THEME, FONTS


class ImportCTPage(ctk.CTkFrame):
    key = "import_ct"

    def __init__(self, master, app_state):
        super().__init__(master, fg_color=THEME["bg"])

        self.state = app_state

        # Translation workflow state
        self.translation_mode = False
        self.translation_stage = None  # None / "ctp" / "cta"
        self.pending_ctp_translation_slice = None

        # Favor the image workspace now that navigation is handled in a sidebar.
        self.grid_columnconfigure(0, weight=8)
        self.grid_columnconfigure(1, weight=4)
        self.grid_rowconfigure(0, weight=1)

        # Left half (images)
        self.left_panel = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(8, 5), pady=8)
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(1, weight=1)

        # Right half (parameters container)
        self.right_panel = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 8), pady=8)
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1)

        # --- state vars (init from shared state) ---
        self.ctp_folder_path = ctk.StringVar(value=getattr(self.state, "ctp_folder", ""))
        self.cta_folder_path = ctk.StringVar(value=getattr(self.state, "cta_folder", ""))

        # 1-based indices in UI
        self.ctp_slice_index = ctk.IntVar(value=max(1, int(getattr(self.state, "ctp_slice", 1))))
        self.ctp_time_index = ctk.IntVar(value=max(1, int(getattr(self.state, "ctp_time", 1))))
        self.cta_slice_index = ctk.IntVar(value=max(1, int(getattr(self.state, "cta_slice", 1))))
        self.cta_time_index = ctk.IntVar(value=max(1, int(getattr(self.state, "cta_time", 1))))

        # vendor selection
        self.ctp_vendor = ctk.StringVar(value=str(getattr(self.state, "ctp_vendor", "")))
        self.cta_vendor = ctk.StringVar(value=str(getattr(self.state, "cta_vendor", "")))

        # view refs
        self._view = {
            "CTP": {
                "upload_canvas": None,
                "photo": None,
                "slice_slider": None,
                "time_slider": None,
                "time_row": None,
                "zoom": 1.0,
                "display_image_size": None,
                "title_label": None,
                "slider_col": None,
                "pan_x": 0.0,
                "pan_y": 0.0,
                "drag_origin": None,
                "drag_pan_start": None,
                "was_dragged": False,
            },
            "CTA": {
                "upload_canvas": None,
                "photo": None,
                "slice_slider": None,
                "time_slider": None,
                "time_row": None,
                "zoom": 1.0,
                "display_image_size": None,
                "title_label": None,
                "slider_col": None,
                "pan_x": 0.0,
                "pan_y": 0.0,
                "drag_origin": None,
                "drag_pan_start": None,
                "was_dragged": False,
            },
        }

        self._movie_after_id = None

        # Build left panels
        self._build_image_panel_ctp(row=0)
        self._build_image_panel_cta(row=1)

        # Build right parameters
        self._build_parameters_panel()

        # Restore if already loaded
        self.after(60, self._restore_if_loaded)

    # ---------------- Vendor Prompt ----------------
    def _prompt_vendor(self, kind: str) -> Optional[str]:
        modal = ctk.CTkToplevel(self)
        modal.title(f"{kind} DICOM Vendor")
        modal.configure(fg_color=THEME["panel"])
        modal.grab_set()
        modal.transient(self.winfo_toplevel())

        modal.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            modal,
            text=f"Select {kind} DICOM Vendor",
            font=FONTS["h2"],
            text_color=THEME["text"],
        )
        title.grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))

        info = ctk.CTkLabel(
            modal,
            text="Choose the scanner vendor before uploading.",
            font=FONTS["body"],
            text_color=THEME["muted"],
        )
        info.grid(row=1, column=0, sticky="w", padx=16, pady=(0, 10))

        var = ctk.StringVar(value="GE")

        radios = ctk.CTkFrame(modal, fg_color="transparent")
        radios.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 12))
        radios.grid_columnconfigure(0, weight=1)

        for i, opt in enumerate(["GE", "Siemens", "Canon"]):
            rb = ctk.CTkRadioButton(
                radios,
                text=opt,
                variable=var,
                value=opt,
                text_color=THEME["text"],
                fg_color=THEME["accent"],
                hover_color=THEME["accent_2"],
            )
            rb.grid(row=i, column=0, sticky="w", pady=6)

        btn_row = ctk.CTkFrame(modal, fg_color="transparent")
        btn_row.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 16))
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)

        result = {"value": None}

        def on_ok():
            result["value"] = var.get()
            modal.destroy()

        def on_cancel():
            result["value"] = None
            modal.destroy()

        btn_cancel = ctk.CTkButton(
            btn_row,
            text="Cancel",
            height=40,
            corner_radius=12,
            fg_color=THEME["panel_2"],
            hover_color=THEME["border_2"],
            text_color=THEME["text"],
            command=on_cancel,
        )
        btn_cancel.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        btn_ok = ctk.CTkButton(
            btn_row,
            text="Continue",
            height=40,
            corner_radius=12,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            command=on_ok,
        )
        btn_ok.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        modal.update_idletasks()
        w, h = 420, 260
        x = modal.winfo_screenwidth() // 2 - w // 2
        y = modal.winfo_screenheight() // 2 - h // 2
        modal.geometry(f"{w}x{h}+{x}+{y}")

        self.wait_window(modal)
        return result["value"]

    # ---------------- Folder Picker ----------------
    def _pick_folder(self, title: str) -> str:
        root = self.winfo_toplevel()
        return filedialog.askdirectory(parent=root, title=title, mustexist=True)

    def _select_folder_for(self, kind: str):
        vendor = self._prompt_vendor(kind)
        if not vendor:
            return

        if kind == "CTP":
            self.ctp_vendor.set(vendor)
            self.state.ctp_vendor = vendor
        else:
            self.cta_vendor.set(vendor)
            self.state.cta_vendor = vendor

        path = self._pick_folder(f"Select {kind} folder")
        if not path:
            return

        if kind == "CTP":
            self.ctp_folder_path.set(path)
            self.state.ctp_folder = path
        else:
            self.cta_folder_path.set(path)
            self.state.cta_folder = path

        self._load_folder(kind, path)

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
        panel.grid(row=row, column=0, sticky="nsew", padx=8, pady=(8, 4) if row == 0 else (4, 8))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=0)

        content = ctk.CTkFrame(panel, fg_color="transparent")
        content.grid(row=0, column=0, sticky="nsew", padx=12, pady=(10, 6))
        content.grid_columnconfigure(0, weight=0)
        content.grid_columnconfigure(1, weight=1)
        content.grid_columnconfigure(2, weight=0)
        content.grid_rowconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            content,
            text=title.replace(" ", "\n", 1),
            font=FONTS["h2"],
            text_color=THEME["text"],
            justify="left",
            anchor="nw",
        )
        title_label.grid(row=0, column=0, sticky="nw", padx=(0, 10), pady=(2, 0))

        upload_canvas = tk.Canvas(
            content,
            bg=THEME["panel_3"],
            highlightthickness=0,
            bd=0,
        )
        upload_canvas.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=0)

        self._view[kind]["upload_canvas"] = upload_canvas
        self._view[kind]["title_label"] = title_label
        self._bind_zoom_events(upload_canvas, kind)

        def on_canvas_configure(_evt=None):
            vol = self.state.ctp_volume if kind == "CTP" else self.state.cta_volume
            if vol:
                self._render_current(kind)
            else:
                self._draw_upload_placeholder(kind)

        upload_canvas.bind("<Configure>", on_canvas_configure)
        upload_canvas.bind("<ButtonPress-1>", lambda e, view_kind=kind: self._on_canvas_press(view_kind, e))
        upload_canvas.bind("<B1-Motion>", lambda e, view_kind=kind: self._on_canvas_drag(view_kind, e))
        upload_canvas.bind("<ButtonRelease-1>", lambda e, view_kind=kind: self._on_canvas_release(view_kind, e))
        upload_canvas.bind("<Double-Button-1>", lambda e, view_kind=kind: self._reset_zoom(view_kind))

        slice_col = ctk.CTkFrame(content, fg_color="transparent")
        slice_col.grid(row=0, column=2, sticky="ns")
        self._view[kind]["slider_col"] = slice_col

        ctk.CTkLabel(slice_col, text="Slice", text_color=THEME["muted"], font=FONTS["small"]).pack(
            pady=(8, 6)
        )

        slice_val = ctk.CTkLabel(
            slice_col,
            text=str(slice_var.get()),
            text_color=THEME["text"],
            font=FONTS["small"],
            width=56,
            fg_color=THEME["panel_3"],
            corner_radius=10,
            anchor="center",
        )
        slice_val.pack(pady=(0, 8))

        slice_slider = ctk.CTkSlider(
            slice_col,
            from_=1,
            to=100,
            number_of_steps=99,
            orientation="vertical",
            variable=slice_var,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
            height=320,
        )
        slice_slider.pack(padx=6, pady=(0, 6), fill="y", expand=True)

        self._view[kind]["slice_slider"] = slice_slider

        def on_slice_change(_v=None):
            v = int(slice_var.get())
            slice_val.configure(text=str(v))
            if kind == "CTP":
                self.state.ctp_slice = v
            else:
                self.state.cta_slice = v
            self._render_current(kind)

        slice_slider.configure(command=on_slice_change)
        on_slice_change()

        if kind == "CTP":
            bottom = ctk.CTkFrame(panel, fg_color="transparent")
            bottom.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 10))
            bottom.grid_columnconfigure(1, weight=1)
            bottom.grid_columnconfigure(2, minsize=72)
            self._view[kind]["time_row"] = bottom

            ctk.CTkLabel(bottom, text="Time Points", text_color=THEME["muted"], font=FONTS["small"]).grid(
                row=0, column=0, sticky="w", padx=(0, 10)
            )

            time_slider = ctk.CTkSlider(
                bottom,
                from_=1,
                to=100,
                number_of_steps=99,
                variable=time_var,
                fg_color=THEME["border"],
                progress_color=THEME["accent"],
                button_color=THEME["accent"],
                button_hover_color=THEME["accent_2"],
            )
            time_slider.grid(row=0, column=1, sticky="ew")

            self._view[kind]["time_slider"] = time_slider

            time_val = ctk.CTkLabel(
                bottom,
                text=str(time_var.get()),
                text_color=THEME["text"],
                font=FONTS["small"],
                width=60,
                fg_color=THEME["panel_3"],
                corner_radius=10,
                anchor="center",
            )
            time_val.grid(row=0, column=2, sticky="e", padx=(10, 0))

            def on_time_change(_v=None):
                v = int(time_var.get())
                time_val.configure(text=str(v))
                self.state.ctp_time = v
                self._render_current(kind)

            time_slider.configure(command=on_time_change)
            on_time_change()
        else:
            self._view[kind]["time_row"] = None
            self._view[kind]["time_slider"] = None

        content.bind("<Configure>", lambda _event, view_kind=kind: self._update_image_box_size(view_kind))
        self.after(40, lambda view_kind=kind: self._update_image_box_size(view_kind))
        self._draw_upload_placeholder(kind)
        return panel

    def _draw_upload_placeholder(self, kind: str):
        upload_canvas = self._view[kind]["upload_canvas"]
        if upload_canvas is None:
            return

        self._view[kind]["display_image_size"] = None
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
            w // 2,
            h // 2,
            text=f"Upload {kind} Image",
            fill=THEME["text"],
            font=("Helvetica", 13),
        )

    def _bind_zoom_events(self, canvas: tk.Canvas, kind: str):
        canvas.bind("<MouseWheel>", lambda e, view_kind=kind: self._on_zoom_event(view_kind, e))
        canvas.bind("<Button-4>", lambda e, view_kind=kind: self._on_zoom_event(view_kind, e))
        canvas.bind("<Button-5>", lambda e, view_kind=kind: self._on_zoom_event(view_kind, e))

    def _on_canvas_press(self, kind: str, event):
        view = self._view[kind]
        view["drag_origin"] = (event.x, event.y)
        view["drag_pan_start"] = (view["pan_x"], view["pan_y"])
        view["was_dragged"] = False

    def _on_canvas_drag(self, kind: str, event):
        view = self._view[kind]
        vol = self.state.ctp_volume if kind == "CTP" else self.state.cta_volume
        if not vol or view["zoom"] <= 1.0 or not view["drag_origin"] or not view["drag_pan_start"]:
            return

        start_x, start_y = view["drag_origin"]
        pan_start_x, pan_start_y = view["drag_pan_start"]
        dx = event.x - start_x
        dy = event.y - start_y
        if abs(dx) > 2 or abs(dy) > 2:
            view["was_dragged"] = True
        view["pan_x"] = pan_start_x + dx
        view["pan_y"] = pan_start_y + dy
        self._render_current(kind)

    def _on_canvas_release(self, kind: str, _event):
        vol = self.state.ctp_volume if kind == "CTP" else self.state.cta_volume
        if not vol:
            self._select_folder_for(kind)
        self._view[kind]["drag_origin"] = None
        self._view[kind]["drag_pan_start"] = None

    def _on_zoom_event(self, kind: str, event):
        vol = self.state.ctp_volume if kind == "CTP" else self.state.cta_volume
        if not vol:
            return

        delta = getattr(event, "delta", 0)
        num = getattr(event, "num", None)
        if delta > 0 or num == 4:
            factor = 1.1
        elif delta < 0 or num == 5:
            factor = 1 / 1.1
        else:
            return

        self._view[kind]["zoom"] = min(6.0, max(1.0, self._view[kind]["zoom"] * factor))
        self._render_current(kind)
        return "break"

    def _reset_zoom(self, kind: str):
        self._view[kind]["zoom"] = 1.0
        self._view[kind]["pan_x"] = 0.0
        self._view[kind]["pan_y"] = 0.0
        vol = self.state.ctp_volume if kind == "CTP" else self.state.cta_volume
        if vol:
            self._render_current(kind)
        return "break"

    def _update_image_box_size(self, kind: str):
        upload_canvas = self._view[kind]["upload_canvas"]
        title_label = self._view[kind]["title_label"]
        slider_col = self._view[kind]["slider_col"]
        if upload_canvas is None or slider_col is None or title_label is None:
            return

        upload_canvas.update_idletasks()
        slider_col.update_idletasks()
        title_label.update_idletasks()
        parent = upload_canvas.master
        if parent is None:
            return

        total_w = max(1, parent.winfo_width())
        total_h = max(1, parent.winfo_height())
        title_w = max(52, title_label.winfo_width())
        slider_w = max(72, slider_col.winfo_width())
        bottom_h = 54 if kind == "CTP" else 0
        target = min(max(220, total_w - title_w - slider_w - 18), max(220, total_h - bottom_h))
        upload_canvas.configure(width=target, height=target)

    # ---------------- Right: Parameters ----------------
    def _build_parameters_panel(self):
        wrap = ctk.CTkFrame(self.right_panel, fg_color=THEME["panel_2"], corner_radius=16)
        wrap.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        wrap.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(wrap, text="Image Options", font=FONTS["h1"], text_color=THEME["text"]).grid(
            row=0, column=0, sticky="w", padx=12, pady=(12, 8)
        )

        self.image_options_box = ctk.CTkFrame(wrap, fg_color=THEME["panel_3"], corner_radius=14)
        self.image_options_box.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        self.image_options_box.grid_columnconfigure(0, weight=1)
        self.image_options_box.grid_columnconfigure(1, weight=1)
        self.image_options_box.grid_columnconfigure(2, weight=1)
        self.image_options_box.grid_columnconfigure(3, weight=1)

        self.ctp_contrast_level = ctk.DoubleVar(value=float(getattr(self.state, "ctp_level", getattr(self.state, "ctp_length", 0.50))))
        self.ctp_contrast_width = ctk.DoubleVar(value=float(getattr(self.state, "ctp_width", 0.50)))
        self.cta_contrast_level = ctk.DoubleVar(value=float(getattr(self.state, "cta_level", getattr(self.state, "cta_length", 0.50))))
        self.cta_contrast_width = ctk.DoubleVar(value=float(getattr(self.state, "cta_width", 0.50)))

        ctk.CTkLabel(self.image_options_box, text="CTP Images", font=FONTS["h2"], text_color=THEME["text"]).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 4)
        )
        _, self.slider_ctp_level = self._slider_row(self.image_options_box, "Level", self.ctp_contrast_level, 1, col_offset=0)
        _, self.slider_ctp_wid = self._slider_row(self.image_options_box, "Width", self.ctp_contrast_width, 2, col_offset=0)

        ctk.CTkLabel(self.image_options_box, text="CTA Images", font=FONTS["h2"], text_color=THEME["text"]).grid(
            row=0, column=2, columnspan=2, sticky="w", padx=12, pady=(10, 4)
        )
        _, self.slider_cta_level = self._slider_row(self.image_options_box, "Level", self.cta_contrast_level, 1, col_offset=2)
        _, self.slider_cta_wid = self._slider_row(self.image_options_box, "Width", self.cta_contrast_width, 2, col_offset=2)

        self.slider_ctp_level.configure(
            command=lambda _v=None: self._on_level_width_change("CTP")
        )
        self.slider_ctp_wid.configure(
            command=lambda _v=None: self._on_level_width_change("CTP")
        )
        self.slider_cta_level.configure(
            command=lambda _v=None: self._on_level_width_change("CTA")
        )
        self.slider_cta_wid.configure(
            command=lambda _v=None: self._on_level_width_change("CTA")
        )

        # Slice thickness buttons
        ctk.CTkLabel(self.image_options_box, text="Slice Thickness", font=FONTS["h2"], text_color=THEME["text"]).grid(
            row=3, column=0, columnspan=4, sticky="w", padx=12, pady=(10, 6)
        )

        self.slice_thickness_row = ctk.CTkFrame(self.image_options_box, fg_color="transparent")
        self.slice_thickness_row.grid(row=4, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 6))
        self.slice_thickness_row.grid_columnconfigure(0, weight=1)
        self.slice_thickness_row.grid_columnconfigure(1, weight=1)

        self.btn_ctp_thickness = ctk.CTkButton(
            self.slice_thickness_row,
            text="Change CTP Slice Thickness",
            height=34,
            corner_radius=12,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            command=lambda: self._change_slice_thickness("CTP"),
        )
        self.btn_ctp_thickness.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.btn_cta_thickness = ctk.CTkButton(
            self.slice_thickness_row,
            text="Change CTA Slice Thickness",
            height=34,
            corner_radius=12,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            command=lambda: self._change_slice_thickness("CTA"),
        )
        self.btn_cta_thickness.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.lbl_ctp_thickness = ctk.CTkLabel(
            self.image_options_box,
            text="Current CTP Slice Thickness: 1.0 mm",
            font=FONTS["small"],
            text_color=THEME["text"],
        )
        self.lbl_ctp_thickness.grid(row=5, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 2))

        self.lbl_cta_thickness = ctk.CTkLabel(
            self.image_options_box,
            text="Current CTA Slice Thickness: 1.0 mm",
            font=FONTS["small"],
            text_color=THEME["text"],
        )
        self.lbl_cta_thickness.grid(row=5, column=2, columnspan=2, sticky="w", padx=12, pady=(0, 2))

        # Translation section
        ctk.CTkLabel(self.image_options_box, text="Set Translations", font=FONTS["h2"], text_color=THEME["text"]).grid(
            row=6, column=0, columnspan=4, sticky="w", padx=12, pady=(6, 6)
        )

        self.translations_row = ctk.CTkFrame(self.image_options_box, fg_color="transparent")
        self.translations_row.grid(row=7, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 10))
        self.translations_row.grid_columnconfigure(0, weight=1)
        self.translations_row.grid_columnconfigure(1, weight=1)
        self.translations_row.grid_columnconfigure(2, weight=1)

        self.btn_add_translation = ctk.CTkButton(
            self.translations_row,
            text="Add Translation",
            height=34,
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
            height=34,
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
            height=34,
            corner_radius=12,
            fg_color=THEME["panel_2"],
            hover_color=THEME["border_2"],
            text_color=THEME["text"],
            command=self._on_select_cta_slice,
        )
        self.btn_select_cta_slice.grid(row=0, column=2, sticky="ew", padx=(8, 0))

        ctk.CTkLabel(wrap, text="Input Parameters", font=FONTS["h1"], text_color=THEME["text"]).grid(
            row=2, column=0, sticky="w", padx=12, pady=(6, 8)
        )

        self.input_params_box = ctk.CTkFrame(wrap, fg_color=THEME["panel_3"], corner_radius=14)
        self.input_params_box.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.input_params_box.grid_columnconfigure(0, weight=1)
        self.input_params_box.grid_columnconfigure(1, weight=1)
        self.input_params_box.grid_columnconfigure(2, weight=1)
        self.input_params_box.grid_columnconfigure(3, weight=1)

        self.param_coronary_artery = ctk.StringVar(value="Left Main")
        self.param_coronary_dominance = ctk.StringVar(value="Right")
        self.param_rest_stress = ctk.StringVar(value="Rest")
        self.param_xray_kv = ctk.StringVar(value="")
        self.param_contrast_concentration = ctk.StringVar(value="")
        self.param_contrast_volume_ml = ctk.StringVar(value="")
        self.param_abp_mmhg = ctk.StringVar(value="")

        self._compact_field(self.input_params_box, 0, 0, "Coronary Artery", "dropdown", self.param_coronary_artery,
                            ["Left Main", "Right Coronary Artery", "LAD", "LCx"])
        self._compact_field(self.input_params_box, 0, 1, "Coronary Dominance", "dropdown", self.param_coronary_dominance,
                            ["Right", "Left"])
        self._compact_field(self.input_params_box, 1, 0, "Rest or Stress Condition", "dropdown", self.param_rest_stress,
                            ["Rest", "Stress"])
        self._compact_field(self.input_params_box, 1, 1, "X-ray Tube Voltage (kV)", "entry", self.param_xray_kv)
        self._compact_field(self.input_params_box, 2, 0, "Contrast Concentration", "entry", self.param_contrast_concentration)
        self._compact_field(self.input_params_box, 2, 1, "Contrast Volume (mL)", "entry", self.param_contrast_volume_ml)
        self._compact_field(self.input_params_box, 3, 0, "Arterial Blood Pressure (mmHg)", "entry", self.param_abp_mmhg)
        self._compact_movie_controls(self.input_params_box, 3, 1)
        self._update_slice_thickness_labels()
        self._refresh_translation_buttons()

    def _update_slice_thickness_labels(self):
        self.lbl_ctp_thickness.configure(
            text=f"Current CTP Slice Thickness: {float(getattr(self.state, 'ctp_slice_thickness_mm', 1.0)):.1f} mm"
        )
        self.lbl_cta_thickness.configure(
            text=f"Current CTA Slice Thickness: {float(getattr(self.state, 'cta_slice_thickness_mm', 1.0)):.1f} mm"
        )

    def _slider_row(self, parent, label, var, row, col_offset=0):
        parent.grid_columnconfigure(col_offset + 1, weight=1)
        ctk.CTkLabel(parent, text=label, text_color=THEME["muted"], font=FONTS["body"]).grid(
            row=row, column=col_offset, sticky="w", padx=12, pady=(2, 2)
        )
        line = ctk.CTkFrame(parent, fg_color="transparent")
        line.grid(row=row, column=col_offset + 1, sticky="ew", padx=(0, 12), pady=(2, 2))
        line.grid_columnconfigure(0, weight=1)
        slider = ctk.CTkSlider(
            line,
            from_=0.0,
            to=1.0,
            variable=var,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
        )
        slider.grid(row=0, column=0, sticky="ew")
        return row + 1, slider

    def _compact_field(self, parent, row, col, label, kind, var, options=None):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=col * 2, columnspan=2, sticky="ew", padx=12, pady=(6, 0))
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text=label, text_color=THEME["text"], font=FONTS["body"]).grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )

        if kind == "dropdown":
            widget = ctk.CTkOptionMenu(
                frame,
                values=options or [],
                variable=var,
                fg_color=THEME["input_bg"],
                button_color=THEME["border"],
                button_hover_color=THEME["border_2"],
                text_color=THEME["text"],
                dropdown_fg_color=THEME["panel_2"],
                dropdown_text_color=THEME["text"],
                dropdown_hover_color=THEME["border"],
                height=32,
            )
        else:
            widget = ctk.CTkEntry(
                frame,
                textvariable=var,
                height=32,
                fg_color=THEME["input_bg"],
                border_color=THEME["input_border"],
                text_color=THEME["text"],
            )
        widget.grid(row=1, column=0, sticky="ew")

    def _compact_movie_controls(self, parent, row, col):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=col * 2, columnspan=2, sticky="ew", padx=12, pady=(6, 0))
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=0)

        ctk.CTkLabel(frame, text="Movie Playback", text_color=THEME["text"], font=FONTS["body"]).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )

        self.btn_play_movie = ctk.CTkButton(
            frame,
            text="Play Movie",
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            height=32,
            corner_radius=12,
            command=self._toggle_movie,
        )
        self.btn_play_movie.grid(row=1, column=0, sticky="ew", padx=(0, 8))

        self.var_movie_speed = ctk.StringVar(value="0.5x")
        ctk.CTkOptionMenu(
            frame,
            values=["0.25x", "0.5x", "0.75x", "1x", "1.25x", "1.5x", "2x"],
            variable=self.var_movie_speed,
            fg_color=THEME["input_bg"],
            button_color=THEME["border"],
            button_hover_color=THEME["border_2"],
            dropdown_fg_color=THEME["panel_2"],
            dropdown_hover_color=THEME["border"],
            height=32,
            width=108,
        ).grid(row=1, column=1, sticky="e")

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

    # ---------------- Level / Width ----------------
    def _on_level_width_change(self, kind: str):
        if kind == "CTP":
            level = float(self.ctp_contrast_level.get())
            width = float(self.ctp_contrast_width.get())
            self.state.ctp_level = level
            self.state.ctp_length = level
            self.state.ctp_width = width
        else:
            level = float(self.cta_contrast_level.get())
            width = float(self.cta_contrast_width.get())
            self.state.cta_level = level
            self.state.cta_length = level
            self.state.cta_width = width

        self._render_current(kind)

    # ---------------- DICOM Loading + Rendering ----------------
    def _load_folder(self, kind: str, folder: str):
        if pydicom is None:
            messagebox.showerror(
                "Missing dependency",
                "pydicom is not installed.\nRun: pip install pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg gdcm"
            )
            return

        try:
            if kind == "CTP":
                vol = self._load_dicom_ctp(folder)
            else:
                vol = self._load_dicom_cta(folder)
        except Exception as e:
            messagebox.showerror("DICOM Load Error", f"Failed to load DICOMs:\n{e}")
            return

        if kind == "CTP":
            self.state.ctp_volume = vol
            self.ctp_time_index.set(1)
            self.ctp_slice_index.set(1)
            self.state.ctp_time = 1
            self.state.ctp_slice = 1
            if vol.get("slice_thickness") is not None:
                self.state.ctp_slice_thickness = float(vol["slice_thickness"])
        else:
            self.state.cta_volume = vol
            self.cta_time_index.set(1)
            self.cta_slice_index.set(1)
            self.state.cta_time = 1
            self.state.cta_slice = 1
            self.state.cta_coronary_mask = None
            self.state.cta_coronary_mask_path = ""
            self.state.cta_segmentation_status = "idle"
            self.state.cta_segmentation_error = ""
            self.state.cta_segmentation_in_progress = False
            if vol.get("slice_thickness") is not None:
                self.state.cta_slice_thickness = float(vol["slice_thickness"])

        self._view[kind]["zoom"] = 1.0
        self._view[kind]["pan_x"] = 0.0
        self._view[kind]["pan_y"] = 0.0
        self._configure_sliders_from_volume(kind, vol)
        self._update_slice_thickness_labels()
        self.after(10, lambda: self._render_current(kind))

    def _load_dicom_ctp(self, folder: str) -> dict:
        files = []
        for root, _, fnames in os.walk(folder):
            for f in fnames:
                files.append(os.path.join(root, f))

        items = []
        slice_thickness = None

        for fp in files:
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
                if not hasattr(ds, "PixelData"):
                    continue

                if slice_thickness is None:
                    slice_thickness = self._extract_slice_thickness(ds)

                if hasattr(ds, "TemporalPositionIdentifier"):
                    t = int(ds.TemporalPositionIdentifier)
                elif hasattr(ds, "TriggerTime"):
                    t = float(ds.TriggerTime) / 1000.0
                elif hasattr(ds, "AcquisitionTime"):
                    t = float(str(ds.AcquisitionTime).replace(":", "") or 0)
                elif hasattr(ds, "ContentTime"):
                    t = float(str(ds.ContentTime).replace(":", "") or 0)
                else:
                    t = 0

                z = self._extract_z_value(ds)

                arr = ds.pixel_array.astype(np.float32)

                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                arr = arr * slope + intercept

                if arr.ndim == 2:
                    items.append((t, z, arr))
                elif arr.ndim == 3:
                    frame_spacing = slice_thickness if slice_thickness is not None else 1.0
                    for i in range(arr.shape[0]):
                        items.append((t, z + i * frame_spacing, arr[i]))
            except Exception:
                continue

        if not items:
            raise ValueError("No readable DICOM images with PixelData found in this folder.")

        times = sorted(set([x[0] for x in items]))
        by_time = {t: [] for t in times}
        for t, z, arr in items:
            by_time[t].append((z, arr))

        first_t = times[0]
        by_time[first_t].sort(key=lambda x: x[0])
        zs = [z for z, _ in by_time[first_t]]

        sample = by_time[first_t][0][1]
        H, W = sample.shape[-2], sample.shape[-1]

        frames = []
        for t in times:
            by_time[t].sort(key=lambda x: x[0])
            stack = []
            for z, arr in by_time[t]:
                if arr.shape[-2:] != (H, W):
                    continue
                stack.append(arr)
            if stack:
                frames.append(np.stack(stack, axis=0))

        if not frames:
            raise ValueError("Loaded DICOMs, but could not form a consistent (T,Z,H,W) volume.")

        vol4d = np.stack(frames, axis=0)

        return {
            "pixels": vol4d,
            "times": times,
            "zs": zs,
            "shape": vol4d.shape,
            "slice_thickness": slice_thickness if slice_thickness is not None else 1.0,
        }

    def _load_dicom_cta(self, folder: str) -> dict:
        files = []
        for root, _, fnames in os.walk(folder):
            for f in fnames:
                files.append(os.path.join(root, f))

        items = []
        slice_thickness = None

        for fp in files:
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
                if not hasattr(ds, "PixelData"):
                    continue

                if slice_thickness is None:
                    slice_thickness = self._extract_slice_thickness(ds)

                z = self._extract_z_value(ds)

                arr = ds.pixel_array.astype(np.float32)

                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                arr = arr * slope + intercept

                if arr.ndim == 2:
                    items.append((z, arr))
                elif arr.ndim == 3:
                    frame_spacing = slice_thickness if slice_thickness is not None else 1.0
                    for i in range(arr.shape[0]):
                        items.append((z + i * frame_spacing, arr[i]))
            except Exception:
                continue

        if not items:
            raise ValueError("No readable CTA DICOM images with PixelData found in this folder.")

        items.sort(key=lambda x: x[0])

        sample = items[0][1]
        H, W = sample.shape[-2], sample.shape[-1]

        slices = []
        zs = []
        for z, arr in items:
            if arr.shape[-2:] != (H, W):
                continue
            slices.append(arr)
            zs.append(z)

        if not slices:
            raise ValueError("CTA DICOMs were found, but slice stacking failed.")

        vol3d = np.stack(slices, axis=0)      # (Z,H,W)
        vol4d = np.expand_dims(vol3d, axis=0) # (1,Z,H,W)

        return {
            "pixels": vol4d,
            "times": [0],
            "zs": zs,
            "shape": vol4d.shape,
            "slice_thickness": slice_thickness if slice_thickness is not None else 1.0,
        }

    def _extract_z_value(self, ds):
        z = None
        if hasattr(ds, "ImagePositionPatient"):
            try:
                z = float(ds.ImagePositionPatient[2])
            except Exception:
                z = None
        if z is None and hasattr(ds, "SliceLocation"):
            try:
                z = float(ds.SliceLocation)
            except Exception:
                z = None
        if z is None and hasattr(ds, "InstanceNumber"):
            try:
                z = float(ds.InstanceNumber)
            except Exception:
                z = None
        if z is None:
            z = 0.0
        return z

    def _extract_slice_thickness(self, ds):
        for attr in ["SpacingBetweenSlices", "SliceThickness"]:
            if hasattr(ds, attr):
                try:
                    val = float(getattr(ds, attr))
                    if val > 0:
                        return val
                except Exception:
                    pass
        return 1.0

    def _configure_sliders_from_volume(self, kind: str, vol: dict):
        T, Z, _, _ = vol["shape"]

        slice_slider = self._view[kind]["slice_slider"]
        time_slider = self._view[kind]["time_slider"]
        time_row = self._view[kind]["time_row"]

        if slice_slider is not None:
            slice_slider.configure(from_=1, to=max(1, Z), number_of_steps=max(1, Z - 1))
            if kind == "CTP":
                self.ctp_slice_index.set(min(max(1, int(self.ctp_slice_index.get())), Z))
                self.state.ctp_slice = int(self.ctp_slice_index.get())
            else:
                self.cta_slice_index.set(min(max(1, int(self.cta_slice_index.get())), Z))
                self.state.cta_slice = int(self.cta_slice_index.get())

        if kind == "CTP":
            if time_row is not None:
                time_row.grid()
            if time_slider is not None:
                time_slider.configure(from_=1, to=max(1, T), number_of_steps=max(1, T - 1))
                self.ctp_time_index.set(min(max(1, int(self.ctp_time_index.get())), T))
                self.state.ctp_time = int(self.ctp_time_index.get())
        else:
            # CTA has no time points -> remove bottom slider
            self.cta_time_index.set(1)
            self.state.cta_time = 1
            if time_slider is not None:
                time_slider.configure(from_=1, to=1, number_of_steps=1)
            if time_row is not None:
                time_row.grid_remove()

    def _render_current(self, kind: str):
        vol = self.state.ctp_volume if kind == "CTP" else self.state.cta_volume
        if not vol:
            return

        upload_canvas = self._view[kind]["upload_canvas"]
        if upload_canvas is None:
            return

        pixels = vol["pixels"]
        T, Z, _, _ = pixels.shape

        if kind == "CTP":
            t_idx = int(self.ctp_time_index.get()) - 1
            z_idx = int(self.ctp_slice_index.get()) - 1
            level = float(getattr(self.state, "ctp_level", getattr(self.state, "ctp_length", 0.5)))
            width = float(getattr(self.state, "ctp_width", 0.5))
        else:
            t_idx = 0
            z_idx = int(self.cta_slice_index.get()) - 1
            level = float(getattr(self.state, "cta_level", getattr(self.state, "cta_length", 0.5)))
            width = float(getattr(self.state, "cta_width", 0.5))

        t_idx = min(max(0, t_idx), T - 1)
        z_idx = min(max(0, z_idx), Z - 1)

        img = pixels[t_idx, z_idx]
        img8 = self._to_uint8_for_display(img, level=level, width=width)
        if kind == "CTA":
            img8 = self._apply_cta_coronary_overlay(img8, z_idx)

        cw = max(10, int(upload_canvas.winfo_width()))
        ch = max(10, int(upload_canvas.winfo_height()))

        pil = Image.fromarray(img8)

        iw, ih = pil.size
        scale = min(cw / max(1, iw), ch / max(1, ih))
        scale *= float(self._view[kind]["zoom"])
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))
        pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self._view[kind]["display_image_size"] = (new_w, new_h)

        max_pan_x = max(0.0, (new_w - cw) / 2.0)
        max_pan_y = max(0.0, (new_h - ch) / 2.0)
        self._view[kind]["pan_x"] = float(np.clip(self._view[kind]["pan_x"], -max_pan_x, max_pan_x))
        self._view[kind]["pan_y"] = float(np.clip(self._view[kind]["pan_y"], -max_pan_y, max_pan_y))

        photo = ImageTk.PhotoImage(pil)
        self._view[kind]["photo"] = photo

        upload_canvas.delete("all")
        x = cw // 2 + int(self._view[kind]["pan_x"])
        y = ch // 2 + int(self._view[kind]["pan_y"])
        upload_canvas.create_image(x, y, image=photo, anchor="center")

    def _to_uint8_for_display(self, img: np.ndarray, level: float, width: float) -> np.ndarray:
        lo = np.percentile(img, 1)
        hi = np.percentile(img, 99)
        if hi <= lo:
            hi = lo + 1.0

        w_scale = 0.25 + (width * 1.75)
        center = (lo + hi) / 2.0

        # same old behavior, just renamed from length -> level
        shift = (level - 0.5) * (hi - lo) * 0.5
        center = center + shift

        span = (hi - lo) * w_scale
        w_lo = center - span / 2.0
        w_hi = center + span / 2.0

        out = np.clip((img - w_lo) / (w_hi - w_lo), 0, 1)
        out = (out * 255.0).astype(np.uint8)
        return out

    def _apply_cta_coronary_overlay(self, img8: np.ndarray, z_idx: int) -> np.ndarray:
        mask_volume = getattr(self.state, "cta_coronary_mask", None)
        if mask_volume is None or not getattr(self.state, "show_coronary_overlay", True):
            return img8

        try:
            if mask_volume.ndim != 3:
                raise RuntimeError(f"Expected a 3D coronary mask, got shape {mask_volume.shape}.")
            if z_idx < 0 or z_idx >= mask_volume.shape[0]:
                raise RuntimeError("CTA slice index is outside the coronary mask range.")

            mask_slice = np.asarray(mask_volume[z_idx]) > 0
            if mask_slice.shape != img8.shape:
                raise RuntimeError(
                    f"CTA coronary mask slice shape {mask_slice.shape} does not match CTA image shape {img8.shape}."
                )

            alpha = 0.42
            base = np.stack([img8, img8, img8], axis=-1).astype(np.float32)
            red = np.zeros_like(base)
            red[..., 0] = 255.0
            mask3 = mask_slice[..., None].astype(np.float32)
            blended = (base * (1.0 - alpha * mask3)) + (red * (alpha * mask3))
            return np.clip(blended, 0, 255).astype(np.uint8)
        except Exception as exc:
            self.state.cta_segmentation_error = f"Coronary overlay unavailable: {exc}"
            return img8

    def _restore_if_loaded(self):
        if getattr(self.state, "ctp_volume", None):
            self._configure_sliders_from_volume("CTP", self.state.ctp_volume)
            self._render_current("CTP")
        else:
            self._draw_upload_placeholder("CTP")

        if getattr(self.state, "cta_volume", None):
            self._configure_sliders_from_volume("CTA", self.state.cta_volume)
            self._render_current("CTA")
        else:
            self._draw_upload_placeholder("CTA")

        self._update_slice_thickness_labels()
        self._refresh_translation_buttons()

    def _toggle_movie(self):
        if self._movie_after_id is not None:
            self.after_cancel(self._movie_after_id)
            self._movie_after_id = None
            self.btn_play_movie.configure(text="Play Movie")
        else:
            if not getattr(self.state, "ctp_volume", None):
                messagebox.showwarning("No CTP loaded", "Load a CTP volume first.")
                return
            self.btn_play_movie.configure(text="Stop")
            self._movie_loop()

    def _movie_loop(self):
        vol = getattr(self.state, "ctp_volume", None)
        if not vol:
            self._movie_after_id = None
            self.btn_play_movie.configure(text="Play Movie")
            return

        max_t = vol["pixels"].shape[0]
        if max_t <= 1:
            self._movie_after_id = None
            self.btn_play_movie.configure(text="Play Movie")
            return

        try:
            speed = float(self.var_movie_speed.get().replace("x", ""))
        except ValueError:
            speed = 0.5

        delay = int(max(40, 250 / max(speed, 0.1)))
        next_val = (int(self.ctp_time_index.get()) % max_t) + 1
        self.ctp_time_index.set(next_val)
        self.state.ctp_time = next_val
        self._render_current("CTP")
        self._movie_after_id = self.after(delay, self._movie_loop)

    # ---------------- Slice Thickness ----------------
    def _change_slice_thickness(self, kind: str):
        vol = self.state.ctp_volume if kind == "CTP" else self.state.cta_volume
        if not vol:
            messagebox.showwarning("No volume loaded", f"Load {kind} first.")
            return

        current = float(getattr(self.state, "ctp_slice_thickness" if kind == "CTP" else "cta_slice_thickness", 1.0))
        value = simpledialog.askstring(
            f"{kind} Slice Thickness",
            f"Enter desired {kind} slice thickness in mm.\n\nCurrent: {current} mm\nExample: 2 or 2mm",
            parent=self.winfo_toplevel(),
        )
        if not value:
            return

        try:
            cleaned = str(value).strip().lower().replace("mm", "").strip()
            desired = float(cleaned)
            if desired <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid thickness", "Please enter a valid positive number, like 2 or 2mm.")
            return

        try:
            new_vol = self._resample_volume_slice_thickness(vol, current, desired)
        except Exception as e:
            messagebox.showerror("Resample Error", f"Could not change slice thickness:\n{e}")
            return

        if kind == "CTP":
            self.state.ctp_volume = new_vol
            self.state.ctp_slice_thickness = desired
            self.ctp_slice_index.set(1)
            self.state.ctp_slice = 1
            self.ctp_time_index.set(1)
            self.state.ctp_time = 1
            self._configure_sliders_from_volume("CTP", new_vol)
            self._render_current("CTP")
        else:
            self.state.cta_volume = new_vol
            self.state.cta_slice_thickness = desired
            self.cta_slice_index.set(1)
            self.state.cta_slice = 1
            self.cta_time_index.set(1)
            self.state.cta_time = 1
            self.state.cta_coronary_mask = None
            self.state.cta_coronary_mask_path = ""
            self.state.cta_segmentation_status = "idle"
            self.state.cta_segmentation_error = ""
            self.state.cta_segmentation_in_progress = False
            self._configure_sliders_from_volume("CTA", new_vol)
            self._render_current("CTA")

        self._update_slice_thickness_labels()

    def _resample_volume_slice_thickness(self, vol: dict, current_thickness: float, desired_thickness: float) -> dict:
        pixels = vol["pixels"]
        if pixels.ndim != 4:
            return vol

        T, Z, H, W = pixels.shape
        if Z <= 1:
            return vol

        resampled_frames = []
        for t in range(T):
            stack = pixels[t]
            resampled_frames.append(
                self._resample_stack_z(stack, current_thickness, desired_thickness)
            )

        new_pixels = np.stack(resampled_frames, axis=0)
        new_zs = list(range(new_pixels.shape[1]))

        return {
            "pixels": new_pixels,
            "times": vol.get("times", [0]),
            "zs": new_zs,
            "shape": new_pixels.shape,
            "slice_thickness": desired_thickness,
        }

    def _resample_stack_z(self, stack: np.ndarray, current_thickness: float, desired_thickness: float) -> np.ndarray:
        z_count = stack.shape[0]
        if z_count <= 1:
            return stack.copy()

        new_count = max(1, int(round(z_count * current_thickness / desired_thickness)))

        # Thicker -> fewer slices -> average neighboring slices
        if desired_thickness >= current_thickness:
            factor = desired_thickness / current_thickness
            out = []
            for i in range(new_count):
                start = int(round(i * factor))
                end = int(round((i + 1) * factor))
                start = max(0, min(start, z_count - 1))
                end = max(start + 1, min(end, z_count))
                avg = stack[start:end].mean(axis=0)
                out.append(avg)
            return np.stack(out, axis=0).astype(stack.dtype)

        # Thinner -> more slices -> interpolate
        old_pos = np.arange(z_count, dtype=np.float32)
        new_pos = np.linspace(0, z_count - 1, new_count, dtype=np.float32)

        low = np.floor(new_pos).astype(int)
        high = np.ceil(new_pos).astype(int)
        high = np.clip(high, 0, z_count - 1)

        frac = (new_pos - low).reshape(-1, 1, 1)

        out = (1.0 - frac) * stack[low] + frac * stack[high]
        return out.astype(stack.dtype)

    # ---------------- Translation ----------------
    def _refresh_translation_buttons(self):
        inactive_fg = THEME["panel_2"]
        inactive_hover = THEME["border_2"]
        active_fg = "#35c759"
        active_hover = "#2fb14d"

        if not self.translation_mode:
            self.btn_add_translation.configure(
                fg_color=THEME["accent"],
                hover_color=THEME["accent_2"],
                text="Add Translation",
                text_color="black",
            )
            self.btn_select_ctp_slice.configure(
                fg_color=inactive_fg,
                hover_color=inactive_hover,
                text_color=THEME["text"],
            )
            self.btn_select_cta_slice.configure(
                fg_color=inactive_fg,
                hover_color=inactive_hover,
                text_color=THEME["text"],
            )
            return

        self.btn_add_translation.configure(
            fg_color=THEME["border"],
            hover_color=THEME["border_2"],
            text="Translation In Progress",
            text_color=THEME["text"],
        )

        if self.translation_stage == "ctp":
            self.btn_select_ctp_slice.configure(
                fg_color=active_fg,
                hover_color=active_hover,
                text_color="black",
            )
            self.btn_select_cta_slice.configure(
                fg_color=inactive_fg,
                hover_color=inactive_hover,
                text_color=THEME["text"],
            )
        elif self.translation_stage == "cta":
            self.btn_select_ctp_slice.configure(
                fg_color=inactive_fg,
                hover_color=inactive_hover,
                text_color=THEME["text"],
            )
            self.btn_select_cta_slice.configure(
                fg_color=active_fg,
                hover_color=active_hover,
                text_color="black",
            )

    def _on_add_translation(self):
        if not self.state.ctp_volume or not self.state.cta_volume:
            messagebox.showwarning("Missing images", "Load both CTP and CTA before adding a translation.")
            return

        self.translation_mode = True
        self.translation_stage = "ctp"
        self.pending_ctp_translation_slice = None
        self._refresh_translation_buttons()

    def _on_select_ctp_slice(self):
        if not self.translation_mode or self.translation_stage != "ctp":
            return

        self.pending_ctp_translation_slice = int(self.ctp_slice_index.get())
        self.translation_stage = "cta"
        self._refresh_translation_buttons()

    def _on_select_cta_slice(self):
        if not self.translation_mode or self.translation_stage != "cta":
            return

        if self.pending_ctp_translation_slice is None:
            return

        ctp_slice = int(self.pending_ctp_translation_slice)
        cta_slice = int(self.cta_slice_index.get())

        self.state.translations[ctp_slice] = cta_slice

        self.translation_mode = False
        self.translation_stage = None
        self.pending_ctp_translation_slice = None
        self._refresh_translation_buttons()

        messagebox.showinfo(
            "Translation Saved",
            f"Saved translation:\nCTP slice {ctp_slice} -> CTA slice {cta_slice}"
        )

    # ---------------- Actions ----------------
    def _on_clear(self):
        self.ctp_folder_path.set("")
        self.cta_folder_path.set("")
        self.state.ctp_folder = ""
        self.state.cta_folder = ""

        self.state.ctp_volume = None
        self.state.cta_volume = None

        self.ctp_slice_index.set(1)
        self.ctp_time_index.set(1)
        self.cta_slice_index.set(1)
        self.cta_time_index.set(1)

        self.state.ctp_slice = 1
        self.state.ctp_time = 1
        self.state.cta_slice = 1
        self.state.cta_time = 1
        self.state.ctp_slice_thickness = 1.0
        self.state.cta_slice_thickness = 1.0

        self.translation_mode = False
        self.translation_stage = None
        self.pending_ctp_translation_slice = None
        self._view["CTP"]["zoom"] = 1.0
        self._view["CTA"]["zoom"] = 1.0
        self._refresh_translation_buttons()
        self._update_slice_thickness_labels()

        self._draw_upload_placeholder("CTP")
        self._draw_upload_placeholder("CTA")

    def _on_compute(self):
        print("Compute clicked")
