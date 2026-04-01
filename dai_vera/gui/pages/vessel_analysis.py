import threading
import tkinter as tk

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from dai_vera.gui.theme import THEME, FONTS
from dai_vera.segmentation.mask_loader import align_mask_to_cta_volume, load_nifti_mask
from dai_vera.segmentation.totalseg_runner import run_coronary_segmentation


class VesselAnalysisPage(ctk.CTkFrame):
    key = "vessel_analysis"

    def __init__(self, master, app_state):
        super().__init__(master, fg_color=THEME["bg"])
        self.state = app_state

        self.var_overlay = ctk.BooleanVar(value=bool(getattr(self.state, "show_coronary_overlay", True)))
        self.var_segmentation_status = ctk.StringVar(value="")
        self.var_workspace_hint = ctk.StringVar(value="")
        self.var_cta_slice = ctk.IntVar(value=max(1, int(getattr(self.state, "cta_slice", 1))))

        self._cta_photo = None
        self._cta_zoom = 1.0
        self._cta_pan_x = 0.0
        self._cta_pan_y = 0.0
        self._cta_drag_origin = None
        self._cta_drag_pan_start = None
        self._cta_was_dragged = False
        self._last_annotation_point = None

        self.grid_columnconfigure(0, weight=4)
        self.grid_columnconfigure(1, weight=8)
        self.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()

        self._apply_button_styles()
        self._refresh_segmentation_ui()
        self.after(50, self._refresh_workspace)

    def _build_left_panel(self):
        self.left_outer = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        self.left_outer.grid(row=0, column=0, sticky="nsew", padx=(8, 5), pady=8)
        self.left_outer.grid_columnconfigure(0, weight=1)
        self.left_outer.grid_rowconfigure(2, weight=1)

        params = ctk.CTkFrame(self.left_outer, fg_color=THEME["panel_2"], corner_radius=16)
        params.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 10))
        params.grid_columnconfigure(1, weight=1)
        self.params = params

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

        ctk.CTkLabel(params, text="Post Lesion Lumen Radius (cm)", font=FONTS["body"]).grid(
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

        ctk.CTkLabel(params, text="Coronary Segmentation", font=FONTS["h2"]).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 8)
        )

        self.btn_segment = ctk.CTkButton(
            params,
            text="Segment Coronary Arteries",
            height=36,
            corner_radius=12,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            command=self._start_segmentation,
        )
        self.btn_segment.grid(row=4, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 8))

        self.switch_overlay = ctk.CTkSwitch(
            params,
            text="Show Coronary Overlay",
            variable=self.var_overlay,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
            text_color=THEME["text"],
            command=self._toggle_overlay,
        )
        self.switch_overlay.grid(row=5, column=0, sticky="w", padx=14, pady=(0, 8))

        self.btn_clear_segmentation = ctk.CTkButton(
            params,
            text="Clear Coronary Segmentation",
            height=34,
            corner_radius=12,
            fg_color=THEME["panel_3"],
            hover_color=THEME["border_2"],
            text_color=THEME["text"],
            command=self._clear_segmentation,
        )
        self.btn_clear_segmentation.grid(row=5, column=1, sticky="ew", padx=(12, 14), pady=(0, 8))

        self.lbl_segmentation_status = ctk.CTkLabel(
            params,
            textvariable=self.var_segmentation_status,
            font=FONTS["small"],
            text_color=THEME["muted"],
            justify="left",
            wraplength=360,
        )
        self.lbl_segmentation_status.grid(row=6, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 14))

        actions = ctk.CTkFrame(self.left_outer, fg_color="transparent")
        actions.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
        actions.grid_columnconfigure(0, weight=1)

        self.selected_view = ctk.StringVar(value="stenosis")

        self.btn_mark_stenosis = ctk.CTkButton(
            actions,
            text="Mark Stenosis",
            height=64,
            corner_radius=16,
            font=FONTS["h2"],
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            command=lambda: self._select_view("stenosis"),
        )
        self.btn_mark_stenosis.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.btn_mark_branch = ctk.CTkButton(
            actions,
            text="Mark Branch",
            height=64,
            corner_radius=16,
            font=FONTS["h2"],
            fg_color=THEME["panel_2"],
            hover_color=THEME["border_2"],
            text_color=THEME["text"],
            command=lambda: self._select_view("branch"),
        )
        self.btn_mark_branch.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        self.btn_mark_breakers = ctk.CTkButton(
            actions,
            text="Mark Breakers",
            height=64,
            corner_radius=16,
            font=FONTS["h2"],
            fg_color=THEME["panel_2"],
            hover_color=THEME["border_2"],
            text_color=THEME["text"],
            command=lambda: self._select_view("breakers"),
        )
        self.btn_mark_breakers.grid(row=2, column=0, sticky="ew")

    def _build_right_panel(self):
        self.right_outer = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        self.right_outer.grid(row=0, column=1, sticky="nsew", padx=(5, 8), pady=8)
        self.right_outer.grid_columnconfigure(0, weight=1)
        self.right_outer.grid_rowconfigure(1, weight=1)

        self.right_title = ctk.CTkLabel(self.right_outer, text="Mark Stenosis", font=FONTS["h1"])
        self.right_title.grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))

        self.workspace = ctk.CTkFrame(self.right_outer, fg_color=THEME["panel_2"], corner_radius=16)
        self.workspace.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.workspace.grid_columnconfigure(0, weight=0)
        self.workspace.grid_columnconfigure(1, weight=1)
        self.workspace.grid_columnconfigure(2, weight=0)
        self.workspace.grid_rowconfigure(1, weight=1)
        self.workspace.grid_rowconfigure(2, weight=0)

        self.workspace_label = ctk.CTkLabel(
            self.workspace,
            text="CTA\nWorkspace",
            font=FONTS["h2"],
            text_color=THEME["text"],
            justify="left",
            anchor="nw",
        )
        self.workspace_label.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(16, 12), pady=(16, 0))

        self.workspace_header = ctk.CTkLabel(
            self.workspace,
            text="Coronary Vessel Analysis",
            font=FONTS["h1"],
            text_color=THEME["text"],
        )
        self.workspace_header.grid(row=0, column=1, columnspan=2, sticky="w", padx=(0, 16), pady=(16, 8))

        self.viewer_canvas = tk.Canvas(self.workspace, bg=THEME["panel_3"], highlightthickness=0, bd=0)
        self.viewer_canvas.grid(row=1, column=1, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self.viewer_canvas.bind("<Configure>", lambda _e: self._render_cta_viewer())
        self.viewer_canvas.bind("<MouseWheel>", self._on_cta_zoom_event)
        self.viewer_canvas.bind("<Button-4>", self._on_cta_zoom_event)
        self.viewer_canvas.bind("<Button-5>", self._on_cta_zoom_event)
        self.viewer_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.viewer_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.viewer_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.viewer_canvas.bind("<Double-Button-1>", self._reset_cta_zoom)

        self.viewer_placeholder = ctk.CTkLabel(
            self.workspace,
            text="CTA viewer will appear here.\nLoad CTA data, then run coronary segmentation.",
            font=FONTS["body"],
            text_color=THEME["muted"],
            justify="center",
        )
        self.viewer_placeholder.place(in_=self.viewer_canvas, relx=0.5, rely=0.5, anchor="center")

        slice_col = ctk.CTkFrame(self.workspace, fg_color="transparent")
        slice_col.grid(row=1, column=2, sticky="ns", padx=(0, 16), pady=(0, 10))

        ctk.CTkLabel(slice_col, text="Slice", text_color=THEME["muted"], font=FONTS["small"]).pack(pady=(4, 6))
        self.lbl_cta_slice_val = ctk.CTkLabel(
            slice_col,
            text=str(self.var_cta_slice.get()),
            font=FONTS["small"],
            width=58,
            fg_color=THEME["panel_3"],
            corner_radius=10,
        )
        self.lbl_cta_slice_val.pack(pady=(0, 8))

        self.slider_cta_slice = ctk.CTkSlider(
            slice_col,
            from_=1,
            to=2,
            number_of_steps=1,
            orientation="vertical",
            variable=self.var_cta_slice,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
            height=360,
            command=self._on_slice_change,
        )
        self.slider_cta_slice.pack(fill="y", expand=True, padx=6, pady=(0, 6))
        self._configure_cta_slice_slider(0)

        footer = ctk.CTkFrame(self.workspace, fg_color="transparent")
        footer.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(0, 16), pady=(0, 14))
        footer.grid_columnconfigure(0, weight=1)

        self.lbl_workspace_hint = ctk.CTkLabel(
            footer,
            textvariable=self.var_workspace_hint,
            font=FONTS["small"],
            text_color=THEME["muted"],
            justify="left",
            anchor="w",
        )
        self.lbl_workspace_hint.grid(row=0, column=0, sticky="ew")

    def _select_view(self, view_key: str):
        self.selected_view.set(view_key)
        if view_key == "stenosis":
            self.right_title.configure(text="Mark Stenosis")
        elif view_key == "branch":
            self.right_title.configure(text="Mark Branch")
        else:
            self.right_title.configure(text="Mark Breakers")
        self._apply_button_styles()
        self._refresh_workspace()

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

    def _refresh_segmentation_ui(self):
        self.var_overlay.set(bool(getattr(self.state, "show_coronary_overlay", True)))

        if self.state.cta_segmentation_in_progress:
            status_text = "Running coronary segmentation..."
        elif self.state.cta_segmentation_error:
            status_text = f"{self.state.cta_segmentation_status.capitalize()}: {self.state.cta_segmentation_error}"
        elif self.state.cta_coronary_mask_path:
            status_text = f"Coronary segmentation complete.\nMask: {self.state.cta_coronary_mask_path}"
        else:
            status_text = "No coronary segmentation loaded."

        self.var_segmentation_status.set(status_text)

        is_running = bool(self.state.cta_segmentation_in_progress)
        self.btn_segment.configure(state="disabled" if is_running else "normal")
        self.btn_clear_segmentation.configure(
            state="normal" if self.state.cta_coronary_mask is not None or self.state.cta_coronary_mask_path else "disabled"
        )

    def _refresh_workspace(self):
        volume = getattr(self.state, "cta_volume", None)
        mode = self.selected_view.get().replace("_", " ").title()

        if not volume:
            self._configure_cta_slice_slider(0)
            self.var_workspace_hint.set("Load CTA data to start vessel analysis.")
            self._draw_placeholder("Load CTA data to start vessel analysis.")
            return

        _, z_slices, _, _ = volume["pixels"].shape
        current_slice = min(max(1, int(getattr(self.state, "cta_slice", self.var_cta_slice.get()))), z_slices)
        self.var_cta_slice.set(current_slice)
        self._configure_cta_slice_slider(z_slices)
        self.lbl_cta_slice_val.configure(text=str(current_slice))

        overlay_state = "on" if self.state.cta_coronary_mask is not None and self.state.show_coronary_overlay else "off"
        self.var_workspace_hint.set(
            f"CTA workspace ready. Current mode: {mode}. Coronary overlay: {overlay_state}. Click to place analysis points."
        )
        self._render_cta_viewer()

    def _configure_cta_slice_slider(self, slice_count: int):
        safe_count = max(0, int(slice_count))
        if safe_count <= 1:
            self.slider_cta_slice.configure(from_=1, to=2, number_of_steps=1)
            if self.var_cta_slice.get() != 1:
                self.var_cta_slice.set(1)
            self.lbl_cta_slice_val.configure(text="1")
            return

        current_value = min(max(1, int(self.var_cta_slice.get())), safe_count)
        self.slider_cta_slice.configure(from_=1, to=safe_count, number_of_steps=safe_count - 1)
        if current_value != self.var_cta_slice.get():
            self.var_cta_slice.set(current_value)

    def _toggle_overlay(self):
        self.state.show_coronary_overlay = bool(self.var_overlay.get())
        self._refresh_segmentation_ui()
        self._render_cta_viewer()

    def _clear_segmentation(self):
        self.state.cta_coronary_mask = None
        self.state.cta_coronary_mask_path = ""
        self.state.cta_segmentation_status = "idle"
        self.state.cta_segmentation_error = ""
        self.state.cta_segmentation_in_progress = False
        self._refresh_segmentation_ui()
        self._refresh_workspace()

    def _start_segmentation(self):
        if self.state.cta_segmentation_in_progress:
            return
        if not self.state.cta_folder or not self.state.cta_volume:
            self.state.cta_segmentation_status = "failed"
            self.state.cta_segmentation_error = "Load a CTA folder before running coronary segmentation."
            self._refresh_segmentation_ui()
            self._refresh_workspace()
            return

        self.state.cta_segmentation_status = "running"
        self.state.cta_segmentation_error = ""
        self.state.cta_segmentation_in_progress = True
        self._refresh_segmentation_ui()

        worker = threading.Thread(target=self._run_segmentation_worker, daemon=True)
        worker.start()

    def _run_segmentation_worker(self):
        try:
            print("Running coronary segmentation...")
            mask_path = run_coronary_segmentation(self.state.cta_folder)
            print("Loading coronary mask...")
            mask = load_nifti_mask(mask_path)
            aligned_mask = align_mask_to_cta_volume(mask, self.state.cta_volume["pixels"].shape)
        except Exception as exc:
            if self.winfo_exists():
                self.after(0, lambda message=str(exc): self._on_segmentation_failed(message))
            return

        if self.winfo_exists():
            self.after(0, lambda path=mask_path, mask_data=aligned_mask: self._on_segmentation_complete(path, mask_data))

    def _on_segmentation_complete(self, mask_path: str, mask: object):
        self.state.cta_coronary_mask = mask
        self.state.cta_coronary_mask_path = mask_path
        self.state.cta_segmentation_status = "success"
        self.state.cta_segmentation_error = ""
        self.state.cta_segmentation_in_progress = False
        print("Coronary segmentation complete.")
        self._refresh_segmentation_ui()
        self._refresh_workspace()

    def _on_segmentation_failed(self, message: str):
        self.state.cta_segmentation_status = "failed"
        self.state.cta_segmentation_error = message
        self.state.cta_segmentation_in_progress = False
        print(f"Failed to run coronary segmentation: {message}")
        self._refresh_segmentation_ui()
        self._refresh_workspace()

    def _on_slice_change(self, _value=None):
        value = int(self.var_cta_slice.get())
        self.state.cta_slice = value
        self.lbl_cta_slice_val.configure(text=str(value))
        self._render_cta_viewer()

    def _on_cta_zoom_event(self, event):
        volume = getattr(self.state, "cta_volume", None)
        if not volume:
            return

        delta = getattr(event, "delta", 0)
        num = getattr(event, "num", None)
        if delta > 0 or num == 4:
            factor = 1.1
        elif delta < 0 or num == 5:
            factor = 1 / 1.1
        else:
            return

        self._cta_zoom = min(6.0, max(1.0, self._cta_zoom * factor))
        self._render_cta_viewer()
        return "break"

    def _reset_cta_zoom(self, _event=None):
        self._cta_zoom = 1.0
        self._cta_pan_x = 0.0
        self._cta_pan_y = 0.0
        self._render_cta_viewer()
        return "break"

    def _on_canvas_press(self, event):
        self._cta_drag_origin = (event.x, event.y)
        self._cta_drag_pan_start = (self._cta_pan_x, self._cta_pan_y)
        self._cta_was_dragged = False

    def _on_canvas_drag(self, event):
        volume = getattr(self.state, "cta_volume", None)
        if not volume or self._cta_zoom <= 1.0 or self._cta_drag_origin is None or self._cta_drag_pan_start is None:
            return

        start_x, start_y = self._cta_drag_origin
        pan_start_x, pan_start_y = self._cta_drag_pan_start
        dx = event.x - start_x
        dy = event.y - start_y
        if abs(dx) > 2 or abs(dy) > 2:
            self._cta_was_dragged = True
        self._cta_pan_x = pan_start_x + dx
        self._cta_pan_y = pan_start_y + dy
        self._render_cta_viewer()

    def _on_canvas_release(self, event):
        dragged = self._cta_was_dragged
        self._cta_drag_origin = None
        self._cta_drag_pan_start = None
        self._cta_was_dragged = False
        if not dragged:
            self._on_canvas_click(event)

    def _on_canvas_click(self, event):
        image = self._get_display_image()
        if image is None:
            return

        canvas_w = max(1, int(self.viewer_canvas.winfo_width()))
        canvas_h = max(1, int(self.viewer_canvas.winfo_height()))
        img_h, img_w = image.shape[:2]

        scale = min(canvas_w / max(1, img_w), canvas_h / max(1, img_h)) * self._cta_zoom
        disp_w = max(1, int(img_w * scale))
        disp_h = max(1, int(img_h * scale))
        center_x = (canvas_w / 2.0) + self._cta_pan_x
        center_y = (canvas_h / 2.0) + self._cta_pan_y
        x0 = center_x - (disp_w / 2.0)
        y0 = center_y - (disp_h / 2.0)

        col = int(np.clip((event.x - x0) * img_w / max(1, disp_w), 0, img_w - 1))
        row = int(np.clip((event.y - y0) * img_h / max(1, disp_h), 0, img_h - 1))
        self._last_annotation_point = (int(self.var_cta_slice.get()), row, col, self.selected_view.get())

        mode = self.selected_view.get().replace("_", " ").title()
        self.var_workspace_hint.set(
            f"{mode} placeholder selected at slice {self.var_cta_slice.get()}, row {row}, col {col}."
        )
        self._render_cta_viewer()

    def _draw_placeholder(self, text: str):
        self.viewer_canvas.delete("all")
        self.viewer_placeholder.configure(text=text)
        self.viewer_placeholder.place(in_=self.viewer_canvas, relx=0.5, rely=0.5, anchor="center")

    def _get_display_image(self):
        volume = getattr(self.state, "cta_volume", None)
        if not volume:
            return None

        pixels = volume["pixels"]
        _, z_slices, _, _ = pixels.shape
        z_idx = min(max(0, int(self.var_cta_slice.get()) - 1), z_slices - 1)
        img = pixels[0, z_idx]
        img8 = self._to_uint8_for_display(
            img,
            level=float(getattr(self.state, "cta_level", getattr(self.state, "cta_length", 0.5))),
            width=float(getattr(self.state, "cta_width", 0.5)),
        )
        return self._apply_cta_coronary_overlay(img8, z_idx)

    def _render_cta_viewer(self):
        image = self._get_display_image()
        if image is None:
            self._draw_placeholder("CTA viewer will appear here.\nLoad CTA data, then run coronary segmentation.")
            return

        self.viewer_placeholder.place_forget()

        canvas_w = max(10, int(self.viewer_canvas.winfo_width()))
        canvas_h = max(10, int(self.viewer_canvas.winfo_height()))
        pil = Image.fromarray(image)
        img_w, img_h = pil.size

        scale = min(canvas_w / max(1, img_w), canvas_h / max(1, img_h))
        scale *= self._cta_zoom
        disp_w = max(1, int(img_w * scale))
        disp_h = max(1, int(img_h * scale))

        max_pan_x = max(0.0, (disp_w - canvas_w) / 2.0)
        max_pan_y = max(0.0, (disp_h - canvas_h) / 2.0)
        self._cta_pan_x = float(np.clip(self._cta_pan_x, -max_pan_x, max_pan_x))
        self._cta_pan_y = float(np.clip(self._cta_pan_y, -max_pan_y, max_pan_y))

        photo = ImageTk.PhotoImage(pil.resize((disp_w, disp_h), Image.Resampling.LANCZOS))
        self._cta_photo = photo

        center_x = (canvas_w / 2.0) + self._cta_pan_x
        center_y = (canvas_h / 2.0) + self._cta_pan_y

        self.viewer_canvas.delete("all")
        self.viewer_canvas.create_image(center_x, center_y, image=photo, anchor="center")

        if self._last_annotation_point and self._last_annotation_point[0] == int(self.var_cta_slice.get()):
            _, row, col, _ = self._last_annotation_point
            x0 = center_x - (disp_w / 2.0)
            y0 = center_y - (disp_h / 2.0)
            px = x0 + (col * disp_w / max(1, img_w))
            py = y0 + (row * disp_h / max(1, img_h))
            size = 6
            self.viewer_canvas.create_line(px - size, py, px + size, py, fill="cyan", width=2)
            self.viewer_canvas.create_line(px, py - size, px, py + size, fill="cyan", width=2)

    def _to_uint8_for_display(self, img: np.ndarray, level: float, width: float) -> np.ndarray:
        lo = np.percentile(img, 1)
        hi = np.percentile(img, 99)
        if hi <= lo:
            hi = lo + 1.0

        w_scale = 0.25 + (width * 1.75)
        center = (lo + hi) / 2.0
        shift = (level - 0.5) * (hi - lo) * 0.5
        center = center + shift

        span = (hi - lo) * w_scale
        w_lo = center - span / 2.0
        w_hi = center + span / 2.0

        out = np.clip((img - w_lo) / (w_hi - w_lo), 0, 1)
        return (out * 255.0).astype(np.uint8)

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
