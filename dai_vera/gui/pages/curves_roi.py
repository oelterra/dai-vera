"""
-------------
Pure UI layer for the CTP / ROI curves page.
Zero business logic lives here.

"""

from __future__ import annotations

import tkinter as tk
from typing import Optional, Literal

import numpy as np

import customtkinter as ctk
from customtkinter import CTkButton

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from dai_vera.gui.theme import THEME, FONTS

# ── logic imports ──────────────────────────────────────────────────────────────
from dai_vera.roi_contour import get_contour, get_roi, get_roi_overlayed, ROIObject
from dai_vera.roi_sampling import get_best_sample
from dai_vera.drawlesioncurves import (
    get_fitted_curve,
    plot_sampled_curve,
    plot_fitted_overlay,
)
from dai_vera.roi_json import save_roi_as_json

# window_to_uint8 and make_test_volume still live in curves_roi_logic
# (it was never moved out — keep importing from there)
from dai_vera.curves_roi_logic import window_to_uint8, make_test_volume

LesionType = Literal["pre", "post"]
_SEGMENTATION_WINDOW = 25   # matches MATLAB segmentationWindowSize


class CurvesROIPage(ctk.CTkFrame):
    key = "curves_roi"

    def __init__(self, master, app_state):
        super().__init__(master, fg_color=THEME["bg"])
        self.state = app_state

        self.grid_columnconfigure(0, weight=1, uniform="half")
        self.grid_columnconfigure(1, weight=1, uniform="half")
        self.grid_rowconfigure(0, weight=1)

        # ── layout panels ─────────────────────────────────────────────────────
        self.left_outer = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        self.left_outer.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        self.left_outer.grid_rowconfigure(0, weight=1)
        self.left_outer.grid_columnconfigure(0, weight=1)

        self.left = ctk.CTkScrollableFrame(self.left_outer, fg_color="transparent")
        self.left.grid(row=0, column=0, sticky="nsew")
        self.left.grid_columnconfigure(0, weight=1)

        self.right_outer = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        self.right_outer.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        self.right_outer.grid_rowconfigure(0, weight=1)
        self.right_outer.grid_columnconfigure(0, weight=1)

        self.right = ctk.CTkScrollableFrame(self.right_outer, fg_color="transparent")
        self.right.grid(row=0, column=0, sticky="nsew")
        self.right.grid_columnconfigure(0, weight=1)

        # ── internal state ────────────────────────────────────────────────────
        self.current_x: Optional[int] = None
        self.current_y: Optional[int] = None
        self._last_boundary_coords: Optional[np.ndarray] = None
        self._movie_after_id: Optional[str] = None
        self._drag_start: Optional[tuple] = None
        self._ctp_photo = None                      # keep PhotoImage reference alive

        self.pre_lesion_block  = None
        self.post_lesion_block = None

        # ROI objects set after each lesion button click
        self.pre_roi:  Optional[ROIObject] = None
        self.post_roi: Optional[ROIObject] = None

        # ── build widgets ─────────────────────────────────────────────────────
        self._build_left_ctp_and_controls()
        self._build_right_graphs()

        # ── deferred init ─────────────────────────────────────────────────────
        self.after(80,  self._render_ctp_image)
        self.after(100, self._inject_test_volume)

    


    # =========================================================================
    # LEFT PANEL — CTP viewer + controls
    # =========================================================================

    def _build_left_ctp_and_controls(self) -> None:
        self._build_ctp_panel()
        self._build_controls_panel()

    # ── CTP image panel ───────────────────────────────────────────────────────

    def _build_ctp_panel(self) -> None:
        self.ctp_panel = ctk.CTkFrame(self.left, fg_color=THEME["panel_2"], corner_radius=16)
        self.ctp_panel.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        self.ctp_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.ctp_panel, text="CTP Images", font=FONTS["h1"]).grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 8)
        )

        content = ctk.CTkFrame(self.ctp_panel, fg_color="transparent")
        content.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 10))
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=0)

        # image canvas
        img_box = ctk.CTkFrame(content, fg_color=THEME["panel_3"], corner_radius=14, height=280)
        img_box.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        img_box.grid_propagate(False)

        self.img_canvas = tk.Canvas(img_box, bg="black", highlightthickness=0)
        self.img_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.img_canvas.bind("<Button-1>", self._on_image_click)

        self.lbl_ctp_source = ctk.CTkLabel(
            img_box, text="", font=FONTS["body"], text_color=THEME["muted"]
        )
        self.lbl_ctp_source.place(relx=0.5, rely=0.5, anchor="center")

        # slice (vertical) slider
        slice_col = ctk.CTkFrame(content, fg_color="transparent")
        slice_col.grid(row=0, column=1, sticky="ns")

        ctk.CTkLabel(slice_col, text="Slice", text_color=THEME["muted"], font=FONTS["small"]).pack(pady=(6, 6))

        self.var_ctp_slice = ctk.IntVar(value=int(self.state.ctp_slice))
        self.slider_ctp_slice = ctk.CTkSlider(
            slice_col,
            from_=0, to=100, number_of_steps=100,
            orientation="vertical",
            variable=self.var_ctp_slice,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
            height=220,
            command=self._on_ctp_slice_change,
        )
        self.slider_ctp_slice.pack(padx=6, pady=(0, 6))

        self.lbl_ctp_slice_val = ctk.CTkLabel(slice_col, text=str(self.var_ctp_slice.get()), font=FONTS["small"])
        self.lbl_ctp_slice_val.pack(pady=(0, 8))

        # time (horizontal) slider
        time_row = ctk.CTkFrame(self.ctp_panel, fg_color="transparent")
        time_row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 14))
        time_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(time_row, text="Time Points", text_color=THEME["muted"], font=FONTS["small"]).grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )

        self.var_ctp_time = ctk.IntVar(value=int(self.state.ctp_time))
        self.slider_ctp_time = ctk.CTkSlider(
            time_row,
            from_=0, to=100, number_of_steps=100,
            variable=self.var_ctp_time,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
            command=self._on_ctp_time_change,
        )
        self.slider_ctp_time.grid(row=0, column=1, sticky="ew")

        self.lbl_ctp_time_val = ctk.CTkLabel(time_row, text=str(self.var_ctp_time.get()), font=FONTS["small"])
        self.lbl_ctp_time_val.grid(row=0, column=2, sticky="e", padx=(10, 0))

    # ── controls panel (below image) ──────────────────────────────────────────

    def _build_controls_panel(self) -> None:
        self.controls = ctk.CTkFrame(self.left, fg_color=THEME["panel_2"], corner_radius=16)
        self.controls.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        self.controls.grid_columnconfigure(0, weight=1)

        # row 0 — Sample ROI / Search ROI
        row1 = ctk.CTkFrame(self.controls, fg_color="transparent")
        row1.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))
        row1.grid_columnconfigure(1, weight=1)
        row1.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(row1, text="Sample ROI", font=FONTS["body"]).grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.var_sample_roi = ctk.StringVar(value="2 x 2")
        ctk.CTkOptionMenu(
            row1,
            values=["2 x 2", "4 x 4", "6 x 6", "8 x 8"],
            variable=self.var_sample_roi,
            fg_color=THEME["input_bg"],
            button_color=THEME["border"],
            button_hover_color=THEME["border_2"],
            dropdown_fg_color=THEME["panel_2"],
            dropdown_hover_color=THEME["border"],
            height=34,
        ).grid(row=0, column=1, sticky="ew")

        ctk.CTkLabel(row1, text="Search ROI", font=FONTS["body"]).grid(row=0, column=2, sticky="w", padx=(18, 10))
        self.var_search_roi = ctk.StringVar(value="1 x 1")
        ctk.CTkOptionMenu(
            row1,
            values=["1 x 1", "2 x 2", "3 x 3", "4 x 4"],
            variable=self.var_search_roi,
            fg_color=THEME["input_bg"],
            button_color=THEME["border"],
            button_hover_color=THEME["border_2"],
            dropdown_fg_color=THEME["panel_2"],
            dropdown_hover_color=THEME["border"],
            height=34,
        ).grid(row=0, column=3, sticky="ew")

        # row 1 — Interpolate
        row2 = ctk.CTkFrame(self.controls, fg_color="transparent")
        row2.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        row2.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(row2, text="Interpolate Current Slice", font=FONTS["body"]).grid(row=0, column=0, sticky="w")
        self.var_interpolate = ctk.StringVar(value="With Next Slice")
        ctk.CTkOptionMenu(
            row2,
            values=["With Next Slice", "With Previous Slice", "Off"],
            variable=self.var_interpolate,
            fg_color=THEME["input_bg"],
            button_color=THEME["border"],
            button_hover_color=THEME["border_2"],
            dropdown_fg_color=THEME["panel_2"],
            dropdown_hover_color=THEME["border"],
            height=34,
        ).grid(row=0, column=1, sticky="ew", padx=(12, 0))

        # rows 2-3 — L / W window sliders
        self.var_len = ctk.DoubleVar(value=float(self.state.ctp_length))
        self.var_wid = ctk.DoubleVar(value=float(self.state.ctp_width))
        self._build_slider_line(self.controls, "L", self.var_len, row=2)
        self._build_slider_line(self.controls, "W", self.var_wid, row=3)

        # row 4 — Set pre/post lesion
        row3 = ctk.CTkFrame(self.controls, fg_color="transparent")
        row3.grid(row=4, column=0, sticky="ew", padx=14, pady=(10, 8))
        row3.grid_columnconfigure(0, weight=1)
        row3.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            row3, text="Set Pre Lesion",
            fg_color=THEME["panel_3"], hover_color=THEME["border_2"],
            height=36, corner_radius=12,
            command=lambda: self._on_set_lesion("pre"),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            row3, text="Set Post Lesion",
            fg_color=THEME["panel_3"], hover_color=THEME["border_2"],
            height=36, corner_radius=12,
            command=lambda: self._on_set_lesion("post"),
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        # row 5 — Play movie / speed / height-positive
        row4 = ctk.CTkFrame(self.controls, fg_color="transparent")
        row4.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 14))
        row4.grid_columnconfigure(1, weight=1)

        self.btn_play = ctk.CTkButton(
            row4, text="Play Movie",
            fg_color=THEME["accent"], hover_color=THEME["accent_2"],
            text_color="black",
            height=36, corner_radius=12,
            command=self._toggle_movie,
        )
        self.btn_play.grid(row=0, column=0, sticky="w")

        self.var_speed = ctk.StringVar(value="0.5x")
        ctk.CTkOptionMenu(
            row4,
            values=["0.25x", "0.5x", "0.75x", "1x", "1.25x", "1.5x", "2x"],
            variable=self.var_speed,
            fg_color=THEME["input_bg"],
            button_color=THEME["border"],
            button_hover_color=THEME["border_2"],
            dropdown_fg_color=THEME["panel_2"],
            dropdown_hover_color=THEME["border"],
            height=34,
            width=120,
        ).grid(row=0, column=1, sticky="w", padx=(12, 0))

        self.var_height_positive = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            row4,
            text="Height (positive)",
            variable=self.var_height_positive,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color=THEME["text"],
        ).grid(row=0, column=2, sticky="e", padx=(12, 0))

    def _build_slider_line(self, parent, label: str, var: ctk.DoubleVar, row: int) -> None:
        line = ctk.CTkFrame(parent, fg_color="transparent")
        line.grid(row=row, column=0, sticky="ew", padx=14, pady=6)
        line.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(line, text=label, font=FONTS["body"]).grid(row=0, column=0, sticky="w", padx=(0, 10))
        ctk.CTkSlider(
            line, from_=0, to=1, variable=var,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
            command=lambda _=None: self._sync_window_to_state(),
        ).grid(row=0, column=1, sticky="ew")

    # =========================================================================
    # RIGHT PANEL — curve blocks
    # =========================================================================

    def _build_right_graphs(self) -> None:
        self._build_curve_block(parent=self.right, title="Pre Lesion Curve",  row=0, is_pre=True)
        self._build_curve_block(parent=self.right, title="Post Lesion Curve", row=1, is_pre=False)

    def _build_curve_block(self, parent, title: str, row: int, is_pre: bool) -> None:
        lesion: LesionType = "pre" if is_pre else "post"

        block = ctk.CTkFrame(parent, fg_color=THEME["panel_2"], corner_radius=16)
        block.grid(
            row=row, column=0, sticky="ew",
            padx=14,
            pady=(14, 10) if row == 0 else (0, 14),
        )
        block.grid_columnconfigure(0, weight=1)
        block.grid_rowconfigure(1, weight=1)

        # header
        header = ctk.CTkFrame(block, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text=title, font=FONTS["h1"]).grid(row=0, column=0, sticky="w")

        btns = ctk.CTkFrame(header, fg_color="transparent")
        btns.grid(row=0, column=1, sticky="e")
        ctk.CTkButton(
            btns, text="Undo",
            fg_color=THEME["panel_3"], hover_color=THEME["border_2"],
            height=30, corner_radius=10,
            command=lambda: self._curve_undo(block),
        ).pack(side="left", padx=(0, 10))
        ctk.CTkButton(
            btns, text="Clear",
            fg_color=THEME["panel_3"], hover_color=THEME["border_2"],
            height=30, corner_radius=10,
            command=lambda: self._curve_clear(block),
        ).pack(side="left")

        # matplotlib figure
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 450)
        ax.set_xlabel("Time (s)", labelpad=10)
        ax.set_ylabel("Enhancement (HU)", labelpad=10)
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.18)

        # per-block data
        block.lesion     = lesion
        block.fig        = fig
        block.ax         = ax
        block.times      = []    # sampled time axis
        block.values     = []    # sampled HU values
        block.start_line = ax.axvline(0,  color=THEME["accent"], linewidth=2)
        block.end_line   = ax.axvline(10, color=THEME["accent"], linewidth=2)
        block.selected_idx = None

        canvas = FigureCanvasTkAgg(fig, master=block)
        w = canvas.get_tk_widget()
        w.configure(bg="black", highlightthickness=0)
        w.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        block.canvas = canvas

        # click on plot to select nearest sampled point
        canvas.mpl_connect("button_press_event",
                           lambda evt, b=block: self._on_plot_click(evt, b))

        # ── Start / End range controls (original) ─────────────────────────────
        controls = ctk.CTkFrame(block, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 6))
        controls.grid_columnconfigure(1, weight=1)
        controls.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(controls, text="Start", text_color=THEME["muted"], font=FONTS["small"]).grid(row=0, column=0, sticky="w")
        block.var_start = ctk.IntVar(value=0)
        block.s_start = ctk.CTkSlider(
            controls, from_=0, to=10, number_of_steps=10,
            variable=block.var_start,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
        )
        block.s_start.grid(row=0, column=1, sticky="ew", padx=(8, 16))

        ctk.CTkLabel(controls, text="End", text_color=THEME["muted"], font=FONTS["small"]).grid(row=0, column=2, sticky="w")
        block.var_end = ctk.IntVar(value=10)
        block.s_end = ctk.CTkSlider(
            controls, from_=0, to=10, number_of_steps=10,
            variable=block.var_end,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
        )
        block.s_end.grid(row=0, column=3, sticky="ew", padx=(8, 0))

        def on_range_change(_=None, b=block):
            a = int(b.var_start.get())
            z = int(b.var_end.get())
            if a > z:
                a, z = z, a
                b.var_start.set(a)
                b.var_end.set(z)
            b.start_line.set_xdata([a, a])
            b.end_line.set_xdata([z, z])
            b.canvas.draw_idle()

        block.s_start.configure(command=on_range_change)
        block.s_end.configure(command=on_range_change)
        on_range_change()

        # ── Baseline / Washout inputs + Fit Curve (new) ───────────────────────
        fit_row = ctk.CTkFrame(block, fg_color="transparent")
        fit_row.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 6))

        ctk.CTkLabel(fit_row, text="Baseline", text_color=THEME["muted"],
                     font=FONTS["small"]).pack(side="left")
        block.var_baseline = ctk.StringVar(value="0")
        ctk.CTkEntry(
            fit_row, textvariable=block.var_baseline,
            width=52, height=28,
            fg_color=THEME["input_bg"], border_color=THEME["border"],
        ).pack(side="left", padx=(6, 16))

        ctk.CTkLabel(fit_row, text="Washout", text_color=THEME["muted"],
                     font=FONTS["small"]).pack(side="left")
        block.var_washout = ctk.StringVar(value="0")
        ctk.CTkEntry(
            fit_row, textvariable=block.var_washout,
            width=52, height=28,
            fg_color=THEME["input_bg"], border_color=THEME["border"],
        ).pack(side="left", padx=(6, 16))

        ctk.CTkButton(
            fit_row, text="Fit Curve",
            fg_color=THEME["accent"], hover_color=THEME["accent_2"],
            text_color="black", height=30, corner_radius=10,
            command=lambda b=block: self._on_fit_curve(b),
        ).pack(side="left")

        # ── Edit Point / Remove Point (new) ───────────────────────────────────
        point_row = ctk.CTkFrame(block, fg_color="transparent")
        point_row.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 14))

        ctk.CTkButton(
            point_row, text="Edit Point",
            fg_color=THEME["panel_3"], hover_color=THEME["border_2"],
            height=30, corner_radius=10,
            command=lambda b=block: self._on_edit_point(b),
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            point_row, text="Remove Point",
            fg_color=THEME["panel_3"], hover_color=THEME["border_2"],
            height=30, corner_radius=10,
            command=lambda b=block: self._on_remove_point(b),
        ).pack(side="left")

        block.lbl_selected = ctk.CTkLabel(
            point_row, text="No point selected",
            text_color=THEME["muted"], font=FONTS["small"],
        )
        block.lbl_selected.pack(side="left", padx=(12, 0))

        if is_pre:
            self.pre_lesion_block = block
        else:
            self.post_lesion_block = block

    # =========================================================================
    # Rendering helpers
    # =========================================================================

    def _render_ctp_image(self) -> None:
        """Render the current CTP slice/time WITHOUT reconfiguring sliders during playback"""
        vol = getattr(self.state, "ctp_volume", None)
        if not vol:
            self.lbl_ctp_source.configure(text="No CTP loaded")
            return

        pixels: np.ndarray = vol["pixels"]  # (T, Z, H, W)
        T, Z, H, W = pixels.shape
        # Clamp sliders to valid ranges to avoid -1 indices
        self.var_ctp_slice.set(min(max(1, self.var_ctp_slice.get()), Z))
        self.var_ctp_time.set(min(max(1, self.var_ctp_time.get()), T))
        self.lbl_ctp_slice_val.configure(text=str(self.var_ctp_slice.get()))
        self.lbl_ctp_time_val.configure(text=str(self.var_ctp_time.get()))
        print("Volume shape:", pixels.shape)


        # Only reconfigure sliders if they're not initialized properly
        # This prevents slider reset during movie playback
        if self.slider_ctp_slice.cget("to") != Z or self.slider_ctp_time.cget("to") != T:

            slice_steps = max(1, Z - 1)
            time_steps  = max(1, T - 1)

            self.slider_ctp_slice.configure(
                from_=1,
                to=max(1, Z),
                number_of_steps=slice_steps
            )

            self.slider_ctp_time.configure(
                from_=1,
                to=max(1, T),
                number_of_steps=time_steps
            )
        t_idx = int(self.var_ctp_time.get()) - 1
        z_idx = int(self.var_ctp_slice.get()) - 1

        # Clamp indices
        t_idx = min(max(0, t_idx), T - 1)
        z_idx = min(max(0, z_idx), Z - 1)

        img8 = window_to_uint8(
            pixels[t_idx, z_idx],
            length=float(getattr(self.state, "ctp_length", 0.5)),
            width=float(getattr(self.state, "ctp_width",  0.5)),
        )

        cw = max(10, self.img_canvas.winfo_width())
        ch = max(10, self.img_canvas.winfo_height())

        photo = ImageTk.PhotoImage(Image.fromarray(img8).resize((cw, ch)))
        self._ctp_photo = photo   # keep reference alive

        self.img_canvas.delete("all")
        self.img_canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")
        self.lbl_ctp_source.configure(text="")

        if self._last_boundary_coords is not None:
            self._draw_contour_on_canvas(self._last_boundary_coords)

    def _redraw_curve(self, block, _lesion_type=None) -> None:
        """Fixed signature to handle the optional lesion_type argument from Undo/Clear"""
        ax = block.ax
        
        # Clear previous fitted lines, but keep the axis setup
        for line in ax.lines[:]:
            if line not in (block.start_line, block.end_line):
                line.remove()
                
        if block.times:
            # Re-plot dots using your custom plotter
            plot_sampled_curve(
                ax=block.ax,
                times=np.array(block.times),
                values=np.array(block.values),
                lesion_type=block.lesion,
                time_unit="s",
            )
        block.canvas.draw_idle()

    # ── canvas overlays ───────────────────────────────────────────────────────

    def _draw_contour_on_canvas(self, boundary_coords: np.ndarray) -> None:
        """Render vessel boundary dots over the CTP image."""
        self._last_boundary_coords = boundary_coords
        self.img_canvas.delete("roi_contour")
        for x, y in boundary_coords:
            self.img_canvas.create_oval(
                y, x, y + 2, x + 2,
                fill="lime", outline="", tags="roi_contour",
            )

    def _draw_crosshair(self, x: int, y: int) -> None:
        self.img_canvas.delete("crosshair")
        size = 6
        self.img_canvas.create_line(x - size, y, x + size, y, fill="red", width=2, tags="crosshair")
        self.img_canvas.create_line(x, y - size, x, y + size, fill="red", width=2, tags="crosshair")

    def _add_draggable_pre_point(self, cx: int, cy: int) -> None:
        self.img_canvas.delete("drag_point")
        self.img_canvas.tag_unbind("drag_point", "<ButtonPress-1>")
        self.img_canvas.tag_unbind("drag_point", "<B1-Motion>")
        self.img_canvas.tag_unbind("drag_point", "<ButtonRelease-1>")

        r = 6
        self.img_canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline="cyan", width=2, fill="", tags="drag_point",
        )

        def on_drag(event):
            self.img_canvas.delete("drag_point")
            self.img_canvas.create_oval(
                event.x - r, event.y - r, event.x + r, event.y + r,
                outline="cyan", width=2, fill="", tags="drag_point",
            )
            self._draw_crosshair(event.x, event.y)

        def on_release(event):
            self.current_x = event.x
            self.current_y = event.y
            self._on_set_lesion("pre")

        self.img_canvas.tag_bind("drag_point", "<ButtonPress-1>",   lambda e: None)
        self.img_canvas.tag_bind("drag_point", "<B1-Motion>",       on_drag)
        self.img_canvas.tag_bind("drag_point", "<ButtonRelease-1>", on_release)


    # draggable point for both pre/post lesions:
    def _add_draggable_point(self, cx: int, cy: int, lesion: LesionType) -> None:
        """Create a draggable crosshair point that sets ROI on release."""
        self._current_drag_lesion = lesion
        self.img_canvas.delete("drag_point")
        r = 6
        self.img_canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline="cyan", width=2, fill="", tags="drag_point",
        )

        def on_drag(event):
            self.img_canvas.delete("drag_point")
            self.img_canvas.create_oval(
                event.x - r, event.y - r, event.x + r, event.y + r,
                outline="cyan", width=2, fill="", tags="drag_point",
            )
            self._draw_crosshair(event.x, event.y)

        def on_release(event):
            self.current_x = event.x
            self.current_y = event.y
            if self._current_drag_lesion:
                self._on_set_lesion(self._current_drag_lesion)

        self.img_canvas.tag_bind("drag_point", "<B1-Motion>", on_drag)
        self.img_canvas.tag_bind("drag_point", "<ButtonRelease-1>", on_release)

    # ── curve range / axes sync ───────────────────────────────────────────────

    def _sync_curve_block_from_data(self, block, times: list, values: list) -> None:
        """Reconfigure range sliders and axis limits to match freshly sampled curve data."""
        if not times:
            return

        ax = block.ax
        ax.set_xlim(times[0] - 0.5, times[-1] + 0.5)
        v_min, v_max = min(values), max(values)
        pad = max(1.0, (v_max - v_min) * 0.15)
        ax.set_ylim(v_min - pad, v_max + pad)

        t_start = int(round(times[0]))
        t_end   = int(round(times[-1]))
        n_steps = max(1, len(times) - 1)

        block.var_start.set(t_start)
        block.var_end.set(t_end)
        block.s_start.configure(from_=t_start, to=max(t_start + 1, t_end), number_of_steps=n_steps)
        block.s_end.configure(from_=t_start,   to=max(t_start + 1, t_end), number_of_steps=n_steps)
        block.start_line.set_xdata([t_start, t_start])
        block.end_line.set_xdata([t_end,   t_end])

    # =========================================================================
    # Event handlers
    # =========================================================================

    def _on_image_click(self, event) -> None:
        self.current_x = event.x
        self.current_y = event.y
        self._draw_crosshair(event.x, event.y)

    def _on_ctp_slice_change(self, _=None) -> None:
        self.lbl_ctp_slice_val.configure(text=str(int(self.var_ctp_slice.get())))
        self.state.ctp_slice = int(self.var_ctp_slice.get())
        self._render_ctp_image()

    def _on_ctp_time_change(self, _=None):
        val = int(self.var_ctp_time.get())
        self.lbl_ctp_time_val.configure(text=str(val))
        self.state.ctp_time = val
        self._render_ctp_image() 

    def _sync_window_to_state(self) -> None:
        self.state.ctp_length = float(self.var_len.get())
        self.state.ctp_width  = float(self.var_wid.get())
        self._render_ctp_image()

    def _toggle_movie(self) -> None:
        if self._movie_after_id is not None:
            self.after_cancel(self._movie_after_id)
            self._movie_after_id = None
            self.btn_play.configure(text="Play Movie")
        else:
            self.btn_play.configure(text="Stop")
            self._movie_loop()

    def _movie_loop(self) -> None:
        """FIXED: Properly cycles through time points without slider conflicts"""
        vol = getattr(self.state, "ctp_volume", None)
        if not vol:
            print("No CTP volume loaded")
            self._movie_after_id = None
            self.btn_play.configure(text="Play Movie")
            return

        max_t = vol["pixels"].shape[0]
        try:
            speed = float(self.var_speed.get().replace("x", ""))
        except ValueError:
            speed = 0.5

        delay = int(max(40, 250 / max(speed, 0.1)))

        # get current 1-based value from slider
        current_val = int(self.var_ctp_time.get())
        # increment and wrap (1-based indexing)
        next_val = (current_val % max_t) + 1
        
        # update slider and state
        self.var_ctp_time.set(next_val)
        self.state.ctp_time = next_val
        self.lbl_ctp_time_val.configure(text=str(next_val))
        
        # render without reconfiguring sliders
        self._render_ctp_image()
        
        self._movie_after_id = self.after(delay, self._movie_loop)

    def _curve_undo(self, block) -> None:
        if block.times:
            block.times.pop()
            block.values.pop()
            block.selected_idx = None
            # Pass the block's internal lesion type ('pre' or 'post')
            self._redraw_curve(block, lesion_type=block.lesion)

    def _curve_clear(self, block) -> None:
        block.times = []
        block.values = []
        block.selected_idx = None
        # Fix the typo "prep" -> block.lesion
        self._redraw_curve(block, lesion_type=block.lesion)

    # =========================================================================
    # New: Set Lesion handler (FIXED graph plotting)
    # =========================================================================
    
    def _on_set_lesion(self, lesion: LesionType) -> None:

        if self.current_x is None or self.current_y is None:
            print(f"[{lesion}] Click on the image first")
            return

        vol = getattr(self.state, "ctp_volume", None)
        if not vol:
            print("No CTP volume loaded")
            return

        pixels: np.ndarray = vol["pixels"]  # (T, Z, H, W)
        T, Z, H, W = pixels.shape
        time_points   = np.asarray(vol.get("times", np.arange(T, dtype=float)))
        pixel_spacing = vol.get("pixel_spacing", (0.5, 0.5))

        t_idx = min(max(0, int(self.var_ctp_time.get())  - 1), T - 1)
        z_idx = min(max(0, int(self.var_ctp_slice.get()) - 1), Z - 1)

        click_row = self.current_y
        click_col = self.current_x
        sample_n  = int(self.var_sample_roi.get().split("x")[0].strip())
        search_n  = int(self.var_search_roi.get().split("x")[0].strip())
        search_px = max(search_n * 20, 20)

        # 1. get_best_sample
        sample = get_best_sample(
            x                 = click_row,
            y                 = click_col,
            window_size       = search_px,
            roi_m             = sample_n,
            roi_n             = sample_n,
            slice_idx         = z_idx,
            four_d_image_set  = pixels,
            time_point_values = time_points,
        )
        interp_times = sample["interpolated_time_points"]
        interp_vals  = sample["interpolated_sampled_points"]
        sx_rows      = sample["search_window_rows"]
        sx_cols      = sample["search_window_cols"]

        if len(interp_times) == 0 or len(interp_vals) == 0:
            print("ERROR: sampled points are empty!")
            return
        

        # 2. get_contour
        contour = get_contour(
            x                  = click_row,
            y                  = click_col,
            search_window_size = _SEGMENTATION_WINDOW,
            slice_idx          = z_idx,
            time_point_idx     = t_idx,
            four_d_image_set   = pixels,
            pixel_spacing      = pixel_spacing,
        )
        print(f"[{lesion}] radius={contour.radius_cm:.3f} cm  area={contour.area_cm2:.4f} cm²")

        # 3. get_roi
        roi_obj = get_roi(
            study_name         = getattr(self.state, "study_name", "Unknown"),
            num_time_points    = T,
            num_slices         = Z,
            x = click_row, y = click_col, z = z_idx, t = t_idx,
            sampled_curve      = interp_vals.tolist(),
            time_points        = time_points.tolist(),
            fitted_curve       = [],
            fitted_time_points = [],
            roi_x_boundary     = sx_rows.tolist(),
            roi_y_boundary     = sx_cols.tolist(),
        )
        if lesion == "pre":
            self.pre_roi  = roi_obj
            # block = self.pre_lesion_block
        else:
            self.post_roi = roi_obj
            # block = self.post_lesion_block

        # 4. save_roi_as_json
        bundle = {
            "preRoiObject":  self._roi_to_dict(self.pre_roi),
            "postRoiObject": self._roi_to_dict(self.post_roi),
        }
        path = save_roi_as_json(bundle)
        print(f"ROI saved → {path}")

        # 5. get_roi_overlayed → burn boundary → re-render canvas
        image_2d = pixels[t_idx, z_idx].copy().astype(np.float32)
        overlaid = get_roi_overlayed(image_2d, sx_rows, sx_cols, overlay_value=1500.0)
        # render it directly 
        self._render_ctp_image_with(overlaid)
        # then draw crosshair 
        self._draw_crosshair(self.current_x, self.current_y)
        self._add_draggable_pre_point(self.current_x, self.current_y)

        # 6. Use plot_sampled_curve from drawlesioncurves.py
        block = self.pre_lesion_block if lesion == "pre" else self.post_lesion_block
        block.times  = interp_times.tolist()
        block.values = interp_vals.tolist()
        block.selected_idx = None
        

        # Clear previous points and plot freshly
        block.ax.cla()  # clear axes

        # Use the proper plotting function instead of manual plotting
        plot_sampled_curve(
            ax          = block.ax,
            times       = np.array(block.times),
            values      = np.array(block.values),
            lesion_type = lesion,
            time_unit   = "s",
        )

        # Reset axes style exactly like MATLAB
        block.ax.set_facecolor("black")
        block.ax.set_xlabel("Time (s)", color="white")
        block.ax.set_ylabel("Enhancement (HU)", color="white")

        for spine in block.ax.spines.values():
            spine.set_color("white")
        block.ax.tick_params(colors="white")

        # Sync sliders to data range
        self._sync_curve_block_from_data(block, block.times, block.values)

        # block.ax.set_xlim(min(block.times) - 0.5, max(block.times) + 0.5)
        # v_min, v_max = min(block.values), max(block.values)
        # pad = max(1.0, (v_max - v_min) * 0.15)
        # block.ax.set_ylim(v_min - pad, v_max + pad)
        # block.ax.plot(block.times, block.values, 'o', color="dodgerblue" if lesion=="pre" else "tomato")
        block.canvas.draw()

    def _render_ctp_image_with(self, image_override: np.ndarray) -> None:
        """Render an already-processed 2-D float image onto the canvas."""
        img8 = window_to_uint8(
            image_override,
            length=float(getattr(self.state, "ctp_length", 0.5)),
            width=float(getattr(self.state, "ctp_width",  0.5)),
        )
        cw = max(10, self.img_canvas.winfo_width())
        ch = max(10, self.img_canvas.winfo_height())
        photo = ImageTk.PhotoImage(Image.fromarray(img8).resize((cw, ch)))
        self._ctp_photo = photo
        self.img_canvas.delete("all")
        self.img_canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")

    # =========================================================================
    # New: Fit Curve
    # =========================================================================

    def _on_fit_curve(self, block) -> None:
        if not block.times:
            print(f"[{block.lesion}] Set lesion first")
            return
        
        print(f"\n=== DEBUG block.values ===")
        print(f"block.values: {block.values[:10] if len(block.values) >= 10 else block.values}")
        print(f"  min={np.min(block.values):.1f}, max={np.max(block.values):.1f}")
        print(f"  mean={np.mean(block.values):.1f}")
        print(f"========================\n")


        try:
            # Ensure we are getting the text from the Entry widgets
            baseline = int(block.var_baseline.get())
            washout = int(block.var_washout.get())
        except ValueError:
            baseline, washout = 0, 0


        try:
            # CALLING THE MATH ENGINE
            result = get_fitted_curve(
                sample_curve = np.asarray(block.values, dtype=float),
                sample_time  = np.asarray(block.times,  dtype=float),
                baseline     = baseline,
                washout      = washout,
            )
        except Exception as exc:
            print(f"[{block.lesion}] Fit failed: {exc}")
            return

        # Log for debugging - check if K is still tiny here
        print(f"[{block.lesion}] K={result.k:.3f} RMSE={result.rmse:.3f}")

        # PLOTTING THE RESULT
        plot_fitted_overlay(
            ax           = block.ax,
            fitted_time  = result.fitted_time,
            fitted_curve = result.fitted_curve,
            lesion_type  = block.lesion,
            # fit_error    = result.rmse # This shows the accuracy in the legend
        )

        block.canvas.draw()

    # =========================================================================
    # New: Plot click → select point / Edit / Remove
    # =========================================================================

    def _on_plot_click(self, event, block) -> None:
        """
        Click on graph to select the nearest sampled point for editing/removal.
        MATLAB equivalent: Sets up for drawpoint() / getpts() operations.
        """
        if event.xdata is None or event.ydata is None or not block.times:
            return
        
        times = np.asarray(block.times)
        values = np.asarray(block.values)
        
        # Find nearest point (Euclidean distance in plot coordinates)
        # Normalize by axis ranges for fair distance calculation
        x_range = block.ax.get_xlim()[1] - block.ax.get_xlim()[0]
        y_range = block.ax.get_ylim()[1] - block.ax.get_ylim()[0]
        
        dx = (times - event.xdata) / x_range
        dy = (values - event.ydata) / y_range
        dists = np.sqrt(dx**2 + dy**2)
        
        idx = int(np.argmin(dists))
        block.selected_idx = idx
        
        # Highlight the selected point
        self._highlight_selected_point(block, idx)
        
        block.lbl_selected.configure(
            text=f"Point {idx + 1}: t={times[idx]:.2f}s  {values[idx]:.1f} HU"
        )


    def _highlight_selected_point(self, block, idx: int) -> None:
        """
        Draw a larger marker around the selected point.
        """
        # Remove previous highlight
        for artist in block.ax.artists:
            if hasattr(artist, '_highlight_marker'):
                artist.remove()
        
        times = block.times
        values = block.values
        colour = _COLOUR.get(block.lesion, "white")
        
        # Draw highlight circle
        highlight = block.ax.plot(
            times[idx], values[idx], 
            'o', 
            markersize=12, 
            markerfacecolor='none',
            markeredgecolor='yellow',
            markeredgewidth=2,
            alpha=0.8
        )[0]
        highlight._highlight_marker = True  # tag it for removal
        
        block.canvas.draw_idle()

    def _on_edit_point(self, block) -> None:
        """
        MATLAB equivalent: EditPointCheckBoxValueChanged
        Let user click directly on the graph to select and modify a point.
        """
        if not block.times:
            print(f"[{block.lesion}] No curve data to edit")
            return
        
        # Simple implementation: use the already-selected point from _on_plot_click
        if block.selected_idx is None:
            print(f"[{block.lesion}] Click a point on the graph first to select it")
            return
        
        # Show dialog to get new HU value
        from tkinter import simpledialog
        idx = block.selected_idx
        current_time = block.times[idx]
        current_hu = block.values[idx]
        
        new_hu = simpledialog.askfloat(
            "Edit Point",
            f"Point at t={current_time:.2f}s\nCurrent HU: {current_hu:.1f}\n\nEnter new HU value:",
            initialvalue=current_hu,
            minvalue=-100,
            maxvalue=1000,
        )
        
        if new_hu is None:  # user cancelled
            return
        
        # Update the value
        block.values[idx] = float(new_hu)
        block.lbl_selected.configure(
            text=f"Point {idx + 1}: t={current_time:.2f}s  {new_hu:.1f} HU (edited)"
        )
        
        # MATLAB: Clear fitted curve when sampled points are modified
        self._clear_fitted_curve(block)
        
        # Redraw sampled curve
        plot_sampled_curve(
            ax=block.ax,
            times=np.array(block.times),
            values=np.array(block.values),
            lesion_type=block.lesion,
            time_unit="s",
        )
        
        # Re-add the range lines (they get cleared by plot_sampled_curve)
        self._restore_range_lines(block)
        
        block.canvas.draw()
        
        # Update ROI object
        self._update_roi_curve_data(block)
        print(f"[{block.lesion}] Point {idx + 1} updated to {new_hu:.1f} HU")


    def _on_remove_point(self, block) -> None:
        """
        MATLAB equivalent: RemovePointPreCheckBoxStressOnlyValueChanged
        Remove the selected point from the curve.
        """
        if block.selected_idx is None:
            print(f"[{block.lesion}] Click a point on the graph first to select it")
            return
        
        if len(block.times) <= 3:
            print(f"[{block.lesion}] Cannot remove point - need at least 3 points for fitting")
            return
        
        idx = block.selected_idx
        removed_time = block.times[idx]
        removed_hu = block.values[idx]
        
        # Remove from both arrays (MATLAB: dataArray(idx) = [])
        block.times.pop(idx)
        block.values.pop(idx)
        block.selected_idx = None
        block.lbl_selected.configure(text="Point removed")
        
        # MATLAB: Clear fitted curve when sampled points are modified
        self._clear_fitted_curve(block)
        
        # Redraw sampled curve
        plot_sampled_curve(
            ax=block.ax,
            times=np.array(block.times),
            values=np.array(block.values),
            lesion_type=block.lesion,
            time_unit="s",
        )
        
        # Re-add the range lines
        self._restore_range_lines(block)
        
        # Sync sliders to new data range
        self._sync_curve_block_from_data(block, block.times, block.values)
        
        block.canvas.draw()
        
        # Update ROI object
        self._update_roi_curve_data(block)
        print(f"[{block.lesion}] Removed point at t={removed_time:.2f}s ({removed_hu:.1f} HU)")

    # =========================================================================
    # Helper methods for Edit/Remove operations
    # =========================================================================

    def _clear_fitted_curve(self, block) -> None:
        """
        Clear fitted curve data from block and ROI object.
        MATLAB: app.fittedCurvePaStressOnly = []
        """
        if block.lesion == "pre" and self.pre_roi:
            self.pre_roi.fitted_curve = []
            self.pre_roi.fitted_time_points = []
        elif block.lesion == "post" and self.post_roi:
            self.post_roi.fitted_curve = []
            self.post_roi.fitted_time_points = []


    def _restore_range_lines(self, block) -> None:
        """
        Re-add the start/end range lines after plot_sampled_curve clears them.
        """
        t_start = int(block.var_start.get())
        t_end = int(block.var_end.get())
        
        # Remove old lines if they exist
        if hasattr(block, 'start_line') and block.start_line in block.ax.lines:
            block.start_line.remove()
        if hasattr(block, 'end_line') and block.end_line in block.ax.lines:
            block.end_line.remove()
        
        # Add new lines
        block.start_line = block.ax.axvline(t_start, color=THEME["accent"], linewidth=2)
        block.end_line = block.ax.axvline(t_end, color=THEME["accent"], linewidth=2)


    def _update_roi_curve_data(self, block) -> None:
        """
        Update the ROI object with current sampled curve data and save to JSON.
        MATLAB equivalent: updates app.paCurveStressOnly, then saveRoiAsJson
        """
        if block.lesion == "pre" and self.pre_roi:
            self.pre_roi.curve = block.values.copy()
            self.pre_roi.time_points = block.times.copy()
        elif block.lesion == "post" and self.post_roi:
            self.post_roi.curve = block.values.copy()
            self.post_roi.time_points = block.times.copy()
        
        # Save to JSON
        bundle = {
            "preRoiObject":  self._roi_to_dict(self.pre_roi),
            "postRoiObject": self._roi_to_dict(self.post_roi),
        }
        path = save_roi_as_json(bundle)
        print(f"ROI updated → {path}")

    # =========================================================================
    # Dev helper — inject synthetic volume
    # =========================================================================

    def _inject_test_volume(self) -> None:
        if getattr(self.state, "ctp_volume", None) is None:

            vol = make_test_volume()

            self.state.ctp_volume = {
                "pixels": vol["pixels"],
                "times": vol["times"],
                "zs": vol["zs"],
                "shape": vol["shape"],
            }

            self.state.ctp_slice = 1
            self.state.ctp_time = 1

            print("Volume shape:", vol["pixels"].shape)

            self._render_ctp_image()

    # helpers

    @staticmethod
    def _roi_to_dict(roi: Optional[ROIObject]) -> dict:
        if roi is None:
            return {}
        return {
            "x": roi.x, "y": roi.y, "z": roi.z, "t": roi.t,
            "curve":            roi.curve,
            "timePoints":       roi.time_points,
            "fittedCurve":      roi.fitted_curve,
            "fittedTimePoints": roi.fitted_time_points,
            "roiXBoundary":     roi.roi_x_boundary,
            "roiYBoundary":     roi.roi_y_boundary,
        }