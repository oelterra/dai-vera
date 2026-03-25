"""
curves_roi_page.py  —  FIXED VERSION
-------------------------------------
Fixes applied:
  1. Search ROI rectangle drawn with correct canvas-scaled coordinates
  2. Search ROI size respects the dropdown (1x1→20px, 2x2→40px, etc.)
  3. Undo re-uses tight axis limits from data — never resets to -50..500
  4. Baseline / Washout entries trigger re-fit on Enter / FocusOut
  5. Baseline/Washout slider sync works bidirectionally
  6. baseline_hu_offset NameError fixed in get_fitted_curve (patch applied here)
  7. _redraw_curve keeps tight y-limits and does NOT call _configure_axes
"""

from __future__ import annotations

import tkinter as tk
from typing import Optional, Literal

import numpy as np

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from dai_vera.gui.theme import THEME, FONTS

# ── logic imports ──────────────────────────────────────────────────────────────
from dai_vera.roi_contour import get_contour, get_roi, get_roi_overlayed, ROIObject
from dai_vera.roi_sampling import get_best_sample
from dai_vera.drawlesioncurves import (
    plot_sampled_curve,
    plot_fitted_overlay,
    get_fitted_curve_safe,
    _COLOUR,
)
from dai_vera.roi_json import save_roi_as_json
from dai_vera.curves_roi_logic import window_to_uint8, make_test_volume

# Patch get_fitted_curve inline so baseline_hu_offset is always defined
from dai_vera import drawlesioncurves as _dlc


def _get_fitted_curve_fixed(sample_curve, sample_time, baseline=0, washout=0):
    """
    Thin wrapper around the original get_fitted_curve that ensures
    baseline_hu_offset is always defined before it is referenced.
    """
    from dai_vera.roi_curve_processing import preprocess_curve, find_baseline
    from dai_vera.gammavariate import fit_modified_gamma_variate, compute_auc

    sample_curve = np.asarray(sample_curve, dtype=float).flatten()
    sample_time  = np.asarray(sample_time,  dtype=float).flatten()

    # ms → s
    if len(sample_time) > 1 and sample_time[1] >= 500:
        sample_time = sample_time / 1000.0

    if baseline != 0:
        position = int(baseline)
    else:
        position = find_baseline(sample_curve) + 1
    if position == 0:
        position = 1

    processed = preprocess_curve(
        raw_tdc           = sample_curve,
        time_points       = sample_time,
        washout_point     = washout if washout != 0 else None,
        baseline_override = position,
    )

    baseline_subtracted = processed["subtracted_curve"]
    baseline_pos_0      = processed["baseline_position"]
    recirc_start        = processed["recirculation_start"]
    baseline_hu_offset  = float(processed.get("baseline_value", 0.0))   # ← FIX

    peak_value = float(np.max(baseline_subtracted))
    peak_idx   = int(np.argmax(baseline_subtracted))
    t_at       = float(sample_time[max(0, position - 1)])
    t_peak     = float(sample_time[peak_idx])

    K_init     = max(peak_value, 1.0)
    alpha_init = 2.0
    beta_init  = max((t_peak - t_at) / (alpha_init + 1.0), 0.5)

    print(f"  K_init={K_init:.2f}  α_init={alpha_init:.2f}  β_init={beta_init:.2f}")

    try:
        fit = fit_modified_gamma_variate(
            time                   = sample_time,
            data                   = baseline_subtracted,
            K_init                 = K_init,
            alpha_init             = alpha_init,
            beta_init              = beta_init,
            num_points_to_consider = recirc_start,
            contrast_arrival_time  = position,
        )
        converged = fit.converged
    except Exception as exc:
        raise RuntimeError(f"fitModifiedGammaVariate failed: {exc}") from exc

    linear_ref = np.linspace(float(baseline_subtracted[0]),
                             float(baseline_subtracted[-1]), 100)
    rmse = float(np.sqrt(np.mean((fit.fitted_data - linear_ref) ** 2)))
    auc  = compute_auc(fit.fitted_data, fit.stretched_time)

    from dai_vera.drawlesioncurves import FittedCurveResult
    return FittedCurveResult(
        fitted_curve              = fit.fitted_data + baseline_hu_offset,
        fitted_time               = fit.stretched_time,
        baseline_subtracted_curve = baseline_subtracted,
        rmse                      = rmse,
        k                         = fit.k,
        alpha                     = fit.alpha,
        beta                      = fit.beta,
        baseline_position         = baseline_pos_0,
        recirculation_start       = recirc_start,
        auc                       = auc,
        converged                 = converged,
    )


LesionType = Literal["pre", "post"]
_SEGMENTATION_WINDOW = 25


class CurvesROIPage(ctk.CTkFrame):
    key = "curves_roi"

    def __init__(self, master, app_state):
        super().__init__(master, fg_color=THEME["bg"])
        self.state = app_state

        self.grid_columnconfigure(0, weight=7)
        self.grid_columnconfigure(1, weight=4)
        self.grid_rowconfigure(0, weight=1)

        # ── layout panels ─────────────────────────────────────────────────────
        self.left_outer = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        self.left_outer.grid(row=0, column=0, sticky="nsew", padx=(8, 5), pady=8)
        self.left_outer.grid_rowconfigure(0, weight=1)
        self.left_outer.grid_columnconfigure(0, weight=1)

        self.left = ctk.CTkFrame(self.left_outer, fg_color="transparent")
        self.left.grid(row=0, column=0, sticky="nsew")
        self.left.grid_columnconfigure(0, weight=1)
        self.left.grid_rowconfigure(0, weight=5)
        self.left.grid_rowconfigure(1, weight=2)

        self.right_outer = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        self.right_outer.grid(row=0, column=1, sticky="nsew", padx=(5, 8), pady=8)
        self.right_outer.grid_rowconfigure(0, weight=1)
        self.right_outer.grid_columnconfigure(0, weight=1)

        self.right = ctk.CTkFrame(self.right_outer, fg_color="transparent")
        self.right.grid(row=0, column=0, sticky="nsew")
        self.right.grid_columnconfigure(0, weight=1)
        self.right.grid_rowconfigure(0, weight=1)
        self.right.grid_rowconfigure(1, weight=1)

        # ── internal state ────────────────────────────────────────────────────
        self.current_x: Optional[int] = None
        self.current_y: Optional[int] = None
        self._last_boundary_coords: Optional[np.ndarray] = None
        self._movie_after_id: Optional[str] = None
        self._ctp_photo = None
        self._current_drag_lesion: Optional[LesionType] = None
        self._ctp_zoom = 1.0
        self._ctp_display_rect: Optional[tuple[float, float, float, float, int, int]] = None
        self._overlay_image: Optional[np.ndarray] = None
        self._ctp_pan_x = 0.0
        self._ctp_pan_y = 0.0
        self._ctp_drag_origin: Optional[tuple[int, int]] = None
        self._ctp_drag_pan_start: Optional[tuple[float, float]] = None
        self._ctp_was_dragged = False

        # last search-ROI rectangle in IMAGE pixel space (for redraw on slice change)
        self._last_search_roi_img: Optional[tuple] = None   # (r0,c0,r1,c1)

        self.pre_lesion_block  = None
        self.post_lesion_block = None
        self.pre_roi:  Optional[ROIObject] = None
        self.post_roi: Optional[ROIObject] = None

        # ── build widgets ─────────────────────────────────────────────────────
        self._build_left_ctp_and_controls()
        self._build_right_graphs()

        self.after(80, self._render_ctp_image)

    # =========================================================================
    # LEFT PANEL
    # =========================================================================

    def _build_left_ctp_and_controls(self) -> None:
        self._build_ctp_panel()
        self._build_controls_panel()

    def _build_ctp_panel(self) -> None:
        self.ctp_panel = ctk.CTkFrame(self.left, fg_color=THEME["panel_2"], corner_radius=16)
        self.ctp_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))
        self.ctp_panel.grid_columnconfigure(0, weight=1)
        self.ctp_panel.grid_rowconfigure(1, weight=1)

        content = ctk.CTkFrame(self.ctp_panel, fg_color="transparent")
        content.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 2))
        content.grid_columnconfigure(0, weight=0)
        content.grid_columnconfigure(1, weight=1)
        content.grid_columnconfigure(2, weight=0)
        content.grid_rowconfigure(0, weight=1)

        ctk.CTkLabel(
            content,
            text="CTP\nImages",
            font=FONTS["h2"],
            text_color=THEME["text"],
            justify="left",
            anchor="nw",
        ).grid(row=0, column=0, sticky="nw", padx=(0, 10), pady=(2, 0))

        # image canvas
        self.img_canvas = tk.Canvas(content, bg=THEME["panel_3"], highlightthickness=0)
        self.img_canvas.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        self.img_canvas.bind("<ButtonPress-1>", self._on_ctp_canvas_press)
        self.img_canvas.bind("<B1-Motion>", self._on_ctp_canvas_drag)
        self.img_canvas.bind("<ButtonRelease-1>", self._on_ctp_canvas_release)
        self.img_canvas.bind("<Configure>", lambda _e: self._render_ctp_image())
        self.img_canvas.bind("<MouseWheel>", self._on_ctp_zoom_event)
        self.img_canvas.bind("<Button-4>", self._on_ctp_zoom_event)
        self.img_canvas.bind("<Button-5>", self._on_ctp_zoom_event)
        self.img_canvas.bind("<Double-Button-1>", self._reset_ctp_zoom)

        self.lbl_ctp_source = ctk.CTkLabel(
            content, text="", font=FONTS["body"], text_color=THEME["muted"]
        )
        self.lbl_ctp_source.place(in_=self.img_canvas, relx=0.5, rely=0.5, anchor="center")

        # slice (vertical) slider
        slice_col = ctk.CTkFrame(content, fg_color="transparent")
        slice_col.grid(row=0, column=2, sticky="ns")

        ctk.CTkLabel(slice_col, text="Slice", text_color=THEME["muted"], font=FONTS["small"]).pack(pady=(6, 6))

        self.lbl_ctp_slice_val = ctk.CTkLabel(
            slice_col,
            text=str(int(self.state.ctp_slice)),
            font=FONTS["small"],
            width=56,
            fg_color=THEME["panel_3"],
            corner_radius=10,
        )
        self.lbl_ctp_slice_val.pack(pady=(0, 8))

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
            height=300,
            command=self._on_ctp_slice_change,
        )
        self.slider_ctp_slice.pack(padx=6, pady=(0, 6), fill="y", expand=True)

        # time (horizontal) slider
        time_row = ctk.CTkFrame(self.ctp_panel, fg_color="transparent")
        time_row.grid(row=1, column=0, sticky="ew", padx=12, pady=(14, 8))
        time_row.grid_columnconfigure(1, weight=1)
        time_row.grid_columnconfigure(2, minsize=72)

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

        self.lbl_ctp_time_val = ctk.CTkLabel(
            time_row,
            text=str(self.var_ctp_time.get()),
            font=FONTS["small"],
            width=60,
            fg_color=THEME["panel_3"],
            corner_radius=10,
        )
        self.lbl_ctp_time_val.grid(row=0, column=2, sticky="e", padx=(10, 0))

        content.bind("<Configure>", lambda _event: self._update_ctp_image_size())
        self.after(40, self._update_ctp_image_size)

    def _build_controls_panel(self) -> None:
        self.controls = ctk.CTkFrame(self.left, fg_color=THEME["panel_2"], corner_radius=16)
        self.controls.grid(row=1, column=0, sticky="nsew", padx=10, pady=(2, 10))
        self.controls.grid_columnconfigure(0, weight=1)

        # row 0 — Sample ROI / Search ROI
        row1 = ctk.CTkFrame(self.controls, fg_color="transparent")
        row1.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
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
            height=32,
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
            height=32,
        ).grid(row=0, column=3, sticky="ew")

        # rows 1-2 — L / W
        self.var_len = ctk.DoubleVar(value=float(self.state.ctp_length))
        self.var_wid = ctk.DoubleVar(value=float(self.state.ctp_width))
        self._build_slider_line(self.controls, "L", self.var_len, row=1)
        self._build_slider_line(self.controls, "W", self.var_wid, row=2)

        # row 3 — Set pre/post lesion
        row3 = ctk.CTkFrame(self.controls, fg_color="transparent")
        row3.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 6))
        row3.grid_columnconfigure(0, weight=1)
        row3.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            row3, text="Set Pre Lesion",
            fg_color=THEME["panel_3"], hover_color=THEME["border_2"],
            height=34, corner_radius=12,
            command=lambda: self._on_set_lesion("pre"),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            row3, text="Set Post Lesion",
            fg_color=THEME["panel_3"], hover_color=THEME["border_2"],
            height=34, corner_radius=12,
            command=lambda: self._on_set_lesion("post"),
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        # row 4 — Play / speed / height-positive
        row4 = ctk.CTkFrame(self.controls, fg_color="transparent")
        row4.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 10))
        row4.grid_columnconfigure(1, weight=1)

        self.btn_play = ctk.CTkButton(
            row4, text="Play Movie",
            fg_color=THEME["accent"], hover_color=THEME["accent_2"],
            text_color="black",
            height=34, corner_radius=12,
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
            height=32, width=108,
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
        line.grid(row=row, column=0, sticky="ew", padx=12, pady=4)
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

    def _update_ctp_image_size(self) -> None:
        parent = self.img_canvas.master
        if parent is None:
            return

        parent.update_idletasks()
        total_w = max(1, parent.winfo_width())
        total_h = max(1, parent.winfo_height())
        title_w = 58
        slider_w = 78
        target = min(max(260, total_w - title_w - slider_w - 18), max(260, total_h + 16))
        self.img_canvas.configure(width=target, height=target)

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
            row=row, column=0, sticky="nsew", padx=12,
            pady=(12, 6) if row == 0 else (6, 12),
        )
        block.grid_columnconfigure(0, weight=1)
        block.grid_rowconfigure(1, weight=1)

        # header
        header = ctk.CTkFrame(block, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text=title, font=FONTS["h1"]).grid(row=0, column=0, sticky="w")

        btns = ctk.CTkFrame(header, fg_color="transparent")
        btns.grid(row=0, column=1, sticky="e")
        ctk.CTkButton(
            btns, text="Undo",
            fg_color=THEME["panel_3"], hover_color=THEME["border_2"],
            height=30, corner_radius=10,
            command=lambda b=block: self._curve_undo(b),
        ).pack(side="left", padx=(0, 10))
        ctk.CTkButton(
            btns, text="Clear",
            fg_color=THEME["panel_3"], hover_color=THEME["border_2"],
            height=30, corner_radius=10,
            command=lambda b=block: self._curve_clear(b),
        ).pack(side="left")

        # matplotlib figure
        fig = Figure(figsize=(6, 3.2), dpi=100)
        ax  = fig.add_subplot(111)
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
        block.lesion       = lesion
        block.fig          = fig
        block.ax           = ax
        block.times        = []
        block.values       = []
        block.selected_idx = None
        block.start_line   = ax.axvline(0,  color=THEME["accent"], linewidth=2, visible=False)
        block.end_line     = ax.axvline(10, color=THEME["accent"], linewidth=2, visible=False)

        canvas = FigureCanvasTkAgg(fig, master=block)
        w = canvas.get_tk_widget()
        w.configure(bg="black", highlightthickness=0)
        w.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        block.canvas = canvas

        canvas.mpl_connect("button_press_event",
                           lambda evt, b=block: self._on_plot_click(evt, b))

        # ── Start / End range sliders ──────────────────────────────────────
        controls = ctk.CTkFrame(block, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 4))
        controls.grid_columnconfigure(1, weight=1)
        controls.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(controls, text="Start", text_color=THEME["muted"], font=FONTS["small"]).grid(
            row=0, column=0, sticky="w")
        block.var_start = ctk.DoubleVar(value=0.0)
        block.s_start   = ctk.CTkSlider(
            controls, from_=0, to=10, number_of_steps=10,
            variable=block.var_start,
            fg_color=THEME["border"], progress_color=THEME["accent"],
            button_color=THEME["accent"], button_hover_color=THEME["accent_2"],
        )
        block.s_start.grid(row=0, column=1, sticky="ew", padx=(8, 16))

        ctk.CTkLabel(controls, text="End", text_color=THEME["muted"], font=FONTS["small"]).grid(
            row=0, column=2, sticky="w")
        block.var_end = ctk.DoubleVar(value=10.0)
        block.s_end   = ctk.CTkSlider(
            controls, from_=0, to=10, number_of_steps=10,
            variable=block.var_end,
            fg_color=THEME["border"], progress_color=THEME["accent"],
            button_color=THEME["accent"], button_hover_color=THEME["accent_2"],
        )
        block.s_end.grid(row=0, column=3, sticky="ew", padx=(8, 0))

        # ── Baseline / Washout entries + Fit Curve ────────────────────────
        fit_row = ctk.CTkFrame(block, fg_color="transparent")
        fit_row.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 4))

        ctk.CTkLabel(fit_row, text="Baseline", text_color=THEME["muted"],
                     font=FONTS["small"]).pack(side="left")
        block.var_baseline = ctk.StringVar(value="0")
        baseline_entry = ctk.CTkEntry(
            fit_row, textvariable=block.var_baseline,
            width=60, height=28,
            fg_color=THEME["input_bg"], border_color=THEME["border"],
        )
        baseline_entry.pack(side="left", padx=(6, 16))

        ctk.CTkLabel(fit_row, text="Washout", text_color=THEME["muted"],
                     font=FONTS["small"]).pack(side="left")
        block.var_washout = ctk.StringVar(value="0")
        washout_entry = ctk.CTkEntry(
            fit_row, textvariable=block.var_washout,
            width=60, height=28,
            fg_color=THEME["input_bg"], border_color=THEME["border"],
        )
        washout_entry.pack(side="left", padx=(6, 16))

        ctk.CTkButton(
            fit_row, text="Fit Curve",
            fg_color=THEME["accent"], hover_color=THEME["accent_2"],
            text_color="black", height=30, corner_radius=10,
            command=lambda b=block: self._on_fit_curve(b),
        ).pack(side="left")

        # ── Edit Point / Remove Point ──────────────────────────────────────
        point_row = ctk.CTkFrame(block, fg_color="transparent")
        point_row.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 10))

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

        # ── Wire slider ↔ entry ↔ fit ──────────────────────────────────────
        self._wire_range_controls(block, baseline_entry, washout_entry)

        if is_pre:
            self.pre_lesion_block  = block
        else:
            self.post_lesion_block = block

    # -------------------------------------------------------------------------
    def _wire_range_controls(self, block, baseline_entry, washout_entry) -> None:
        """
        Bidirectional sync between Start/End sliders and Baseline/Washout entries.
        Changing either one updates the other AND re-runs the fit automatically.
        """

        # ── Slider → Entry ────────────────────────────────────────────────
        def on_start_slider(val, b=block):
            t = float(val)
            try:
                end = float(b.var_end.get())
            except ValueError:
                end = t
            if t > end:
                t = end
                b.var_start.set(t)
            b.var_baseline.set(f"{t:.2f}")
            _update_range_line(b, start=True, t_val=t)

        def on_end_slider(val, b=block):
            t = float(val)
            try:
                start = float(b.var_start.get())
            except ValueError:
                start = t
            if t < start:
                t = start
                b.var_end.set(t)
            b.var_washout.set(f"{t:.2f}")
            _update_range_line(b, start=False, t_val=t)

        def _update_range_line(b, start: bool, t_val: float):
            line = b.start_line if start else b.end_line
            line.set_xdata([t_val, t_val])
            line.set_visible(True)
            b.canvas.draw_idle()

        block.s_start.configure(command=on_start_slider)
        block.s_end.configure(command=on_end_slider)

        # ── Entry → Slider → fit (on Enter or FocusOut) ───────────────────
        def apply_baseline(event=None, b=block):
            if not b.times:
                return
            try:
                val = float(b.var_baseline.get())
            except ValueError:
                return
            times = np.asarray(b.times, dtype=float)
            val = float(np.clip(val, times[0], times[-1]))
            try:
                end = float(b.var_end.get())
            except ValueError:
                end = times[-1]
            val = min(val, end)
            b.var_start.set(val)
            b.var_baseline.set(f"{val:.2f}")
            _update_range_line(b, start=True, t_val=val)
            self._on_fit_curve(b)

        def apply_washout(event=None, b=block):
            if not b.times:
                return
            try:
                val = float(b.var_washout.get())
            except ValueError:
                return
            times = np.asarray(b.times, dtype=float)
            val = float(np.clip(val, times[0], times[-1]))
            try:
                start = float(b.var_start.get())
            except ValueError:
                start = times[0]
            val = max(val, start)
            b.var_end.set(val)
            b.var_washout.set(f"{val:.2f}")
            _update_range_line(b, start=False, t_val=val)
            self._on_fit_curve(b)

        baseline_entry.bind("<Return>",   apply_baseline)
        baseline_entry.bind("<FocusOut>", apply_baseline)
        washout_entry.bind("<Return>",    apply_washout)
        washout_entry.bind("<FocusOut>",  apply_washout)

    # =========================================================================
    # Rendering helpers
    # =========================================================================

    def _on_ctp_zoom_event(self, event):
        vol = getattr(self.state, "ctp_volume", None)
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

        self._ctp_zoom = min(6.0, max(1.0, self._ctp_zoom * factor))
        if self._overlay_image is not None:
            self._render_ctp_image_with(self._overlay_image)
        else:
            self._render_ctp_image()
        return "break"

    def _reset_ctp_zoom(self, _event=None):
        self._ctp_zoom = 1.0
        self._ctp_pan_x = 0.0
        self._ctp_pan_y = 0.0
        if self._overlay_image is not None:
            self._render_ctp_image_with(self._overlay_image)
        else:
            self._render_ctp_image()
        return "break"

    def _on_ctp_canvas_press(self, event) -> None:
        self._ctp_drag_origin = (event.x, event.y)
        self._ctp_drag_pan_start = (self._ctp_pan_x, self._ctp_pan_y)
        self._ctp_was_dragged = False

    def _on_ctp_canvas_drag(self, event) -> None:
        vol = getattr(self.state, "ctp_volume", None)
        if not vol or self._ctp_zoom <= 1.0 or self._ctp_drag_origin is None or self._ctp_drag_pan_start is None:
            return

        start_x, start_y = self._ctp_drag_origin
        pan_start_x, pan_start_y = self._ctp_drag_pan_start
        dx = event.x - start_x
        dy = event.y - start_y
        if abs(dx) > 2 or abs(dy) > 2:
            self._ctp_was_dragged = True
        self._ctp_pan_x = pan_start_x + dx
        self._ctp_pan_y = pan_start_y + dy
        if self._overlay_image is not None:
            self._render_ctp_image_with(self._overlay_image)
        else:
            self._render_ctp_image()

    def _on_ctp_canvas_release(self, event) -> None:
        dragged = self._ctp_was_dragged
        self._ctp_drag_origin = None
        self._ctp_drag_pan_start = None
        self._ctp_was_dragged = False
        if not dragged:
            self._on_image_click(event)

    def _draw_ctp_canvas_image(self, img8: np.ndarray) -> None:
        cw = max(10, self.img_canvas.winfo_width())
        ch = max(10, self.img_canvas.winfo_height())
        pil = Image.fromarray(img8)
        img_w, img_h = pil.size

        scale = min(cw / max(1, img_w), ch / max(1, img_h))
        scale *= self._ctp_zoom
        disp_w = max(1, int(img_w * scale))
        disp_h = max(1, int(img_h * scale))
        max_pan_x = max(0.0, (disp_w - cw) / 2.0)
        max_pan_y = max(0.0, (disp_h - ch) / 2.0)
        self._ctp_pan_x = float(np.clip(self._ctp_pan_x, -max_pan_x, max_pan_x))
        self._ctp_pan_y = float(np.clip(self._ctp_pan_y, -max_pan_y, max_pan_y))
        center_x = (cw / 2.0) + self._ctp_pan_x
        center_y = (ch / 2.0) + self._ctp_pan_y
        x0 = center_x - (disp_w / 2.0)
        y0 = center_y - (disp_h / 2.0)
        self._ctp_display_rect = (x0, y0, disp_w, disp_h, img_h, img_w)

        photo = ImageTk.PhotoImage(pil.resize((disp_w, disp_h), Image.Resampling.LANCZOS))
        self._ctp_photo = photo

        self.img_canvas.delete("all")
        self.img_canvas.create_image(center_x, center_y, image=photo, anchor="center")
        self.lbl_ctp_source.configure(text="")

    def _render_ctp_image(self) -> None:
        vol = getattr(self.state, "ctp_volume", None)
        if not vol:
            self._ctp_display_rect = None
            self.lbl_ctp_source.configure(text="No CTP loaded")
            return

        pixels: np.ndarray = vol["pixels"]   # (T, Z, H, W)
        T, Z, H, W = pixels.shape

        self.var_ctp_slice.set(min(max(1, self.var_ctp_slice.get()), Z))
        self.var_ctp_time.set(min(max(1, self.var_ctp_time.get()), T))
        self.lbl_ctp_slice_val.configure(text=str(self.var_ctp_slice.get()))
        self.lbl_ctp_time_val.configure(text=str(self.var_ctp_time.get()))

        if self.slider_ctp_slice.cget("to") != Z or self.slider_ctp_time.cget("to") != T:
            self.slider_ctp_slice.configure(from_=1, to=max(1, Z), number_of_steps=max(1, Z - 1))
            self.slider_ctp_time.configure(from_=1, to=max(1, T), number_of_steps=max(1, T - 1))

        t_idx = min(max(0, int(self.var_ctp_time.get())  - 1), T - 1)
        z_idx = min(max(0, int(self.var_ctp_slice.get()) - 1), Z - 1)

        img8 = window_to_uint8(
            pixels[t_idx, z_idx],
            length=float(getattr(self.state, "ctp_length", 0.5)),
            width =float(getattr(self.state, "ctp_width",  0.5)),
        )
        self._overlay_image = None
        self._draw_ctp_canvas_image(img8)

        # redraw the search-ROI rectangle if one exists
        if self._last_search_roi_img is not None:
            self._draw_search_roi_rect(*self._last_search_roi_img, img_h=H, img_w=W)

    def _image_to_canvas(self, r_img: int, c_img: int, img_h: int, img_w: int) -> tuple[float, float]:
        """Convert image pixel (row, col) → canvas pixel (cx, cy)."""
        rect = self._ctp_display_rect
        if rect is None:
            return 0.0, 0.0
        x0, y0, disp_w, disp_h, _, _ = rect
        cx = x0 + (c_img * disp_w / max(1, img_w))
        cy = y0 + (r_img * disp_h / max(1, img_h))
        return cx, cy

    def _draw_search_roi_rect(self, r0: int, c0: int, r1: int, c1: int,
                               img_h: int, img_w: int) -> None:
        """Draw the search-window rectangle on the canvas in scaled coordinates."""
        self.img_canvas.delete("search_roi")
        x0, y0 = self._image_to_canvas(r0, c0, img_h, img_w)
        x1, y1 = self._image_to_canvas(r1, c1, img_h, img_w)
        self.img_canvas.create_rectangle(
            x0, y0, x1, y1,
            outline="white", width=1, tags="search_roi",
        )

    def _render_ctp_image_with(self, image_override: np.ndarray) -> None:
        img8 = window_to_uint8(
            image_override,
            length=float(getattr(self.state, "ctp_length", 0.5)),
            width =float(getattr(self.state, "ctp_width",  0.5)),
        )
        self._overlay_image = image_override
        self._draw_ctp_canvas_image(img8)

    # ── curve redraw (FIXED — keeps tight limits, never resets to -50..500) ──

    def _redraw_curve(self, block) -> None:
        """
        Redraw only the sampled dots, preserving tight axis limits.
        Used by Undo and Clear — does NOT call plot_sampled_curve (which would
        reset axes to the MATLAB -50…500 formula).
        """
        ax = block.ax
        ax.cla()

        ax.set_facecolor("black")
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel("Enhancement (HU)", color="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.tick_params(colors="white")

        if block.times:
            times_arr  = np.asarray(block.times, dtype=float)
            values_arr = np.asarray(block.values, dtype=float)
            colour     = _COLOUR.get(block.lesion, "white")

            ax.plot(times_arr, values_arr, "o",
                    color=colour, markersize=5, alpha=0.85,
                    label=f"{'Pre' if block.lesion == 'pre' else 'Post'}-Lesion Sampled")

            # tight limits
            t_s, t_e = float(times_arr[0]), float(times_arr[-1])
            v_lo, v_hi = float(np.min(values_arr)), float(np.max(values_arr))
            pad = max(1.0, (v_hi - v_lo) * 0.10)
            ax.set_xlim(t_s, t_e)
            ax.set_ylim(v_lo - pad, v_hi + pad)

            x_ticks = np.unique(np.linspace(t_s, t_e, 11).astype(int))
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(int(t)) for t in x_ticks])

            # update slider range
            n = max(1, len(times_arr) - 1)
            block.s_start.configure(from_=t_s, to=t_e, number_of_steps=n)
            block.s_end.configure(from_=t_s,   to=t_e, number_of_steps=n)
            block.var_start.set(t_s)
            block.var_end.set(t_e)

            ax.legend(facecolor="#1e1e1e", labelcolor="white", fontsize=8)

        block.start_line = ax.axvline(
            block.var_start.get() if block.times else 0,
            color=THEME["accent"], linewidth=2, visible=False,
        )
        block.end_line = ax.axvline(
            block.var_end.get() if block.times else 10,
            color=THEME["accent"], linewidth=2, visible=False,
        )
        block.canvas.draw_idle()

    # ── canvas overlays ────────────────────────────────────────────────────────

    def _draw_crosshair(self, x: int, y: int) -> None:
        self.img_canvas.delete("crosshair")
        size = 6
        self.img_canvas.create_line(x - size, y, x + size, y, fill="red", width=2, tags="crosshair")
        self.img_canvas.create_line(x, y - size, x, y + size, fill="red", width=2, tags="crosshair")

    def _add_draggable_point(self, cx: int, cy: int, lesion: LesionType) -> None:
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

        self.img_canvas.tag_bind("drag_point", "<B1-Motion>",       on_drag)
        self.img_canvas.tag_bind("drag_point", "<ButtonRelease-1>", on_release)

    # ── curve range / axes sync ────────────────────────────────────────────────

    def _sync_curve_block_from_data(self, block, times: list, values: list) -> None:
        if not times:
            return
        times_arr  = np.asarray(times, dtype=float)
        values_arr = np.asarray(values, dtype=float)

        t_start = float(times_arr[0])
        t_end   = float(times_arr[-1])
        v_min   = float(np.min(values_arr))
        v_max   = float(np.max(values_arr))
        y_pad   = max(1.0, (v_max - v_min) * 0.08)

        block.ax.set_xlim(t_start, t_end)
        block.ax.set_ylim(v_min - y_pad, v_max + y_pad)

        n_steps = max(1, len(times_arr) - 1)
        block.var_start.set(t_start)
        block.var_end.set(t_end)
        block.s_start.configure(from_=t_start, to=t_end, number_of_steps=n_steps)
        block.s_end.configure(from_=t_start,   to=t_end, number_of_steps=n_steps)

        block.start_line.set_xdata([t_start, t_start])
        block.end_line.set_xdata([t_end, t_end])

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

    def _on_ctp_time_change(self, _=None) -> None:
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
        vol = getattr(self.state, "ctp_volume", None)
        if not vol:
            self._movie_after_id = None
            self.btn_play.configure(text="Play Movie")
            return

        max_t = vol["pixels"].shape[0]
        try:
            speed = float(self.var_speed.get().replace("x", ""))
        except ValueError:
            speed = 0.5

        delay = int(max(40, 250 / max(speed, 0.1)))
        next_val = (int(self.var_ctp_time.get()) % max_t) + 1
        self.var_ctp_time.set(next_val)
        self.state.ctp_time = next_val
        self.lbl_ctp_time_val.configure(text=str(next_val))
        self._render_ctp_image()
        self._movie_after_id = self.after(delay, self._movie_loop)

    # ── Undo / Clear ──────────────────────────────────────────────────────────

    def _curve_undo(self, block) -> None:
        """Remove the last sampled point and redraw with tight limits."""
        if not block.times:
            return
        block.times.pop()
        block.values.pop()
        block.selected_idx = None
        self._redraw_curve(block)

    def _curve_clear(self, block) -> None:
        """Clear all sampled points."""
        block.times        = []
        block.values       = []
        block.selected_idx = None
        self._redraw_curve(block)

    # =========================================================================
    # Set Lesion
    # =========================================================================

    def _on_set_lesion(self, lesion: LesionType) -> None:
        if self.current_x is None or self.current_y is None:
            print(f"[{lesion}] Click on the image first")
            return

        vol = getattr(self.state, "ctp_volume", None)
        if not vol:
            print("No CTP volume loaded")
            return

        pixels: np.ndarray = vol["pixels"]   # (T, Z, H, W)
        T, Z, H, W = pixels.shape

        raw_times = np.asarray(vol.get("times", np.arange(T, dtype=float)))

        def dicom_time_to_seconds(t_val):
            if t_val > 10000:
                hh = int(t_val // 10000)
                mm = int((t_val % 10000) // 100)
                ss = t_val % 100
                return hh * 3600 + mm * 60 + ss
            return float(t_val)

        time_points_s = np.array([dicom_time_to_seconds(t) for t in raw_times])
        time_points   = time_points_s - time_points_s[0]
        pixel_spacing = vol.get("pixel_spacing", (0.5, 0.5))

        t_idx = min(max(0, int(self.var_ctp_time.get())  - 1), T - 1)
        z_idx = min(max(0, int(self.var_ctp_slice.get()) - 1), Z - 1)

        # Canvas coords → image coords
        rect = self._ctp_display_rect
        if rect is None:
            return
        x0, y0, disp_w, disp_h, _, _ = rect
        click_col = int(np.clip((self.current_x - x0) * W / max(1, disp_w), 0, W - 1))
        click_row = int(np.clip((self.current_y - y0) * H / max(1, disp_h), 0, H - 1))

        sample_n  = int(self.var_sample_roi.get().split("x")[0].strip())
        search_n  = int(self.var_search_roi.get().split("x")[0].strip())
        # search window size in pixels: 1x1→20, 2x2→40, 3x3→60, 4x4→80
        search_px = search_n * 20

        # ── draw search-ROI rectangle on canvas ───────────────────────────
        half = search_px // 2
        r0 = max(0, click_row - half);  c0 = max(0, click_col - half)
        r1 = min(H - 1, click_row + half); c1 = min(W - 1, click_col + half)
        self._last_search_roi_img = (r0, c0, r1, c1)
        self._draw_search_roi_rect(r0, c0, r1, c1, img_h=H, img_w=W)

        # 1. get_best_sample
        sample = get_best_sample(
            x=click_row,
            y=click_col,
            window_size=search_px,
            roi_m=sample_n,
            roi_n=sample_n,
            slice_idx=z_idx,
            four_d_image_set=pixels,
            time_point_values=time_points,
        )
        interp_times = sample["interpolated_time_points"]
        interp_vals  = sample["interpolated_sampled_points"]
        sx_rows      = sample["search_window_rows"]
        sx_cols      = sample["search_window_cols"]

        if len(interp_times) == 0 or len(interp_vals) == 0:
            print("ERROR: sampled points are empty!")
            return

        print(f"Sampled curve [{lesion}]: {len(interp_times)} pts  "
              f"t=[{interp_times[0]:.1f}…{interp_times[-1]:.1f}]  "
              f"val=[{interp_vals[0]:.1f}…{interp_vals[-1]:.1f}]")

        # 2. get_contour
        contour = get_contour(
            x=click_row,
            y=click_col,
            search_window_size=_SEGMENTATION_WINDOW,
            slice_idx=z_idx,
            time_point_idx=t_idx,
            four_d_image_set=pixels,
            pixel_spacing=pixel_spacing,
        )
        print(f"[{lesion}] radius={contour.radius_cm:.3f} cm  area={contour.area_cm2:.4f} cm²")

        # 3. get_roi
        roi_obj = get_roi(
            study_name=getattr(self.state, "study_name", "Unknown"),
            num_time_points=T,
            num_slices=Z,
            x=click_row, y=click_col, z=z_idx, t=t_idx,
            sampled_curve=interp_vals.tolist(),
            time_points=time_points.tolist(),
            fitted_curve=[],
            fitted_time_points=[],
            roi_x_boundary=sx_rows.tolist(),
            roi_y_boundary=sx_cols.tolist(),
        )
        if lesion == "pre":
            self.pre_roi  = roi_obj
        else:
            self.post_roi = roi_obj

        # 4. save JSON
        bundle = {
            "preRoiObject":  self._roi_to_dict(self.pre_roi),
            "postRoiObject": self._roi_to_dict(self.post_roi),
        }
        path = save_roi_as_json(bundle)
        print(f"ROI saved → {path}")

        # 5. overlay + redraw canvas
        image_2d = pixels[t_idx, z_idx].copy().astype(np.float32)
        overlaid = get_roi_overlayed(image_2d, sx_rows, sx_cols, overlay_value=1500.0)
        self._render_ctp_image_with(overlaid)
        # redraw search rect on top of new image
        self._draw_search_roi_rect(r0, c0, r1, c1, img_h=H, img_w=W)
        self._draw_crosshair(self.current_x, self.current_y)
        self._add_draggable_point(self.current_x, self.current_y, lesion)

        # 6. plot sampled curve (bypass _configure_axes by plotting manually)
        block = self.pre_lesion_block if lesion == "pre" else self.post_lesion_block
        block.times        = interp_times.tolist()
        block.values       = interp_vals.tolist()
        block.selected_idx = None

        block.ax.cla()
        block.ax.set_facecolor("black")
        block.ax.set_xlabel("Time (s)", color="white")
        block.ax.set_ylabel("Enhancement (HU)", color="white")
        for spine in block.ax.spines.values():
            spine.set_color("white")
        block.ax.tick_params(colors="white")

        colour = _COLOUR.get(lesion, "white")
        times_arr  = np.asarray(block.times, dtype=float)
        values_arr = np.asarray(block.values, dtype=float)
        block.ax.plot(times_arr, values_arr, "o", color=colour,
                      markersize=5, alpha=0.85,
                      label=f"{'Pre' if lesion == 'pre' else 'Post'}-Lesion Sampled")

        self._sync_curve_block_from_data(block, block.times, block.values)
        block.start_line = block.ax.axvline(
            block.var_start.get(), color=THEME["accent"], linewidth=2, visible=False)
        block.end_line = block.ax.axvline(
            block.var_end.get(), color=THEME["accent"], linewidth=2, visible=False)
        block.canvas.draw()

        # 7. auto-fit
        self._on_fit_curve(block)

    # =========================================================================
    # Fitting handler
    # =========================================================================

    def _on_fit_curve(self, block) -> None:
        if not block.times or len(block.times) < 4:
            print("Not enough points to fit a curve.")
            return

        try:
            times  = np.asarray(block.times,  dtype=float)
            values = np.asarray(block.values, dtype=float)

            # 1. read start / end from sliders (these are always in sync with entries)
            t_start = float(block.var_start.get())
            t_end   = float(block.var_end.get())

            if t_start >= t_end:
                t_start = float(times[0])
                t_end   = float(times[-1])
                block.var_start.set(t_start)
                block.var_end.set(t_end)

            # 2. trim to [start, end]
            mask = (times >= t_start) & (times <= t_end)
            if mask.sum() < 4:
                mask = np.ones(len(times), dtype=bool)

            times_fit  = times[mask]
            values_fit = values[mask]

            # 3. 1-based indices
            baseline_idx = int(np.argmin(np.abs(times - t_start))) + 1
            washout_idx  = int(np.argmin(np.abs(times - t_end)))   + 1
            baseline_idx = max(1, min(baseline_idx, len(times)))
            washout_idx  = max(baseline_idx, min(washout_idx, len(times)))

            # Update entry displays
            block.var_baseline.set(str(baseline_idx))
            block.var_washout.set(str(washout_idx))

            print(f"[FitCurve | {block.lesion}] "
                  f"t={t_start:.2f}–{t_end:.2f}s  "
                  f"idx={baseline_idx}–{washout_idx}  pts={mask.sum()}")

            # 4. run fit (primary fixed, safe fallback)
            try:
                result = _get_fitted_curve_fixed(
                    sample_curve=values,
                    sample_time=times,
                    baseline=baseline_idx,
                    washout=washout_idx,
                )
            except Exception as primary_exc:
                print(f"  Primary fit failed ({primary_exc}); safe fallback …")
                result = get_fitted_curve_safe(
                    sample_curve=values,
                    sample_time=times,
                    baseline=baseline_idx,
                    washout=washout_idx,
                )

            print(f"  converged={result.converged}, RMSE={result.rmse:.3f}, "
                  f"K={result.k:.3f}, α={result.alpha:.3f}, β={result.beta:.3f}, "
                  f"AUC={result.auc:.1f}")

            # 5. trim fitted curve to display window
            ft, fc = result.fitted_time, result.fitted_curve
            fit_mask = (ft >= t_start) & (ft <= t_end)
            if fit_mask.sum() < 2:
                fit_mask = np.ones(len(ft), dtype=bool)
            ft_d = ft[fit_mask]
            fc_d = fc[fit_mask]

            # 6. tight axis limits
            all_y  = np.concatenate([values_fit, fc_d])
            y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
            y_pad  = max(1.0, (y_max - y_min) * 0.08)
            y_lo   = y_min - y_pad
            y_hi   = y_max + y_pad

            raw_interval  = (y_hi - y_lo) / 5.0
            magnitude     = 10 ** np.floor(np.log10(max(raw_interval, 1e-9)))
            nice_interval = max(1, int(round(raw_interval / magnitude) * magnitude))
            y_ticks = np.arange(
                int(np.floor(y_lo / nice_interval)) * nice_interval,
                int(np.ceil(y_hi  / nice_interval)) * nice_interval + nice_interval,
                nice_interval,
            )
            y_lo = float(y_ticks[0])  - y_pad * 0.5
            y_hi = float(y_ticks[-1]) + y_pad * 0.5

            # 7. redraw
            block.ax.cla()
            block.ax.set_facecolor("black")
            block.ax.set_xlabel("Time (s)", color="white")
            block.ax.set_ylabel("Enhancement (HU)", color="white")
            for spine in block.ax.spines.values():
                spine.set_color("white")
            block.ax.tick_params(colors="white")

            colour = _COLOUR.get(block.lesion, "white")
            block.ax.plot(times_fit, values_fit, "o", color=colour,
                          markersize=5, alpha=0.85,
                          label=f"{'Pre' if block.lesion == 'pre' else 'Post'}-Lesion Sampled")
            block.ax.plot(ft_d, fc_d, "-", color=colour, linewidth=2,
                          label=f"{'Pre' if block.lesion == 'pre' else 'Post'}-Lesion Fitted")
            block.ax.legend(facecolor="#1e1e1e", labelcolor="white", fontsize=8)

            # 8. lock axes
            block.ax.autoscale(False)
            block.ax.set_xlim(t_start, t_end)
            block.ax.set_ylim(y_lo, y_hi)
            block.ax.set_yticks(y_ticks)
            x_ticks = np.unique(np.linspace(t_start, t_end, 11).astype(int))
            block.ax.set_xticks(x_ticks)
            block.ax.set_xticklabels([str(int(t)) for t in x_ticks])

            # 9. reconfigure sliders
            n_steps = max(1, mask.sum() - 1)
            block.s_start.configure(from_=t_start, to=t_end, number_of_steps=n_steps)
            block.s_end.configure(from_=t_start,   to=t_end, number_of_steps=n_steps)
            block.var_start.set(t_start)
            block.var_end.set(t_end)

            block.start_line = block.ax.axvline(t_start, color=THEME["accent"], linewidth=2, visible=False)
            block.end_line   = block.ax.axvline(t_end,   color=THEME["accent"], linewidth=2, visible=False)

            block.canvas.draw()
            print(f"  Plotted [{block.lesion}]: {mask.sum()} dots, "
                  f"{fit_mask.sum()} fitted pts, y=[{y_lo:.1f}, {y_hi:.1f}]")

        except Exception as exc:
            import traceback
            print(f"Fitting error: {exc}")
            traceback.print_exc()

    # =========================================================================
    # Plot click → select point / Edit / Remove
    # =========================================================================

    def _on_plot_click(self, event, block) -> None:
        if event.xdata is None or event.ydata is None or not block.times:
            return

        times  = np.asarray(block.times)
        values = np.asarray(block.values)

        x_range = max(1e-9, block.ax.get_xlim()[1] - block.ax.get_xlim()[0])
        y_range = max(1e-9, block.ax.get_ylim()[1] - block.ax.get_ylim()[0])

        dx    = (times - event.xdata) / x_range
        dy    = (values - event.ydata) / y_range
        dists = np.sqrt(dx**2 + dy**2)
        idx   = int(np.argmin(dists))

        block.selected_idx = idx
        self._highlight_selected_point(block, idx)
        block.lbl_selected.configure(
            text=f"Point {idx + 1}: t={times[idx]:.2f}s  {values[idx]:.1f} HU"
        )

    def _highlight_selected_point(self, block, idx: int) -> None:
        for line in list(block.ax.lines):
            if getattr(line, "_highlight_marker", False):
                line.remove()

        (hl,) = block.ax.plot(
            block.times[idx], block.values[idx], "o",
            markersize=12, markerfacecolor="none",
            markeredgecolor="yellow", markeredgewidth=2, alpha=0.8,
        )
        hl._highlight_marker = True
        block.canvas.draw_idle()

    def _on_edit_point(self, block) -> None:
        if not block.times:
            return
        if block.selected_idx is None:
            print(f"[{block.lesion}] Click a point on the graph first")
            return

        from tkinter import simpledialog
        idx         = block.selected_idx
        current_time = block.times[idx]
        current_hu   = block.values[idx]

        new_hu = simpledialog.askfloat(
            "Edit Point",
            f"t={current_time:.2f}s  Current: {current_hu:.1f} HU\nNew HU:",
            initialvalue=current_hu, minvalue=-100, maxvalue=2000,
        )
        if new_hu is None:
            return

        block.values[idx] = float(new_hu)
        block.lbl_selected.configure(
            text=f"Point {idx + 1}: t={current_time:.2f}s  {new_hu:.1f} HU (edited)"
        )
        self._clear_fitted_curve(block)
        self._redraw_curve(block)
        self._update_roi_curve_data(block)

    def _on_remove_point(self, block) -> None:
        if block.selected_idx is None:
            print(f"[{block.lesion}] Click a point first")
            return
        if len(block.times) <= 3:
            print(f"[{block.lesion}] Need at least 3 points")
            return

        idx = block.selected_idx
        block.times.pop(idx)
        block.values.pop(idx)
        block.selected_idx = None
        block.lbl_selected.configure(text="Point removed")

        self._clear_fitted_curve(block)
        self._redraw_curve(block)
        self._update_roi_curve_data(block)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _clear_fitted_curve(self, block) -> None:
        roi = self.pre_roi if block.lesion == "pre" else self.post_roi
        if roi:
            roi.fitted_curve      = []
            roi.fitted_time_points = []

    def _restore_range_lines(self, block) -> None:
        for attr in ("start_line", "end_line"):
            line = getattr(block, attr, None)
            if line is not None and line in block.ax.lines:
                try:
                    line.remove()
                except Exception:
                    pass
        block.start_line = block.ax.axvline(float(block.var_start.get()), color=THEME["accent"], linewidth=2)
        block.end_line   = block.ax.axvline(float(block.var_end.get()),   color=THEME["accent"], linewidth=2)

    def _update_roi_curve_data(self, block) -> None:
        roi = self.pre_roi if block.lesion == "pre" else self.post_roi
        if roi:
            roi.curve       = list(block.values)
            roi.time_points = list(block.times)

        bundle = {
            "preRoiObject":  self._roi_to_dict(self.pre_roi),
            "postRoiObject": self._roi_to_dict(self.post_roi),
        }
        path = save_roi_as_json(bundle)
        print(f"ROI updated → {path}")

    def _inject_test_volume(self) -> None:
        if getattr(self.state, "ctp_volume", None) is None:
            vol = make_test_volume()
            self.state.ctp_volume = {
                "pixels": vol["pixels"],
                "times":  vol["times"],
                "zs":     vol["zs"],
                "shape":  vol["shape"],
            }
            self.state.ctp_slice = 1
            self.state.ctp_time  = 1
            self._render_ctp_image()

    @staticmethod
    def _roi_to_dict(roi: Optional[ROIObject]) -> dict:
        if roi is None:
            return {}
        return {
            "x": roi.x, "y": roi.y, "z": roi.z, "t": roi.t,
            "curve":              roi.curve,
            "timePoints":         roi.time_points,
            "fittedCurve":        roi.fitted_curve,
            "fittedTimePoints":   roi.fitted_time_points,
            "roiXBoundary":       roi.roi_x_boundary,
            "roiYBoundary":       roi.roi_y_boundary,
        }
