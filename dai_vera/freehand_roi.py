"""
freehand_roi.py
───────────────
Standalone freehand polygon ROI handler.

Usage from CurvesROI page:
    1. Add to __init__:
           from freehand_roi import FreehandROI
           self._freehand = FreehandROI(self)

    2. Change button commands in _build_controls_panel:
           command=lambda: self._freehand.on_lesion_clicked("pre"),
           command=lambda: self._freehand.on_lesion_clicked("post"),

    That's it. Normal (non-freehand) clicks route through to
    self._on_set_lesion as before. Freehand clicks are collected
    here and the result is pushed back into the page's state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from roi_contour import get_contour, get_roi_overlayed, get_roi
from roi_sampling import get_best_sample

if TYPE_CHECKING:
    from typing import Literal
    LesionType = Literal["pre", "post"]


class FreehandROI:
    """
    Manages freehand polygon collection on the CurvesROI page canvas.

    Lifecycle:
        1. User selects "freehand" in Search ROI dropdown and clicks
           Set Pre/Post Lesion.
        2. Canvas switches to polygon-collection mode (crosshair cursor).
        3. Each left-click adds a vertex (green dot + connecting line).
        4. Double-click or right-click closes the polygon.
        5. The closed polygon is processed (contour → sample → ROI → overlay).
        6. Canvas returns to normal click mode.
    """

    def __init__(self, page) -> None:
        """
        Parameters
        ----------
        page : CurvesROI page instance — must expose:
            .img_canvas, .var_search_roi, .var_sample_roi,
            .var_ctp_time, .var_ctp_slice, .state,
            ._ctp_display_rect, ._on_image_click(),
            ._image_to_canvas(), ._render_ctp_image_with(),
            ._draw_curve_to_block(), ._roi_to_dict(),
            .pre_roi, .post_roi,
            .pre_lesion_block, .post_lesion_block
        """
        self.page = page

        # freehand state
        self._active = False
        self._points: list[tuple[int, int]] = []  # canvas coords
        self._lesion: Optional[str] = None

    # ──────────────────────────────────────────────────────────────────
    # Public entry point — replaces direct _on_set_lesion calls
    # ──────────────────────────────────────────────────────────────────

    def on_lesion_clicked(self, lesion: str) -> None:
        """
        Called by Set Pre/Post Lesion button.
        Routes to freehand mode or normal mode based on dropdown.
        """
        search_val = self.page.var_search_roi.get().strip().lower()

        if search_val == "freehand":
            self._start(lesion)
        else:
            self.page._on_set_lesion(lesion)

    # ──────────────────────────────────────────────────────────────────
    # Freehand polygon collection
    # ──────────────────────────────────────────────────────────────────

    def _start(self, lesion: str) -> None:
        """Enter polygon collection mode."""
        if self._active:
            self._cancel()

        self._active = True
        self._points = []
        self._lesion = lesion

        canvas = self.page.img_canvas
        canvas.config(cursor="crosshair")

        # Left-click  = add vertex
        # Right-click  = close polygon and process
        # Enter        = close polygon and process
        # Escape       = cancel
        # NOTE: we intentionally do NOT bind <Double-Button-1> because
        # tkinter fires <Button-1> before <Double-Button-1>, causing
        # a spurious vertex + premature close.
        canvas.bind("<Button-1>", self._on_click)
        canvas.bind("<Button-3>", lambda e: self._finish())
        toplevel = self.page.winfo_toplevel()
        toplevel.bind("<Return>", lambda e: self._finish())
        toplevel.bind("<Escape>", lambda e: self._cancel())

        print(
            f"[Freehand] Click points to define {lesion} ROI. "
            f"Right-click or press Enter to close. Esc to cancel."
        )

    def _on_click(self, event) -> None:
        """Add a vertex on left-click."""
        if not self._active:
            return

        cx, cy = event.x, event.y
        self._points.append((cx, cy))

        canvas = self.page.img_canvas
        r = 4
        canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline="#00FF00", fill="#00FF00", width=1,
            tags="freehand_poly",
        )

        if len(self._points) >= 2:
            px, py = self._points[-2]
            canvas.create_line(
                px, py, cx, cy,
                fill="#00FF00", width=2, tags="freehand_poly",
            )

        print(f"[Freehand] Point {len(self._points)}: canvas=({cx},{cy})")

    def _finish(self) -> None:
        """Close polygon and process."""
        if not self._active:
            return
        if len(self._points) < 3:
            print("[Freehand] Need at least 3 points.")
            return

        # draw closing line
        canvas = self.page.img_canvas
        fx, fy = self._points[0]
        lx, ly = self._points[-1]
        canvas.create_line(
            lx, ly, fx, fy,
            fill="#00FF00", width=2, tags="freehand_poly",
        )

        lesion = self._lesion
        points = list(self._points)
        self._restore()
        self._process(lesion, points)

    def _cancel(self) -> None:
        """Cancel without processing."""
        print("[Freehand] Cancelled.")
        self.page.img_canvas.delete("freehand_poly")
        self._restore()

    def _restore(self) -> None:
        """Exit collection mode, restore normal bindings."""
        self._active = False
        self._points = []
        self._lesion = None

        canvas = self.page.img_canvas
        canvas.config(cursor="")
        canvas.bind("<Button-1>", self.page._on_image_click)
        canvas.unbind("<Button-3>")
        toplevel = self.page.winfo_toplevel()
        toplevel.unbind("<Return>")
        toplevel.unbind("<Escape>")

    # ──────────────────────────────────────────────────────────────────
    # Processing — mirrors MATLAB freehand branch
    #   1. canvas coords → image coords
    #   2. get_contour  (polygon vertices)   ← contour FIRST
    #   3. get_best_sample (polygon vertices) ← sample SECOND
    #   4. build ROI, save, overlay, draw curve
    # ──────────────────────────────────────────────────────────────────

    def _process(self, lesion: str, canvas_points: list[tuple[int, int]]) -> None:
        page = self.page

        vol = getattr(page.state, "ctp_volume", None)
        if not vol:
            print("No CTP volume loaded")
            return

        pixels = vol["pixels"]  # (T, Z, H, W)
        T, Z, H, W = pixels.shape

        raw_times = np.asarray(vol.get("times", np.arange(T, dtype=float)))
        time_points_s = np.array([self._dicom_time_to_sec(t) for t in raw_times])
        time_points = time_points_s - time_points_s[0]
        pixel_spacing = vol.get("pixel_spacing", (0.5, 0.5))

        t_idx = min(max(0, int(page.var_ctp_time.get()) - 1), T - 1)
        z_idx = min(max(0, int(page.var_ctp_slice.get()) - 1), Z - 1)

        rect = page._ctp_display_rect
        if rect is None:
            return
        x0, y0, disp_w, disp_h, _, _ = rect

        # ── canvas → image coords ────────────────────────────────────
        poly_rows = np.array(
            [int(np.clip((cy - y0) * H / max(1, disp_h), 0, H - 1))
             for (_, cy) in canvas_points],
            dtype=int,
        )
        poly_cols = np.array(
            [int(np.clip((cx - x0) * W / max(1, disp_w), 0, W - 1))
             for (cx, _) in canvas_points],
            dtype=int,
        )

        print(f"[Freehand {lesion}] {len(poly_rows)} vertices")

        _SEG_WINDOW = 25

        # ── 1. contour (polygon branch) ──────────────────────────────
        contour = get_contour(
            x=poly_rows,
            y=poly_cols,
            search_window_size=_SEG_WINDOW,
            slice_idx=z_idx,
            time_point_idx=t_idx,
            four_d_image_set=pixels,
            pixel_spacing=pixel_spacing,
        )
        print(
            f"[{lesion}] freehand  radius={contour.radius_cm:.3f} cm  "
            f"area={contour.area_cm2:.4f} cm²"
        )

        # ── 2. sample (polygon branch) ───────────────────────────────
        sample_n = int(page.var_sample_roi.get().split("x")[0].strip())

        sample = get_best_sample(
            x=poly_rows,
            y=poly_cols,
            window_size=_SEG_WINDOW,
            roi_m=sample_n,
            roi_n=sample_n,
            slice_idx=z_idx,
            four_d_image_set=pixels,
            time_point_values=time_points,
        )
        interp_times = sample["interpolated_time_points"]
        interp_vals = sample["interpolated_sampled_points"]
        sx_rows = sample["search_window_rows"]
        sx_cols = sample["search_window_cols"]

        if len(interp_times) == 0 or len(interp_vals) == 0:
            print("ERROR: sampled points are empty!")
            return

        print(
            f"Sampled curve [{lesion}]: {len(interp_times)} pts  "
            f"t=[{interp_times[0]:.1f}…{interp_times[-1]:.1f}]  "
            f"val=[{interp_vals[0]:.1f}…{interp_vals[-1]:.1f}]"
        )

        # ── 3. ROI object ────────────────────────────────────────────
        roi_obj = get_roi(
            study_name=getattr(page.state, "study_name", "Unknown"),
            num_time_points=T,
            num_slices=Z,
            x=int(poly_rows[0]),
            y=int(poly_cols[0]),
            z=z_idx,
            t=t_idx,
            sampled_curve=interp_vals.tolist(),
            time_points=time_points.tolist(),
            fitted_curve=[],
            fitted_time_points=[],
            roi_x_boundary=sx_rows.tolist(),
            roi_y_boundary=sx_cols.tolist(),
        )

        if lesion == "pre":
            page.pre_roi = roi_obj
        else:
            page.post_roi = roi_obj

        bundle = {
            "preRoiObject": page._roi_to_dict(page.pre_roi),
            "postRoiObject": page._roi_to_dict(page.post_roi),
        }
        from roi_json import save_roi_as_json
        path = save_roi_as_json(bundle)
        print(f"ROI saved → {path}")

        # ── 4. overlay ───────────────────────────────────────────────
        image_2d = pixels[t_idx, z_idx].copy().astype(np.float32)
        overlaid = get_roi_overlayed(image_2d, sx_rows, sx_cols, overlay_value=1500.0)
        page._render_ctp_image_with(overlaid)

        # redraw polygon outline on canvas
        page.img_canvas.delete("freehand_poly")
        self._draw_overlay(poly_rows, poly_cols, H, W)

        # ── 5. curve block ───────────────────────────────────────────
        block = page.pre_lesion_block if lesion == "pre" else page.post_lesion_block
        block.times = interp_times.tolist()
        block.values = interp_vals.tolist()
        block.selected_idx = None
        block._undo_stack.clear()

        times_arr = np.asarray(block.times, dtype=float)
        values_arr = np.asarray(block.values, dtype=float)
        page._draw_curve_to_block(block, times_arr, values_arr)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _draw_overlay(self, img_rows, img_cols, img_h, img_w) -> None:
        """Draw closed polygon outline on canvas from image coords."""
        canvas = self.page.img_canvas
        canvas.delete("freehand_overlay")

        if len(img_rows) < 2:
            return

        pts = []
        for r, c in zip(img_rows, img_cols):
            cx, cy = self.page._image_to_canvas(r, c, img_h, img_w)
            pts.extend([cx, cy])
        # close
        cx0, cy0 = self.page._image_to_canvas(
            img_rows[0], img_cols[0], img_h, img_w
        )
        pts.extend([cx0, cy0])

        canvas.create_line(
            *pts, fill="#00FF00", width=2, tags="freehand_overlay",
        )

    @staticmethod
    def _dicom_time_to_sec(t_val) -> float:
        if t_val > 10000:
            hh = int(t_val // 10000)
            mm = int((t_val % 10000) // 100)
            ss = t_val % 100
            return hh * 3600 + mm * 60 + ss
        return float(t_val)