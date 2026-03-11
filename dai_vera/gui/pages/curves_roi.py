import customtkinter as ctk
import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_erosion

from dai_vera.gui.theme import THEME, FONTS
import numpy as np 
from datetime import datetime 

from dai_vera.roi import ROI

# get contour indices
def detect_roi_contour(image, x, y, window):
    half = window // 2
    sx = max(0, x - half)
    sy = max(0, y - half)
    ex = min(image.shape[0], x + half)
    ey = min(image.shape[1], y + half)

    block = image[sx:ex, sy:ey]
    if block.size == 0:
        return None
    if block.max() == block.min():
        return None

    # MATLAB uses graythresh on full image, not just block
    # approximate by using full image percentile as threshold
    level = threshold_otsu(image)
    block_binary = block > level

    labeled = label(block_binary)
    if labeled.max() == 0:
        return None

    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)

    # filled mask of largest region
    filled_mask = np.zeros_like(block_binary)
    filled_mask[largest.coords[:, 0], largest.coords[:, 1]] = True

    # boundary only = filled minus eroded (matches MATLAB bwperim)
    eroded = binary_erosion(filled_mask)
    boundary = filled_mask & ~eroded

    # filled coords for curve sampling (interior pixels)
    filled_coords = largest.coords.copy()
    filled_coords[:, 0] += sx
    filled_coords[:, 1] += sy

    # boundary coords for overlay drawing
    bY, bX = np.where(boundary)
    boundary_coords = np.column_stack((bY + sx, bX + sy))

    # measurements matching MATLAB getLongAndShortAxis
    props = regionprops(filled_mask.astype(int))[0]
    long_axis = props.major_axis_length
    short_axis = props.minor_axis_length

    return {
        "filled_coords": filled_coords,    # for curve sampling
        "boundary_coords": boundary_coords, # for overlay drawing
        "long_axis": long_axis,
        "short_axis": short_axis,
        "area": largest.area,
    }

class CurvesROIPage(ctk.CTkFrame):
    key = "curves_roi"

    def __init__(self, master, app_state):
        super().__init__(master, fg_color=THEME["bg"])
        self.state = app_state

        self.grid_columnconfigure(0, weight=1, uniform="half")
        self.grid_columnconfigure(1, weight=1, uniform="half")
        self.grid_rowconfigure(0, weight=1)

        # ---------------- LEFT (SCROLLABLE) ----------------
        self.left_outer = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        self.left_outer.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        self.left_outer.grid_rowconfigure(0, weight=1)
        self.left_outer.grid_columnconfigure(0, weight=1)

        self.left = ctk.CTkScrollableFrame(self.left_outer, fg_color="transparent")
        self.left.grid(row=0, column=0, sticky="nsew")
        self.left.grid_columnconfigure(0, weight=1)

        # ---------------- RIGHT (SCROLLABLE) ----------------
        self.right_outer = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        self.right_outer.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        self.right_outer.grid_rowconfigure(0, weight=1)
        self.right_outer.grid_columnconfigure(0, weight=1)

        self.right = ctk.CTkScrollableFrame(self.right_outer, fg_color="transparent")
        self.right.grid(row=0, column=0, sticky="nsew")
        self.right.grid_columnconfigure(0, weight=1)

        # Build left + right sections
        self.pre_lesion_block = None
        self.post_lesion_block = None
        self._build_left_ctp_and_controls()
        self._build_right_graphs()
        

        # movie control
        self._movie_after_id = None
        self.after(80, self._render_ctp_image)
        self.after(100, self._inject_test_volume)

        self._last_contour_coords = None

    # ---------------- LEFT ----------------
    def _build_left_ctp_and_controls(self):
        # CTP panel
        self.ctp_panel = ctk.CTkFrame(self.left, fg_color=THEME["panel_2"], corner_radius=16)
        self.ctp_panel.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        self.ctp_panel.grid_columnconfigure(0, weight=1)
        self.ctp_panel.grid_columnconfigure(1, weight=0)

        ctk.CTkLabel(self.ctp_panel, text="CTP Images", font=FONTS["h1"]).grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 8)
        )

        # Content row
        content = ctk.CTkFrame(self.ctp_panel, fg_color="transparent")
        content.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 10))
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=0)

        # Placeholder image canvas (replace with actual DICOM view later)
        img_box = ctk.CTkFrame(content, fg_color=THEME["panel_3"], corner_radius=14, height=280)
        img_box.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        img_box.grid_propagate(False)

        self.img_canvas = tk.Canvas(
            img_box,
            bg="black",
            highlightthickness=0
        )
        self.img_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.current_x = None
        self.current_y = None

        self.img_canvas.bind("<Button-1>", self._on_image_click)

        self.lbl_ctp_source = ctk.CTkLabel(
            img_box,
            text="",
            font=FONTS["body"],
            text_color=THEME["muted"],
        )
        self.lbl_ctp_source.place(relx=0.5, rely=0.5, anchor="center")

        # Slice slider
        slice_col = ctk.CTkFrame(content, fg_color="transparent")
        slice_col.grid(row=0, column=1, sticky="ns")

        ctk.CTkLabel(slice_col, text="Slice", text_color=THEME["muted"], font=FONTS["small"]).pack(pady=(6, 6))

        self.var_ctp_slice = ctk.IntVar(value=int(self.state.ctp_slice))
        self.slider_ctp_slice = ctk.CTkSlider(
            slice_col, 
            from_=0, 
            to=100, 
            number_of_steps=100,
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

        # Time slider
        time_row = ctk.CTkFrame(self.ctp_panel, fg_color="transparent")
        time_row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 14))
        time_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(time_row, text="Time Points", text_color=THEME["muted"], font=FONTS["small"]).grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )

        self.var_ctp_time = ctk.IntVar(value=int(self.state.ctp_time))
        self.slider_ctp_time = ctk.CTkSlider(
            time_row, 
            from_=0, 
            to=100, 
            number_of_steps=100,
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

        # Controls box under image
        self.controls = ctk.CTkFrame(self.left, fg_color=THEME["panel_2"], corner_radius=16)
        self.controls.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        self.controls.grid_columnconfigure(0, weight=1)

        # Row: Sample ROI + Search ROI
        row1 = ctk.CTkFrame(self.controls, fg_color="transparent")
        row1.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))
        row1.grid_columnconfigure(1, weight=1)
        row1.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(row1, text="Sample ROI", font=FONTS["body"]).grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.var_sample_roi = ctk.StringVar(value="2 x 2")
        self.dd_sample_roi = ctk.CTkOptionMenu(
            row1,
            values=["2 x 2", "4 x 4", "6 x 6", "8 x 8"],
            variable=self.var_sample_roi,
            fg_color=THEME["input_bg"],
            button_color=THEME["border"],
            button_hover_color=THEME["border_2"],
            dropdown_fg_color=THEME["panel_2"],
            dropdown_hover_color=THEME["border"],
            height=34,
        )
        self.dd_sample_roi.grid(row=0, column=1, sticky="ew")

        ctk.CTkLabel(row1, text="Search ROI", font=FONTS["body"]).grid(row=0, column=2, sticky="w", padx=(18, 10))
        self.var_search_roi = ctk.StringVar(value="1 x 1")
        self.dd_search_roi = ctk.CTkOptionMenu(
            row1,
            values=["1 x 1", "2 x 2", "3 x 3", "4 x 4"],
            variable=self.var_search_roi,
            fg_color=THEME["input_bg"],
            button_color=THEME["border"],
            button_hover_color=THEME["border_2"],
            dropdown_fg_color=THEME["panel_2"],
            dropdown_hover_color=THEME["border"],
            height=34,
        )
        self.dd_search_roi.grid(row=0, column=3, sticky="ew")

        # Row: Interpolate current slice dropdown
        row2 = ctk.CTkFrame(self.controls, fg_color="transparent")
        row2.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        row2.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(row2, text="Interpolate Current Slice", font=FONTS["body"]).grid(row=0, column=0, sticky="w")
        self.var_interpolate = ctk.StringVar(value="With Next Slice")
        self.dd_interpolate = ctk.CTkOptionMenu(
            row2,
            values=["With Next Slice", "With Previous Slice", "Off"],
            variable=self.var_interpolate,
            fg_color=THEME["input_bg"],
            button_color=THEME["border"],
            button_hover_color=THEME["border_2"],
            dropdown_fg_color=THEME["panel_2"],
            dropdown_hover_color=THEME["border"],
            height=34,
        )
        self.dd_interpolate.grid(row=0, column=1, sticky="ew", padx=(12, 0))

        # Row: Length/Width sliders (full width)
        self.var_len = ctk.DoubleVar(value=float(self.state.ctp_length))
        self.var_wid = ctk.DoubleVar(value=float(self.state.ctp_width))

        self._slider_line(self.controls, "L", self.var_len, row=2)
        self._slider_line(self.controls, "W", self.var_wid, row=3)

        # Row: Set pre/post
        row3 = ctk.CTkFrame(self.controls, fg_color="transparent")
        row3.grid(row=4, column=0, sticky="ew", padx=14, pady=(10, 8))
        row3.grid_columnconfigure(0, weight=1)
        row3.grid_columnconfigure(1, weight=1)

        self.btn_set_pre = ctk.CTkButton(
            row3, text="Set Pre Lesion",
            fg_color=THEME["panel_3"], hover_color=THEME["border_2"],
            height=36, corner_radius=12,
            command=self._on_set_pre_lesion,
        )
        self.btn_set_pre.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.btn_set_post = ctk.CTkButton(
            row3, text="Set Post Lesion",
            fg_color=THEME["panel_3"], hover_color=THEME["border_2"],
            height=36, corner_radius=12,
            command=self._on_set_post_lesion,
        )
        self.btn_set_post.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        # Row: Play movie + speed + height positive checkbox
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
        self.dd_speed = ctk.CTkOptionMenu(
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
        )
        self.dd_speed.grid(row=0, column=1, sticky="w", padx=(12, 0))

        self.var_height_positive = ctk.BooleanVar(value=False)
        self.chk_height_positive = ctk.CTkCheckBox(
            row4,
            text="Height (positive)",
            variable=self.var_height_positive,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color=THEME["text"],
        )
        self.chk_height_positive.grid(row=0, column=2, sticky="e", padx=(12, 0))
    
    def draw_contour(self, boundary_coords):
        """Draw vessel boundary outline — matches MATLAB getROIOverlayed."""
        self._last_contour_coords = boundary_coords
        self.img_canvas.delete("roi_contour")
        for x, y in boundary_coords:
            self.img_canvas.create_oval(
                y, x, y + 2, x + 2,    # slightly larger so it's visible
                fill="lime", outline="", tags="roi_contour"
            )

    def _on_image_click(self, event):
        self.current_x = event.x
        self.current_y = event.y
        self._draw_crosshair(event.x, event.y)

    def _draw_crosshair(self, x, y):
        self.img_canvas.delete("crosshair")
        size = 6
        self.img_canvas.create_line(
            x - size, y, x + size, y,
            fill="red", width=2, tags="crosshair"
        )
        self.img_canvas.create_line(
            x, y - size, x, y + size,
            fill="red", width=2, tags="crosshair"
        )
    def draw_pre_roi(self, roi):
        self.img_canvas.delete("pre_roi")

        half = roi.size // 2

        x0 = self.current_x - half
        y0 = self.current_y - half
        x1 = self.current_x + half
        y1 = self.current_y + half

        self.img_canvas.create_rectangle(
            x0, y0, x1, y1,
            outline="lime",
            width=2,
            tags="pre_roi"
        )
    
    # Pre curve generator
    def update_pre_curve(self, filled_coords, image_shape):
        vol = getattr(self.state, "ctp_volume", None)
        if not vol:
            return

        pixels = vol["pixels"]        # (T, Z, H, W)
        T, Z, H, W = pixels.shape
        z_idx = min(max(0, int(self.var_ctp_slice.get()) - 1), Z - 1)

        # build mask from filled region coords
        mask = np.zeros(image_shape, dtype=bool)
        rows = np.clip(filled_coords[:, 0], 0, image_shape[0] - 1)
        cols = np.clip(filled_coords[:, 1], 0, image_shape[1] - 1)
        mask[rows, cols] = True

        if not mask.any():
            print("Mask is empty — no pixels to sample")
            return

        # use real time values from volume if available
        vol_times = vol.get("times", None)
        if vol_times and len(vol_times) == T:
            times = np.array(vol_times, dtype=float)
            if len(times) > 1 and times[1] >= 500:
                times = times / 1000.0
        else:
            times = np.arange(T, dtype=float)

        values = np.array([
            float(np.mean(pixels[t, z_idx][mask])) for t in range(T)
        ])

        if self.var_height_positive.get():
            values = np.abs(values)

        block = self.pre_lesion_block
        block.points = [(float(t), float(v)) for t, v in zip(times, values)]
        block.fitted_time = None
        block.fitted_curve = None

        # rescale axes to fit actual data
        ax = block.ax
        ax.set_xlim(float(times[0]) - 0.5, float(times[-1]) + 0.5)
        y_min, y_max = float(values.min()), float(values.max())
        pad = max(1.0, (y_max - y_min) * 0.15)
        ax.set_ylim(y_min - pad, y_max + pad)

        # update range sliders to span full time range
        t_start = int(round(float(times[0])))
        t_end   = int(round(float(times[-1])))
        block.range_start = t_start
        block.range_end   = t_end
        block.var_start.set(t_start)
        block.var_end.set(t_end)
        block.s_start.configure(from_=t_start, to=max(t_start+1, t_end),
                                number_of_steps=max(1, T - 1))
        block.s_end.configure(from_=t_start, to=max(t_start+1, t_end),
                            number_of_steps=max(1, T - 1))
        block.start_line.set_xdata([t_start, t_start])
        block.end_line.set_xdata([t_end, t_end])

        self._redraw_curve(block)

    def _slider_line(self, parent, label, var, row):
        line = ctk.CTkFrame(parent, fg_color="transparent")
        line.grid(row=row, column=0, sticky="ew", padx=14, pady=6)
        line.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(line, text=label, font=FONTS["body"]).grid(row=0, column=0, sticky="w", padx=(0, 10))
        s = ctk.CTkSlider(
            line, from_=0, to=1, variable=var,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
            command=lambda _=None: self._sync_len_wid_to_state(),
        )
        s.grid(row=0, column=1, sticky="ew")

    def _sync_len_wid_to_state(self):
        self.state.ctp_length = float(self.var_len.get())
        self.state.ctp_width = float(self.var_wid.get())

    def _render_ctp_image(self):
        vol = getattr(self.state, "ctp_volume", None)
        if not vol:
            self.lbl_ctp_source.configure(text="No CTP loaded")
            return

        pixels = vol["pixels"]   # (T, Z, H, W)
        T, Z, H, W = pixels.shape

        # clamp vars BEFORE reconfiguring sliders to avoid ZeroDivisionError
        self.var_ctp_slice.set(min(max(1, int(self.var_ctp_slice.get())), Z))
        self.var_ctp_time.set(min(max(1, int(self.var_ctp_time.get())), T))

        self.slider_ctp_slice.configure(from_=1, to=max(2, Z), number_of_steps=max(1, Z - 1))
        self.slider_ctp_time.configure(from_=1, to=max(2, T), number_of_steps=max(1, T - 1))

        t_idx = int(self.var_ctp_time.get()) - 1
        z_idx = int(self.var_ctp_slice.get()) - 1

        img = pixels[t_idx, z_idx]

        img8 = self._to_uint8(img,
                            length=float(getattr(self.state, "ctp_length", 0.5)),
                            width=float(getattr(self.state, "ctp_width", 0.5)))

        cw = max(10, self.img_canvas.winfo_width())
        ch = max(10, self.img_canvas.winfo_height())

        from PIL import Image, ImageTk
        pil = Image.fromarray(img8).resize((cw, ch))
        photo = ImageTk.PhotoImage(pil)
        self._ctp_photo = photo

        self.img_canvas.delete("all")
        self.img_canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")
        self.lbl_ctp_source.configure(text="")

        if self._last_contour_coords is not None:
            self.draw_contour(self._last_contour_coords)

    def _to_uint8(self, img: np.ndarray, length: float, width: float) -> np.ndarray:
        lo = np.percentile(img, 1)
        hi = np.percentile(img, 99)
        if hi <= lo:
            hi = lo + 1.0
        w_scale = 0.25 + width * 1.75
        center = (lo + hi) / 2.0 + (length - 0.5) * (hi - lo) * 0.5
        span = (hi - lo) * w_scale
        out = np.clip((img - (center - span / 2)) / span, 0, 1)
        return (out * 255).astype(np.uint8)
    

    def _on_ctp_slice_change(self, _=None):
        self.lbl_ctp_slice_val.configure(text=str(int(self.var_ctp_slice.get())))
        self.state.ctp_slice = int(self.var_ctp_slice.get())
        self._render_ctp_image()        


    def _on_ctp_time_change(self, _=None):
        self.lbl_ctp_time_val.configure(text=str(int(self.var_ctp_time.get())))
        self.state.ctp_time = int(self.var_ctp_time.get())
        self._render_ctp_image()
        
    def _on_set_pre_lesion(self):
        if self.current_x is None or self.current_y is None:
            return

        vol = getattr(self.state, "ctp_volume", None)
        if not vol:
            return

        pixels = vol["pixels"]
        T, Z, H, W = pixels.shape

        t_idx = min(max(0, int(self.var_ctp_time.get()) - 1), T - 1)
        z_idx = min(max(0, int(self.var_ctp_slice.get()) - 1), Z - 1)
        image = pixels[t_idx, z_idx]

        search_size = int(self.var_search_roi.get().split("x")[0].strip())

        result = detect_roi_contour(
            image,
            self.current_y,
            self.current_x,
            max(search_size * 20, 40)
        )

        if result is None:
            print("No contour found — click on a brighter vessel region")
            return

        print(f"Contour: {len(result['boundary_coords'])} boundary px, "
            f"{len(result['filled_coords'])} filled px, "
            f"long={result['long_axis']:.1f}, short={result['short_axis']:.1f}")

        # boundary for drawing, filled for sampling
        self.draw_contour(result["boundary_coords"])
        self.update_pre_curve(result["filled_coords"], image.shape)

        self.state.pre_lesion_roi = {
            "x": self.current_x,
            "y": self.current_y,
            "z": z_idx,
            "t": t_idx,
            "long_axis": result["long_axis"],
            "short_axis": result["short_axis"],
            "area": result["area"],
            "time_series": [v for _, v in self.pre_lesion_block.points],
        }

        self._add_draggable_pre_point(self.current_x, self.current_y)
        self._save_roi_as_json()
        
    def _on_set_post_lesion(self):
        pass

    def _toggle_movie(self):
        if self._movie_after_id is not None:
            self.after_cancel(self._movie_after_id)
            self._movie_after_id = None
            self.btn_play.configure(text="Play Movie")
            return

        self.btn_play.configure(text="Stop")
        self._movie_loop()

    def _movie_loop(self):
        speed = self.var_speed.get().replace("x", "")
        try:
            speed = float(speed)
        except Exception:
            speed = 0.5

        delay = int(max(40, 250 / max(speed, 0.1)))

        vol = getattr(self.state, "ctp_volume", None)
        max_t = 100
        if vol:
            max_t = vol["pixels"].shape[0]   # T dimension

        cur = int(self.var_ctp_time.get())
        nxt = cur + 1 if cur < max_t else 1
        self.var_ctp_time.set(nxt)
        self._on_ctp_time_change()

        self._movie_after_id = self.after(delay, self._movie_loop)

    # ---------------- RIGHT: GRAPHS ----------------
    def _build_right_graphs(self):
        self._build_curve_block(
            parent=self.right,
            title="Pre Lesion Curve",
            row=0,
            storage_attr="pre_points",
            start_attr="pre_start",
            end_attr="pre_end",
        )
        self._build_curve_block(
            parent=self.right,
            title="Post Lesion Curve",
            row=1,
            storage_attr="post_points",
            start_attr="post_start",
            end_attr="post_end",
        )

    def _build_curve_block(self, parent, title, row, storage_attr, start_attr, end_attr):
        block = ctk.CTkFrame(parent, fg_color=THEME["panel_2"], corner_radius=16)
        block.grid(row=row, column=0, sticky="ew", padx=14, pady=(14, 10) if row == 0 else (0, 14))
        block.grid_columnconfigure(0, weight=1)
        # ✅ allow graph row to expand vertically
        block.grid_rowconfigure(1, weight=1)

        # header row with undo/clear (no extra "boxed" look)
        header = ctk.CTkFrame(block, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text=title, font=FONTS["h1"]).grid(row=0, column=0, sticky="w")

        btns = ctk.CTkFrame(header, fg_color="transparent")
        btns.grid(row=0, column=1, sticky="e")

        btn_undo = ctk.CTkButton(
            btns, text="Undo",
            fg_color=THEME["panel_3"], hover_color=THEME["border_2"],
            height=30, corner_radius=10,
            command=lambda: self._curve_undo(block),
        )
        btn_undo.pack(side="left", padx=(0, 10))

        btn_clear = ctk.CTkButton(
            btns, text="Clear",
            fg_color=THEME["panel_3"], hover_color=THEME["border_2"],
            height=30, corner_radius=10,
            command=lambda: self._curve_clear(block),
        )
        btn_clear.pack(side="left")

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

        # ✅ pad so xlabel is visible
        fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.18)

        # per-block storage
        block.points = []
        block.range_start = 0
        block.range_end = 10

        # range lines (green)
        block.start_line = ax.axvline(block.range_start, color=THEME["accent"], linewidth=2)
        block.end_line = ax.axvline(block.range_end, color=THEME["accent"], linewidth=2)

        block.fig = fig
        block.ax = ax

        block.canvas = FigureCanvasTkAgg(fig, master=block)
        w = block.canvas.get_tk_widget()
        w.configure(bg="black", highlightthickness=0)
        # ✅ expand both directions so it fills and doesn't clip labels
        w.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))

        # click adds a point (placeholder for curve selection logic)
        def on_click(event):
            if event.xdata is None or event.ydata is None:
                return
            x = float(event.xdata)
            y = float(event.ydata)

            # restrict to selected time window
            if not (block.range_start <= x <= block.range_end):
                return 
            
            x = round(x) 

            block.points.append((x, y))
            # block.points.append((float(event.xdata), float(event.ydata)))
            block.points.sort(key=lambda p: p[0])
            self._redraw_curve(block)

        block.canvas.mpl_connect("button_press_event", on_click)

        # range controls
        controls = ctk.CTkFrame(block, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))
        controls.grid_columnconfigure(1, weight=1)
        controls.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(controls, text="Start", text_color=THEME["muted"], font=FONTS["small"]).grid(
            row=0, column=0, sticky="w"
        )
        block.var_start = ctk.IntVar(value=block.range_start)
        block.s_start = ctk.CTkSlider(
            controls, from_=0, to=10, number_of_steps=10,
            variable=block.var_start,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
        )
        block.s_start.grid(row=0, column=1, sticky="ew", padx=(8, 16))

        ctk.CTkLabel(controls, text="End", text_color=THEME["muted"], font=FONTS["small"]).grid(
            row=0, column=2, sticky="w"
        )
        block.var_end = ctk.IntVar(value=block.range_end)
        block.s_end = ctk.CTkSlider(
            controls, from_=0, to=10, number_of_steps=10,
            variable=block.var_end,
            fg_color=THEME["border"],
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_2"],
        )
        block.s_end.grid(row=0, column=3, sticky="ew", padx=(8, 0))

        def on_range_change(_=None):
            a = int(block.var_start.get())
            b = int(block.var_end.get())
            if a > b:
                a, b = b, a
                block.var_start.set(a)
                block.var_end.set(b)

            block.range_start = a
            block.range_end = b

            block.start_line.set_xdata([a, a])
            block.end_line.set_xdata([b, b])
            block.canvas.draw_idle()

        block.s_start.configure(command=on_range_change)
        block.s_end.configure(command=on_range_change)
        on_range_change()

        # replace the existing return with:
        if row == 0:
            self.pre_lesion_block = block
        else:
            self.post_lesion_block = block

    def _redraw_curve(self, block):
        ax = block.ax

        for line in ax.lines[:]:
            if line not in (block.start_line, block.end_line):
                line.remove()
        for coll in ax.collections[:]:
            coll.remove()

        if block.points:
            xs = [p[0] for p in block.points]
            ys = [p[1] for p in block.points]
            ax.plot(xs, ys, color="white", linewidth=1, alpha=0.6)
            ax.scatter(xs, ys, color="white", s=20, alpha=0.6)

        # cyan fitted curve on top if available
        fitted_t = getattr(block, "fitted_time", None)
        fitted_c = getattr(block, "fitted_curve", None)
        if fitted_t is not None and fitted_c is not None:
            ax.plot(fitted_t, fitted_c, color="cyan", linewidth=2)

        block.canvas.draw_idle()

    def _curve_undo(self, block):
        if block.points:
            block.points.pop()
        self._redraw_curve(block)

    def _curve_clear(self, block):
        block.points = []
        self._redraw_curve(block)

    def _add_draggable_pre_point(self, cx, cy):
        self.img_canvas.delete("drag_point")
        # unbind first to prevent stacking callbacks on repeated calls
        self.img_canvas.tag_unbind("drag_point", "<ButtonPress-1>")
        self.img_canvas.tag_unbind("drag_point", "<B1-Motion>")
        self.img_canvas.tag_unbind("drag_point", "<ButtonRelease-1>")

        r = 6
        self.img_canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline="cyan", width=2, fill="", tags="drag_point"
        )
        self._drag_start = None

        def on_press(event):
            self._drag_start = (event.x, event.y)

        def on_drag(event):
            self.img_canvas.delete("drag_point")
            self.img_canvas.create_oval(
                event.x - r, event.y - r, event.x + r, event.y + r,
                outline="cyan", width=2, fill="", tags="drag_point"
            )
            self._draw_crosshair(event.x, event.y)

        def on_release(event):
            self.current_x = event.x
            self.current_y = event.y
            self._on_set_pre_lesion()

        self.img_canvas.tag_bind("drag_point", "<ButtonPress-1>", on_press)
        self.img_canvas.tag_bind("drag_point", "<B1-Motion>", on_drag)
        self.img_canvas.tag_bind("drag_point", "<ButtonRelease-1>", on_release)
    
        
    def _save_roi_as_json(self):
        import json, os
        roi = getattr(self.state, "pre_lesion_roi", None)
        if not roi:
            return
        # make time_series JSON-serialisable
        safe = {k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in roi.items()}
        out = os.path.join(os.path.expanduser("~"), "pre_lesion_roi.json")
        with open(out, "w") as f:
            json.dump(safe, f, indent=2)
        print(f"ROI saved → {out}")

    def _inject_test_volume(self):
        """
        Creates a fake (T=24, Z=10, H=256, W=256) volume with a
        gamma-variate-shaped contrast bolus at a known pixel location
        so the pre-lesion curve pipeline can be tested end-to-end.
        """
        T, Z, H, W = 24, 10, 256, 256

        # gamma variate curve peaking around t=8
        t = np.arange(T, dtype=float)
        bolus = np.where(t > 4, 1.0 * (t - 4)**2.5 * np.exp(-(t - 4) / 1.5), 0.0)
        bolus = bolus / bolus.max() * 300   # scale to 300 HU peak

        pixels = np.random.normal(-50, 20, (T, Z, H, W)).astype(np.float32)

        # plant a bright vessel blob at row=128, col=128, all slices
        for t_i in range(T):
            pixels[t_i, :, 120:136, 120:136] += float(bolus[t_i])

        self.state.ctp_volume = {
            "pixels": pixels,
            "times": list(np.arange(T) * 2.0),   # 2s spacing → 0,2,4,...46s
            "zs": list(range(Z)),
            "shape": (T, Z, H, W),
        }
        self.state.ctp_slice = 1
        self.state.ctp_time = 1
        print("Test volume injected — click near centre (128, 128) of the image")
        self._render_ctp_image()