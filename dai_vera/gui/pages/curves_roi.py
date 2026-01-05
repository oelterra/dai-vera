import customtkinter as ctk
import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from dai_vera.gui.theme import THEME, FONTS


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
        self._build_left_ctp_and_controls()
        self._build_right_graphs()

        # movie control
        self._movie_after_id = None

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

        # Placeholder image canvas (you’ll replace with actual DICOM view later)
        img_box = ctk.CTkFrame(content, fg_color=THEME["panel_3"], corner_radius=14, height=280)
        img_box.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        img_box.grid_propagate(False)

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
            slice_col, from_=1, to=100, number_of_steps=99,
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
            time_row, from_=1, to=100, number_of_steps=99,
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

    def _on_ctp_slice_change(self, _=None):
        self.lbl_ctp_slice_val.configure(text=str(int(self.var_ctp_slice.get())))
        self.state.ctp_slice = int(self.var_ctp_slice.get())

    def _on_ctp_time_change(self, _=None):
        self.lbl_ctp_time_val.configure(text=str(int(self.var_ctp_time.get())))
        self.state.ctp_time = int(self.var_ctp_time.get())

    def _on_set_pre_lesion(self):
        pass

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

        cur = int(self.var_ctp_slice.get())
        nxt = cur + 1
        if nxt > 100:
            nxt = 1
        self.var_ctp_slice.set(nxt)
        self._on_ctp_slice_change()

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
            block.points.append((float(event.xdata), float(event.ydata)))
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

    def _redraw_curve(self, block):
        ax = block.ax

        # remove old curve artists (keep the 2 range lines)
        # ax.lines includes start/end lines + curve line(s)
        # keep first two lines which are start_line and end_line
        kept = [block.start_line, block.end_line]
        ax.lines = kept

        # remove old scatters
        ax.collections.clear()

        if block.points:
            xs = [p[0] for p in block.points]
            ys = [p[1] for p in block.points]
            ax.plot(xs, ys, color="white", linewidth=1)
            ax.scatter(xs, ys, color="white", s=35)

        block.canvas.draw_idle()

    def _curve_undo(self, block):
        if block.points:
            block.points.pop()
        self._redraw_curve(block)

    def _curve_clear(self, block):
        block.points = []
        self._redraw_curve(block)
