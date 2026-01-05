import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from dai_vera.gui.theme import THEME, FONTS


class FFRResultsPage(ctk.CTkFrame):
    key = "ffr_results"

    def __init__(self, master, app_state):
        super().__init__(master, fg_color=THEME["bg"])
        self.state = app_state

        # 2 columns, right narrower
        self.grid_columnconfigure(0, weight=3, uniform="main")
        self.grid_columnconfigure(1, weight=2, uniform="main")
        self.grid_rowconfigure(0, weight=1)

        # ---------------- LEFT: Graph panel ----------------
        left = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)

        graph_block = ctk.CTkFrame(left, fg_color=THEME["panel_2"], corner_radius=16)
        graph_block.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        graph_block.grid_columnconfigure(0, weight=1)
        graph_block.grid_rowconfigure(2, weight=1)

        # Header
        header = ctk.CTkFrame(graph_block, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text="FFR/Flow Velocity Graph", font=FONTS["h1"]).grid(
            row=0, column=0, sticky="w"
        )

        # Clear button (top left under title in the reference image)
        self.btn_graph_clear = ctk.CTkButton(
            graph_block,
            text="Clear",
            height=30,
            corner_radius=10,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            command=self._on_clear_graph,
        )
        self.btn_graph_clear.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 8))

        # Options row
        opts = ctk.CTkFrame(graph_block, fg_color="transparent")
        opts.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        opts.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(opts, text="Graph Appearance", font=FONTS["body"]).grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )
        self.entry_graph_appearance = ctk.CTkEntry(
            opts,
            height=32,
            fg_color=THEME["input_bg"],
            border_color=THEME["border"],
            text_color=THEME["text"],
            placeholder_text="",
        )
        self.entry_graph_appearance.grid(row=0, column=1, sticky="ew", padx=(0, 18))

        self.var_flip_slice = ctk.BooleanVar(value=False)
        self.chk_flip_slice = ctk.CTkCheckBox(
            opts,
            text="Flip Slice Numbers",
            variable=self.var_flip_slice,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color=THEME["text"],
        )
        self.chk_flip_slice.grid(row=0, column=2, sticky="e")

        # Matplotlib graph with 3 axes: left y (FFR), right y (Flow), bottom x (Slice)
        fig = Figure(figsize=(6, 5), dpi=100)
        fig.patch.set_facecolor("black")
        ax_ffr = fig.add_subplot(111)
        ax_ffr.set_facecolor("black")

        # Right y-axis
        ax_flow = ax_ffr.twinx()

        # Style axes
        for spine in ax_ffr.spines.values():
            spine.set_color("white")
        for spine in ax_flow.spines.values():
            spine.set_color("white")

        ax_ffr.tick_params(colors="white")
        ax_flow.tick_params(colors="white")

        ax_ffr.set_xlabel("Slice Number", labelpad=10, color="white")
        ax_ffr.set_ylabel("FFR", labelpad=10, color="white")
        ax_flow.set_ylabel("Flow Velocity (cm/s)", labelpad=12, color="white")

        # Placeholder ranges
        ax_ffr.set_xlim(0, 20)
        ax_ffr.set_ylim(0, 1.0)
        ax_flow.set_ylim(0, 20)

        # Pad so labels fit
        fig.subplots_adjust(left=0.12, right=0.88, bottom=0.14, top=0.95)

        self.fig = fig
        self.ax_ffr = ax_ffr
        self.ax_flow = ax_flow

        self.canvas = FigureCanvasTkAgg(fig, master=graph_block)
        w = self.canvas.get_tk_widget()
        w.configure(bg="black", highlightthickness=0)
        w.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))

        self._draw_placeholder_graph()

        # ---------------- RIGHT: Outputs + Save/Export ----------------
        right = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=18)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=0)

        # Output Parameters box
        out_box = ctk.CTkFrame(right, fg_color=THEME["panel_2"], corner_radius=16)
        out_box.grid(row=0, column=0, sticky="nsew", padx=14, pady=(14, 10))
        out_box.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(out_box, text="Output Parameters", font=FONTS["h1"]).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 12)
        )

        # Output fields (readonly entries for now)
        self.out_ffr = self._out_row(out_box, 1, "FFR")
        self.out_pre_flow = self._out_row(out_box, 2, "Pre-lesion Flow Velocity (cm/s)")
        self.out_post_flow = self._out_row(out_box, 3, "Post-lesion Flow Velocity (cm/s)")
        self.out_pre_shear = self._out_row(out_box, 4, "Pre-lesion Shear Press (Pa)")
        self.out_post_shear = self._out_row(out_box, 5, "Post-lesion Shear Press (Pa)")

        self.btn_clear_outputs = ctk.CTkButton(
            out_box,
            text="Clear Outputs",
            height=34,
            corner_radius=12,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            command=self._on_clear_outputs,
        )
        self.btn_clear_outputs.grid(row=6, column=0, columnspan=2, sticky="ew", padx=14, pady=(16, 14))

        # Save/Export box
        se_box = ctk.CTkFrame(right, fg_color=THEME["panel_2"], corner_radius=16)
        se_box.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        se_box.grid_columnconfigure(0, weight=1)
        se_box.grid_columnconfigure(1, weight=1)

        self.btn_save = ctk.CTkButton(
            se_box,
            text="Save",
            height=36,
            corner_radius=12,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            command=self._on_save,
        )
        self.btn_save.grid(row=0, column=0, sticky="ew", padx=(14, 8), pady=14)

        self.btn_export = ctk.CTkButton(
            se_box,
            text="Export",
            height=36,
            corner_radius=12,
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            command=self._on_export,
        )
        self.btn_export.grid(row=0, column=1, sticky="ew", padx=(8, 14), pady=14)

    # ---------- helpers ----------
    def _out_row(self, parent, row, label):
        ctk.CTkLabel(parent, text=label, font=FONTS["body"]).grid(
            row=row, column=0, sticky="w", padx=14, pady=8
        )
        e = ctk.CTkEntry(
            parent,
            height=34,
            fg_color=THEME["input_bg"],
            border_color=THEME["border"],
            text_color=THEME["text"],
        )
        e.grid(row=row, column=1, sticky="ew", padx=(12, 14), pady=8)
        return e

    def _draw_placeholder_graph(self):
        # Placeholder grid & empty plot (matches your screenshot vibe)
        self.ax_ffr.cla()
        self.ax_flow.cla()

        self.ax_ffr.set_facecolor("black")
        self.ax_ffr.set_xlim(0, 20)
        self.ax_ffr.set_ylim(0, 1.0)
        self.ax_ffr.set_xlabel("Slice Number", labelpad=10, color="white")
        self.ax_ffr.set_ylabel("FFR", labelpad=10, color="white")

        self.ax_flow.set_ylim(0, 20)
        self.ax_flow.set_ylabel("Flow Velocity (cm/s)", labelpad=12, color="white")

        for spine in self.ax_ffr.spines.values():
            spine.set_color("white")
        for spine in self.ax_flow.spines.values():
            spine.set_color("white")

        self.ax_ffr.tick_params(colors="white")
        self.ax_flow.tick_params(colors="white")

        # Draw tick marks similar to the reference (no data yet)
        self.ax_ffr.set_xticks(list(range(0, 21, 1)))
        self.ax_ffr.set_yticks([i / 10 for i in range(0, 11)])
        self.ax_flow.set_yticks(list(range(0, 21, 1)))

        self.canvas.draw_idle()

    # ---------- callbacks ----------
    def _on_clear_graph(self):
        self._draw_placeholder_graph()

    def _on_clear_outputs(self):
        for e in [self.out_ffr, self.out_pre_flow, self.out_post_flow, self.out_pre_shear, self.out_post_shear]:
            e.delete(0, "end")

    def _on_save(self):
        pass  # later: save results to state/file

    def _on_export(self):
        pass  # later: export report/csv/pdf
