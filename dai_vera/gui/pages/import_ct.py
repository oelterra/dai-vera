# dai_vera/gui/pages/import_ct.py
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from dai_vera.gui.theme import THEME, FONTS


class ImportCTPage(ctk.CTkFrame):
    key = "import_ct"
    title = "CTA / CTP Import and Prep"

    def __init__(self, master):
        super().__init__(master, fg_color=THEME["bg"])

        # ---------- top title ----------
        ctk.CTkLabel(
            self,
            text=self.title,
            font=FONTS["title"],
            text_color=THEME["text"],
        ).pack(anchor="w", padx=16, pady=(12, 8))

        # ---------- main two-column layout ----------
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        main.grid_columnconfigure(0, weight=1, uniform="col")
        main.grid_columnconfigure(1, weight=2, uniform="col")
        main.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(main, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right = ctk.CTkFrame(main, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # Variables to hold selected folder paths
        self.cta_path = ctk.StringVar(value="No folder selected")
        self.ctp_path = ctk.StringVar(value="No folder selected")

        # ---------- left: CTA + CTP upload panels ----------
        left.grid_rowconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)

        self._build_upload_panel(
            parent=left,
            title="CTA Upload",
            subtitle="Select a CTA folder (DICOM series or dataset folder)",
            path_var=self.cta_path,
            on_browse=self._browse_cta,
            row=0,
        )

        self._build_upload_panel(
            parent=left,
            title="CTP Upload",
            subtitle="Select a CTP folder (DICOM series or dataset folder)",
            path_var=self.ctp_path,
            on_browse=self._browse_ctp,
            row=1,
        )

        # ---------- right: parameters panel ----------
        self._build_parameters_panel(right)

    # ---------------------- Folder picking (Mac-safe) ----------------------

    def _pick_folder(self, title: str) -> str:
        """
        Mac fix: create a hidden Tk root so the folder dialog appears on top.
        Without this, dialogs sometimes open behind the main window.
        """
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

        path = filedialog.askdirectory(title=title, mustexist=True)
        root.destroy()
        return path

    # ---------------------- UI building blocks ----------------------

    def _build_upload_panel(self, parent, title, subtitle, path_var, on_browse, row: int):
        panel = ctk.CTkFrame(parent, fg_color=THEME["panel"], corner_radius=12)
        panel.grid(row=row, column=0, sticky="nsew", pady=(0, 12) if row == 0 else (12, 0))
        panel.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text=title, font=FONTS["h1"], text_color=THEME["text"]).grid(
            row=0, column=0, sticky="w"
        )
        ctk.CTkLabel(header, text=subtitle, font=FONTS["small"], text_color=THEME["muted"]).grid(
            row=1, column=0, sticky="w", pady=(2, 0)
        )

        path_box = ctk.CTkFrame(panel, fg_color=THEME["panel_2"], corner_radius=10)
        path_box.grid(row=1, column=0, sticky="ew", padx=14, pady=(6, 10))
        path_box.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            path_box,
            textvariable=path_var,
            font=FONTS["small"],
            text_color=THEME["text"],
            wraplength=340,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=10)

        btn_row = ctk.CTkFrame(panel, fg_color="transparent")
        btn_row.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 12))
        btn_row.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(
            btn_row,
            text="Browse Folder",
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            corner_radius=10,
            height=34,
            command=on_browse,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            btn_row,
            text="Clear",
            fg_color=THEME["border"],
            hover_color="#333333",
            text_color=THEME["text"],
            corner_radius=10,
            height=34,
            command=lambda: path_var.set("No folder selected"),
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

    def _build_parameters_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=THEME["panel"], corner_radius=12)
        panel.pack(fill="both", expand=True)
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(panel, text="Scan Info & Parameters", font=FONTS["h1"], text_color=THEME["text"]).grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 8)
        )

        form = ctk.CTkFrame(panel, fg_color="transparent")
        form.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        form.grid_columnconfigure(0, weight=1)

        # --- Scan info (placeholders) ---
        self.patient_id = ctk.StringVar(value="")
        self.study_name = ctk.StringVar(value="")

        self._labeled_entry(form, "Patient / Case ID", self.patient_id, row=0)
        self._labeled_entry(form, "Study Name", self.study_name, row=1)

        # --- Parameters (placeholders; we can rename these to match your exact names later) ---
        self.seg_threshold = ctk.DoubleVar(value=0.50)
        self.smoothing = ctk.DoubleVar(value=0.20)
        self.max_iters = ctk.IntVar(value=150)

        self._labeled_slider(form, "Segmentation Threshold", self.seg_threshold, 0.0, 1.0, row=2)
        self._labeled_slider(form, "Smoothing", self.smoothing, 0.0, 1.0, row=3)
        self._labeled_entry(form, "Max Iterations", self.max_iters, row=4)

        # Bottom row button
        bottom = ctk.CTkFrame(panel, fg_color="transparent")
        bottom.grid(row=2, column=0, sticky="ew", padx=14, pady=(6, 12))
        bottom.grid_columnconfigure(0, weight=1)

        self.compute_btn = ctk.CTkButton(
            bottom,
            text="Compute",
            fg_color=THEME["accent"],
            hover_color=THEME["accent_2"],
            text_color="black",
            corner_radius=10,
            height=38,
            command=self._on_compute,
        )
        self.compute_btn.grid(row=0, column=0, sticky="e")

    def _labeled_entry(self, parent, label, var, row: int):
        block = ctk.CTkFrame(parent, fg_color=THEME["panel_2"], corner_radius=10)
        block.grid(row=row, column=0, sticky="ew", pady=8)
        block.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(block, text=label, font=FONTS["small"], text_color=THEME["muted"]).grid(
            row=0, column=0, sticky="w", padx=10, pady=(8, 0)
        )
        entry = ctk.CTkEntry(block, textvariable=var, height=34)
        entry.grid(row=1, column=0, sticky="ew", padx=10, pady=(6, 10))

    def _labeled_slider(self, parent, label, var, vmin, vmax, row: int):
        block = ctk.CTkFrame(parent, fg_color=THEME["panel_2"], corner_radius=10)
        block.grid(row=row, column=0, sticky="ew", pady=8)
        block.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(block, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
        top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(top, text=label, font=FONTS["small"], text_color=THEME["muted"]).grid(
            row=0, column=0, sticky="w"
        )
        val = ctk.CTkLabel(top, text=f"{var.get():.2f}", font=FONTS["small"], text_color=THEME["text"])
        val.grid(row=0, column=1, sticky="e")

        slider = ctk.CTkSlider(block, from_=vmin, to=vmax, variable=var)
        slider.grid(row=1, column=0, sticky="ew", padx=10, pady=(6, 6))

        def _update_label(_):
            val.configure(text=f"{var.get():.2f}")

        slider.configure(command=_update_label)

    # ---------------------- actions ----------------------

    def _browse_cta(self):
        path = self._pick_folder("Select CTA folder")
        if path:
            self.cta_path.set(path)

    def _browse_ctp(self):
        path = self._pick_folder("Select CTP folder")
        if path:
            self.ctp_path.set(path)

    def _on_compute(self):
        msg = (
            f"CTA Folder:\n{self.cta_path.get()}\n\n"
            f"CTP Folder:\n{self.ctp_path.get()}\n\n"
            f"Threshold: {self.seg_threshold.get():.2f}\n"
            f"Smoothing: {self.smoothing.get():.2f}\n"
            f"Max Iters: {self.max_iters.get()}"
        )
        messagebox.showinfo("Compute (Placeholder)", msg)
