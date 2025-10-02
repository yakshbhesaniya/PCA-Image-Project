"""
Tkinter GUI for the PCT project
 - file selection
 - compute PCA
 - preview principal components (thumbnails + big preview)
 - reconstruct using top-k components and save outputs
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ..pct.processor import PCTProcessor
from ..pct.io import read_images_as_stack, save_image_uint8
from ..pct.utils import normalize_to_uint8, float_stack_to_scaled_uint8
import numpy as np

class PCAApp:
    def __init__(self, root):
        self.root = root
        root.title("PCT (PCA on Image Bands)")
        self.processor = PCTProcessor()
        self.filepaths = []
        self.orig_dtype = None
        self.orig_min = None
        self.orig_max = None

        # store PhotoImage refs to avoid GC
        self._thumb_refs = []
        self._big_ref = None

        self._build_ui()

    def _build_ui(self):
        # top frame — controls
        frm_top = ttk.Frame(self.root, padding=8)
        frm_top.pack(fill=tk.X, side=tk.TOP)

        btn_select = ttk.Button(frm_top, text="Select Input File(s)", command=self.select_files)
        btn_select.pack(side=tk.LEFT, padx=4)

        btn_load = ttk.Button(frm_top, text="Load & Compute PCA", command=self.load_and_compute)
        btn_load.pack(side=tk.LEFT, padx=4)

        ttk.Label(frm_top, text=" k (reconstruct):").pack(side=tk.LEFT, padx=(10,2))
        self.entry_k = ttk.Entry(frm_top, width=5)
        self.entry_k.insert(0, "3")
        self.entry_k.pack(side=tk.LEFT)

        btn_recon = ttk.Button(frm_top, text="Reconstruct & Save", command=self.reconstruct_and_save)
        btn_recon.pack(side=tk.LEFT, padx=6)

        btn_quit = ttk.Button(frm_top, text="Quit", command=self.root.quit)
        btn_quit.pack(side=tk.RIGHT)

        # main panes
        main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # left: file list & parameters
        left_frame = ttk.Frame(main_pane, width=260)
        main_pane.add(left_frame, weight=0)

        tk.Label(left_frame, text="Selected files:").pack(anchor="w")
        self.lst_files = tk.Listbox(left_frame, height=8)
        self.lst_files.pack(fill=tk.X, pady=4)

        lbl_info = tk.Label(left_frame, text="Info: No image loaded.", wraplength=240, justify="left")
        lbl_info.pack(fill=tk.X, pady=(6,0))
        self.lbl_info = lbl_info

        # preview: center - big view
        center_frame = ttk.Frame(main_pane)
        main_pane.add(center_frame, weight=1)

        self.canvas_big = tk.Canvas(center_frame, bg="#222", height=420)
        self.canvas_big.pack(fill=tk.BOTH, expand=True)

        # control to navigate PCs
        nav_frame = ttk.Frame(center_frame)
        nav_frame.pack(fill=tk.X, pady=4)
        ttk.Label(nav_frame, text="PC index:").pack(side=tk.LEFT)
        self.slider_pc = ttk.Scale(nav_frame, from_=1, to=6, orient=tk.HORIZONTAL, command=self.on_pc_slider)
        self.slider_pc.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=6)
        self.lbl_pc = ttk.Label(nav_frame, text="PC 1")
        self.lbl_pc.pack(side=tk.RIGHT)

        # right: thumbnails
        right_frame = ttk.Frame(main_pane, width=260)
        main_pane.add(right_frame, weight=0)
        ttk.Label(right_frame, text="Principal Components (thumbnails)").pack(anchor="w")
        self.frm_thumbs = ttk.Frame(right_frame)
        self.frm_thumbs.pack(fill=tk.BOTH, expand=True, pady=4)

        # bottom status
        self.status = ttk.Label(self.root, text="Ready.", relief=tk.SUNKEN, anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def set_status(self, txt):
        self.status.config(text=txt)
        self.root.update_idletasks()

    def select_files(self):
        paths = filedialog.askopenfilenames(title="Select input image (single multiband or many single-band files)")
        if not paths:
            return
        self.filepaths = list(paths)
        self.lst_files.delete(0, tk.END)
        for p in self.filepaths:
            self.lst_files.insert(tk.END, os.path.basename(p))
        self.set_status(f"Selected {len(self.filepaths)} files.")

    def load_and_compute(self):
        if not self.filepaths:
            messagebox.showerror("No files", "Please select input file(s) first.")
            return
        try:
            self.set_status("Reading images...")
            stack, orig_dtype, omin, omax = read_images_as_stack(self.filepaths)
            self.orig_dtype = orig_dtype
            self.orig_min = omin
            self.orig_max = omax
            self.processor.load_stack(stack)
            self.set_status("Computing PCA (covariance, eigen) ...")
            self.processor.compute_pca()
            B = self.processor.B
            self.lbl_info.config(text=f"H={self.processor.H}, W={self.processor.W}, B={B}\norig dtype={self.orig_dtype}\norig range=[{self.orig_min:.1f}, {self.orig_max:.1f}]")
            # update slider
            self.slider_pc.config(from_=1, to=max(1, B))
            self.slider_pc.set(1)
            self.update_thumbnails(n_show=min(12, B))
            self.show_large_pc(0)
            self.set_status("PCA computed.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.set_status("Error during load/compute.")

    def update_thumbnails(self, n_show=6):
        # clear existing
        for child in self.frm_thumbs.winfo_children():
            child.destroy()
        self._thumb_refs.clear()
        B = self.processor.B
        cols = 2
        n_show = min(n_show, B)
        # create grid of thumbnails
        for i in range(n_show):
            pc = self.processor.get_pc_image(i)
            u8 = normalize_to_uint8(pc)
            pil = Image.fromarray(u8)
            pil.thumbnail((240, 120))
            ph = ImageTk.PhotoImage(pil)
            btn = tk.Button(self.frm_thumbs, image=ph, command=lambda idx=i: self.on_thumb_click(idx))
            btn.grid(row=i//cols, column=i%cols, padx=4, pady=4)
            self._thumb_refs.append(ph)  # keep a reference

    def on_thumb_click(self, idx):
        self.slider_pc.set(idx+1)
        self.show_large_pc(idx)

    def on_pc_slider(self, _ev=None):
        idx = int(round(self.slider_pc.get())) - 1
        idx = max(0, min(self.processor.B - 1, idx))
        self.lbl_pc.config(text=f"PC {idx+1}")
        self.show_large_pc(idx)

    def show_large_pc(self, idx):
        pc = self.processor.get_pc_image(idx)
        u8 = normalize_to_uint8(pc)
        pil = Image.fromarray(u8).convert("RGB")
        # fit to canvas size
        cw = self.canvas_big.winfo_width() or 800
        ch = self.canvas_big.winfo_height() or 420
        pil.thumbnail((cw-10, ch-10))
        self._big_ref = ImageTk.PhotoImage(pil)
        self.canvas_big.delete("all")
        self.canvas_big.create_image(cw//2, ch//2, image=self._big_ref)
        self.set_status(f"Showing PC {idx+1}")

    def reconstruct_and_save(self):
        if self.processor.pcs is None:
            messagebox.showerror("Error", "Compute PCA first.")
            return
        try:
            k = int(self.entry_k.get())
            if k < 1 or k > self.processor.B:
                raise ValueError("k out of range")
        except Exception as e:
            messagebox.showerror("Invalid k", "Enter an integer k between 1 and number of bands.")
            return

        outdir = filedialog.askdirectory(title="Select output folder (will create outputs/ if not present)")
        if not outdir:
            return
        os.makedirs(outdir, exist_ok=True)
        try:
            self.set_status("Reconstructing...")
            rec = self.processor.reconstruct(k)
            mse = self.processor.compute_mse(rec)
            # Save PC images
            pc_dir = os.path.join(outdir, "PC_images")
            os.makedirs(pc_dir, exist_ok=True)
            for i in range(self.processor.B):
                pc = self.processor.get_pc_image(i)
                u8 = normalize_to_uint8(pc)
                path = os.path.join(pc_dir, f"PC_{i+1:02d}.png")
                save_image_uint8(path, u8)

            # Save reconstructed images (npy + band PNGs + rgb if 3 bands)
            stack_u8 = float_stack_to_scaled_uint8(rec, orig_min=self.orig_min, orig_max=self.orig_max)
            # per-band PNGs
            recon_bands_dir = os.path.join(outdir, "reconstructed_bands")
            os.makedirs(recon_bands_dir, exist_ok=True)
            for b in range(self.processor.B):
                save_image_uint8(os.path.join(recon_bands_dir, f"recon_band_{b+1:02d}.png"), stack_u8[:,:,b])

            # Save RGB if 3 bands
            if self.processor.B == 3:
                save_image_uint8(os.path.join(outdir, "reconstructed_rgb.png"), stack_u8)

            # Save npy (float)
            np.save(os.path.join(outdir, "reconstructed_stack.npy"), rec)

            # Save a small report
            with open(os.path.join(outdir, "PCA_report.txt"), "w") as f:
                f.write("PCT / PCA Report\n")
                f.write("=================\n")
                f.write(f"Input files: {self.filepaths}\n")
                f.write(f"Image shape: H={self.processor.H}, W={self.processor.W}, B={self.processor.B}\n")
                f.write(f"k used for reconstruction: {k}\n")
                f.write(f"MSE: {mse:.6f}\n\n")
                f.write("Eigenvalues (descending):\n")
                for idx, v in enumerate(self.processor.eigvals):
                    f.write(f"  {idx+1}: {v:.6e}\n")
            self.set_status(f"Saved outputs to {outdir}  —  MSE={mse:.6f}")
            messagebox.showinfo("Saved", f"Outputs saved to:\n{outdir}\n\nMSE = {mse:.6f}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            self.set_status("Error while saving outputs.")
