import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from ..pct.processor import PCTProcessor
from ..pct.io import read_images_as_stack
from ..pct.utils import normalize_to_uint8


class PCAApp:
    def __init__(self, root):
        self.root = root
        root.title("PCT")
        self.processor = PCTProcessor()
        self.filepaths = []
        self.orig_dtype = None
        self.orig_min = None
        self.orig_max = None
        self._thumb_refs = []
        self._big_ref = None
        self._current_mode = "pc"  # "pc" or "recon"
        self.rec_bands = None
        self.current_band_idx = 0
        self._build_ui()

    # ------------------------- UI SETUP -------------------------
    def _build_ui(self):
        frm_top = ttk.Frame(self.root, padding=8)
        frm_top.pack(fill=tk.X, side=tk.TOP)

        ttk.Button(frm_top, text="Select Files", command=self.select_files).pack(side=tk.LEFT, padx=4)
        ttk.Button(frm_top, text="Load & Compute PCA", command=self.load_and_compute).pack(side=tk.LEFT, padx=4)
        ttk.Label(frm_top, text="k (reconstruct):").pack(side=tk.LEFT, padx=(10, 2))
        self.entry_k = ttk.Entry(frm_top, width=5)
        self.entry_k.insert(0, "3")
        self.entry_k.pack(side=tk.LEFT)
        ttk.Button(frm_top, text="Reconstruct", command=self.reconstruct_and_display).pack(side=tk.LEFT, padx=6)
        #ttk.Button(frm_top, text="Quit", command=self.root.quit).pack(side=tk.RIGHT)

        main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # LEFT
        left = ttk.Frame(main_pane, width=220)
        main_pane.add(left, weight=0)
        tk.Label(left, text="Selected files:").pack(anchor="w")
        self.lst_files = tk.Listbox(left, height=6)
        self.lst_files.pack(fill=tk.X, pady=4)
        self.lbl_info = tk.Label(left, text="No image loaded.", wraplength=200, justify="left")
        self.lbl_info.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(left, text="Report:").pack(anchor="w", pady=(10, 2))
        self.txt_report = tk.Text(left, height=12, wrap="word")
        self.txt_report.pack(fill=tk.BOTH, expand=True)

        # CENTER
        center = ttk.Frame(main_pane)
        main_pane.add(center, weight=1)
        ttk.Label(center, text="PC / Reconstruction Display").pack(anchor="w")
        self.canvas_big = tk.Canvas(center, bg="#111", height=420)
        self.canvas_big.pack(fill=tk.BOTH, expand=True)
        # Add slider for reconstructed bands
        self.slider_band = ttk.Scale(center, from_=0, to=0, orient=tk.HORIZONTAL, command=self._on_slider_move)
        self.slider_band.pack(fill=tk.X, pady=4)
        self.slider_band_label = ttk.Label(center, text="")
        self.slider_band_label.pack(anchor="center")

        # RIGHT
        right = ttk.Frame(main_pane, width=220)
        main_pane.add(right, weight=0)
        self.lbl_right_title = ttk.Label(right, text="Principal Components (thumbnails)")
        self.lbl_right_title.pack(anchor="w")
        self.frm_thumbs = ttk.Frame(right)
        self.frm_thumbs.pack(fill=tk.BOTH, expand=True)

        # STATUS BAR
        self.status = ttk.Label(self.root, text="Ready.", relief=tk.SUNKEN, anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------- UTILS -------------------------
    def set_status(self, txt):
        self.status.config(text=txt)
        self.root.update_idletasks()

    # ------------------------- FILE SELECTION -------------------------
    def select_files(self):
        paths = filedialog.askopenfilenames(title="Select image(s)")
        if not paths:
            return
        self.filepaths = list(paths)
        self.lst_files.delete(0, tk.END)
        for p in self.filepaths:
            self.lst_files.insert(tk.END, os.path.basename(p))
        self.set_status(f"Selected {len(paths)} files.")

    # ------------------------- PCA COMPUTE -------------------------
    def load_and_compute(self):
        if not self.filepaths:
            messagebox.showerror("No files", "Please select files first.")
            return
        self.set_status("Loading...")
        stack, dtype, omin, omax = read_images_as_stack(self.filepaths)
        self.processor.load_stack(stack)
        self.orig_dtype, self.orig_min, self.orig_max = dtype, omin, omax
        self.set_status("Computing PCA...")
        self.processor.compute_pca()
        self.lbl_info.config(text=f"H={self.processor.H}, W={self.processor.W}, B={self.processor.B}")
        self.update_thumbnails()
        self.show_large_pc(0)
        self.update_report("PCA computed successfully.")
        self.set_status("Done.")
        self._current_mode = "pc"

    def update_report(self, extra_text=""):
        rep = f"PCA Report\n===========\nBands: {self.processor.B}\n"
        rep += "Eigenvalues (descending):\n"
        for i, v in enumerate(self.processor.eigvals):
            rep += f"  PC{i+1}: {v:.6f}\n"
        if extra_text:
            rep += "\n" + extra_text
        self.txt_report.delete("1.0", tk.END)
        self.txt_report.insert(tk.END, rep)

    # ------------------------- PC DISPLAY -------------------------
    def update_thumbnails(self):
        for w in self.frm_thumbs.winfo_children():
            w.destroy()
        self._thumb_refs.clear()
        self.lbl_right_title.config(text="Principal Components (thumbnails)")

        cols = 2
        for i in range(self.processor.B):  # show all PCs
            pc = self.processor.get_pc_image(i)
            u8 = normalize_to_uint8(pc)
            pil = Image.fromarray(u8)
            pil.thumbnail((120, 80))
            img = ImageTk.PhotoImage(pil)
            b = tk.Button(self.frm_thumbs, image=img, command=lambda idx=i: self.show_large_pc(idx))
            b.grid(row=i // cols, column=i % cols, padx=4, pady=4)
            self._thumb_refs.append(img)

    def show_large_pc(self, idx):
        pc = self.processor.get_pc_image(idx)
        u8 = normalize_to_uint8(pc)
        pil = Image.fromarray(u8).convert("RGB")
        cw = self.canvas_big.winfo_width() or 800
        ch = self.canvas_big.winfo_height() or 420
        pil.thumbnail((cw - 10, ch - 10))
        self._big_ref = ImageTk.PhotoImage(pil)
        self.canvas_big.delete("all")
        self.canvas_big.create_image(cw // 2, ch // 2, image=self._big_ref)
        self.set_status(f"Showing PC {idx + 1}")

    # ------------------------- RECONSTRUCTION -------------------------
    def reconstruct_and_display(self):
        try:
            k = int(self.entry_k.get())
        except:
            messagebox.showerror("Invalid", "Enter integer k.")
            return
        self.set_status("Reconstructing...")
        rec = self.processor.reconstruct(k)
        mse = self.processor.compute_mse(rec)
        self.rec_bands = rec
        self._current_mode = "recon"
        self.display_reconstructed_mode()
        self.update_report(f"Reconstructed using k={k}\nMSE={mse:.6f}")
        self.set_status("Displayed reconstruction.")

    def display_reconstructed_mode(self):
        """Switch the right panel to show Orig/Rec thumbnails and enable slider"""
        for w in self.frm_thumbs.winfo_children():
            w.destroy()
        self._thumb_refs.clear()
        self.lbl_right_title.config(text="Reconstructed Bands (Orig vs Rec)")

        cols = 2
        H, W, B = self.processor.H, self.processor.W, self.processor.B
        for i in range(B):
            orig_u8 = normalize_to_uint8(self.processor.stack[:, :, i])
            rec_u8 = normalize_to_uint8(self.rec_bands[:, :, i])
            orig_pil = Image.fromarray(orig_u8).convert("RGB")
            rec_pil = Image.fromarray(rec_u8).convert("RGB")
            orig_pil.thumbnail((100, 70))
            rec_pil.thumbnail((100, 70))
            orig_img = ImageTk.PhotoImage(orig_pil)
            rec_img = ImageTk.PhotoImage(rec_pil)
            f = ttk.Frame(self.frm_thumbs)
            f.grid(row=i, column=0, padx=2, pady=2)
            tk.Label(f, image=orig_img).pack(side=tk.LEFT, padx=2)
            tk.Label(f, image=rec_img).pack(side=tk.LEFT, padx=2)
            tk.Label(f, text=f"Band {i+1}").pack(side=tk.LEFT, padx=4)
            self._thumb_refs.extend([orig_img, rec_img])

        # Setup slider to browse bands
        self.slider_band.config(from_=0, to=B - 1)
        self.slider_band.set(0)
        self.show_reconstructed_band(0)

    def _on_slider_move(self, event=None):
        if self._current_mode == "recon" and self.rec_bands is not None:
            idx = int(float(self.slider_band.get()))
            self.show_reconstructed_band(idx)

    def show_reconstructed_band(self, idx):
        """Display a single band comparison on the big canvas"""
        H, W, B = self.processor.H, self.processor.W, self.processor.B
        cw = self.canvas_big.winfo_width() or 800
        ch = self.canvas_big.winfo_height() or 420

        orig_u8 = normalize_to_uint8(self.processor.stack[:, :, idx])
        rec_u8 = normalize_to_uint8(self.rec_bands[:, :, idx])

        orig_pil = Image.fromarray(orig_u8).resize((cw // 2 - 20, ch - 40))
        rec_pil = Image.fromarray(rec_u8).resize((cw // 2 - 20, ch - 40))

        orig_img = ImageTk.PhotoImage(orig_pil)
        rec_img = ImageTk.PhotoImage(rec_pil)
        self._thumb_refs.extend([orig_img, rec_img])

        self.canvas_big.delete("all")
        self.canvas_big.create_image(cw // 4, ch // 2, image=orig_img)
        self.canvas_big.create_image(3 * cw // 4, ch // 2, image=rec_img)
        self.canvas_big.create_text(cw // 4, 20, text=f"Original Band {idx + 1}", fill="white")
        self.canvas_big.create_text(3 * cw // 4, 20, text=f"Reconstructed Band {idx + 1}", fill="white")
        self.slider_band_label.config(text=f"Band {idx + 1} / {B}")
        self.set_status(f"Showing reconstructed band {idx + 1}")
