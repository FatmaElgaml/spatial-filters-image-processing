"""Spatial Filters – Experiment 1 (noise + smoothing) and Experiment 2 (edges + sharpening)."""
import math
import os
import tkinter as tk
from collections import OrderedDict
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import filters as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")


class ExperimentTab(ttk.Frame):
    """Control panel (left) + single grid plot (right)."""
    default_file = ""

    def __init__(self, master):
        super().__init__(master)
        self.rgb = None
        self.base = None
        self.results = OrderedDict()      # label -> (image, title)
        self.mode = tk.StringVar(value="Grayscale")
        self._build_layout()
        self._try_load_default()

    def _build_layout(self):
        self.panel = ttk.Frame(self, padding=10)
        self.panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(self.panel, text="📂 Load image", command=self.load_image).pack(fill=tk.X)
        self.file_lbl = ttk.Label(self.panel, text="(no image)", wraplength=200)
        self.file_lbl.pack(fill=tk.X, pady=(2, 0))
        self.type_lbl = ttk.Label(self.panel, text="", wraplength=200, foreground="#1a5fb4")
        self.type_lbl.pack(fill=tk.X, pady=(0, 6))

        box = ttk.LabelFrame(self.panel, text="Work in", padding=6)
        box.pack(fill=tk.X, pady=4)
        self.rb_gray = ttk.Radiobutton(box, text="Grayscale", value="Grayscale",
                                       variable=self.mode, command=self.on_mode_change)
        self.rb_rgb = ttk.Radiobutton(box, text="RGB", value="RGB",
                                      variable=self.mode, command=self.on_mode_change)
        self.rb_gray.pack(anchor=tk.W)
        self.rb_rgb.pack(anchor=tk.W)

        self.build_controls()

        ttk.Separator(self.panel).pack(fill=tk.X, pady=8)
        ttk.Button(self.panel, text="Reset", command=self.reset).pack(fill=tk.X, pady=2)
        ttk.Button(self.panel, text="💾 Save figure", command=self.save_figure).pack(fill=tk.X, pady=2)

        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        NavigationToolbar2Tk(self.canvas, right).update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def build_controls(self):
        raise NotImplementedError

    # ---------- image ----------
    def _try_load_default(self):
        for folder in (IMG_DIR, BASE_DIR):
            p = os.path.join(folder, self.default_file)
            if os.path.exists(p):
                self._set_image(p)
                return

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if path:
            self._set_image(path)

    def _set_image(self, path):
        try:
            self.rgb = F.read_image_rgb(path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.file_lbl.config(text=os.path.basename(path))
        if F.is_grayscale_image(self.rgb):          # black & white image -> gray only
            self.mode.set("Grayscale")
            self.rb_rgb.state(["disabled"])
            self.type_lbl.config(text="Black & white image → Grayscale")
        else:                                       # colour image -> user chooses
            self.rb_rgb.state(["!disabled"])
            self.mode.set("RGB")
            self.type_lbl.config(text="Colour image → choose Grayscale or RGB")
        self.on_mode_change()

    def on_mode_change(self):
        if self.rgb is None:
            return
        self.base = F.to_mode(self.rgb, self.mode.get())
        self.reset()

    def require_image(self):
        if self.base is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return False
        return True

    # ---------- state / drawing ----------
    def reset(self):
        self.clear_state()
        self.results.clear()
        if self.base is not None:
            self.results["Original"] = (self.base, "Original")
        self.refresh()

    def clear_state(self):
        pass

    def refresh(self):
        self.fig.clf()
        n = len(self.results)
        if n:
            cols = min(4, n)
            rows = math.ceil(n / cols)
            for i, (img, title) in enumerate(self.results.values()):
                ax = self.fig.add_subplot(rows, cols, i + 1)
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                else:
                    ax.imshow(img)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            self.fig.tight_layout()
        self.canvas.draw_idle()

    def save_figure(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if path:
            self.fig.savefig(path, dpi=200, bbox_inches="tight")

    def add_button(self, text, command, parent=None):
        ttk.Button(parent or self.panel, text=text, command=command).pack(fill=tk.X, pady=2)


# =================================== Experiment 1 ===================================
class Exp1Tab(ExperimentTab):
    default_file = "veggies.png"
    FILTERS = (("Average", F.average_filter), ("Median", F.median_filter),
               ("Gaussian", F.gaussian_filter))

    def __init__(self, master):
        self.noisy = None
        super().__init__(master)

    def clear_state(self):
        self.noisy = None

    def build_controls(self):
        row = ttk.Frame(self.panel)
        row.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(row, text="Noise σ:").pack(side=tk.LEFT)
        self.sigma = tk.DoubleVar(value=25)
        ttk.Spinbox(row, from_=1, to=100, textvariable=self.sigma, width=6).pack(side=tk.LEFT, padx=4)
        self.add_button("Gaussian Noise", self.apply_noise)

        for name, fn in self.FILTERS:
            box = ttk.LabelFrame(self.panel, text=name, padding=4)
            box.pack(fill=tk.X, pady=3)
            for k in (3, 5):
                self.add_button(f"{name} {k}x{k}",
                                lambda n=name, f=fn, k=k: self.apply_filter(n, f, k), box)
        self.add_button("▶ Show ALL", self.show_all)

    def apply_noise(self):
        if not self.require_image():
            return
        self.results.clear()
        self.results["Original"] = (self.base, "Original")
        self.noisy = F.add_gaussian_noise(self.base, 0, float(self.sigma.get()))
        self.results["Noisy"] = (self.noisy, f"Gaussian noise (σ={self.sigma.get():g})")
        self.refresh()

    def apply_filter(self, name, fn, k):
        if not self.require_image():
            return
        if self.noisy is None:
            self.apply_noise()
        label = f"{name} {k}x{k}"
        self.results[label] = (fn(self.noisy, k), label)
        self.refresh()

    def show_all(self):
        if not self.require_image():
            return
        self.apply_noise()
        for name, fn in self.FILTERS:
            for k in (3, 5):
                self.apply_filter(name, fn, k)


# =================================== Experiment 2 ===================================
class Exp2Tab(ExperimentTab):
    default_file = "sunflower.png"

    def __init__(self, master):
        self.noisy = None
        self.noisy_title = ""
        super().__init__(master)

    def clear_state(self):
        self.noisy = None

    def build_controls(self):
        row = ttk.Frame(self.panel)
        row.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(row, text="Noise σ:").pack(side=tk.LEFT)
        self.sigma = tk.DoubleVar(value=25)
        ttk.Spinbox(row, from_=1, to=100, textvariable=self.sigma, width=6).pack(side=tk.LEFT, padx=4)
        self.add_button("Gaussian Noise", self.apply_noise)

        ttk.Label(self.panel, text="Edge detection + sharpening",
                  font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(10, 2))
        for name in F.EDGE_FUNCS:
            self.add_button(name, lambda n=name: self.apply_edge(n))
        self.add_button("▶ Show ALL", self.show_all)

    def apply_noise(self):
        if not self.require_image():
            return
        self.results.clear()
        self.results["Original"] = (self.base, "Original")
        self.noisy = F.add_gaussian_noise(self.base, 0, float(self.sigma.get()))
        self.noisy_title = f"Gaussian noise (σ={self.sigma.get():g})"
        self.results["Noisy"] = (self.noisy, self.noisy_title)
        self.refresh()

    def apply_edge(self, name):
        if not self.require_image():
            return
        src = self.noisy if self.noisy is not None else self.base   # الصورة اللي هنشتغل عليها
        edge = F.EDGE_FUNCS[name](src)
        self.results[f"{name} edges"] = (F.edge_to_display(edge), f"{name} edges")
        self.results[f"{name} sharpened"] = (F.sharpen_with_edges(src, edge, name),
                                             f"Sharpened using {name}")
        self.refresh()

    def show_all(self):
        if not self.require_image():
            return
        self.results.clear()
        self.results["Original"] = (self.base, "Original")
        if self.noisy is not None:
            self.results["Noisy"] = (self.noisy, self.noisy_title)
        for name in F.EDGE_FUNCS:
            self.apply_edge(name)
def main():
    root = tk.Tk()
    root.title("Spatial Filters")
    root.geometry("1200x780")
    nb = ttk.Notebook(root)
    nb.pack(fill=tk.BOTH, expand=True)
    nb.add(Exp1Tab(nb), text="Experiment 1 – Noise & Smoothing")
    nb.add(Exp2Tab(nb), text="Experiment 2 – Edges & Sharpening")
    root.mainloop()


if __name__ == "__main__":
    main()
