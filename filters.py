"""Image-processing functions for the Spatial Filters assignment (no GUI code here)."""
import cv2
import numpy as np


# ----------------------------- I/O helpers -----------------------------
def read_image_rgb(path):
    """Read an image (works with non-ASCII paths) and return it as RGB uint8."""
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def is_grayscale_image(rgb, tol=3):
    """True if the image is black & white / gray (R ≈ G ≈ B everywhere)."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return int(max(np.abs(r - g).max(), np.abs(g - b).max())) <= tol


def to_mode(rgb, mode):
    """mode = 'Grayscale' or 'RGB'."""
    if mode == "Grayscale":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb


# ------------------------ Experiment 1: noise + smoothing ------------------------
def add_gaussian_noise(img, mean=0.0, sigma=25.0):
    noise = np.random.normal(mean, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def average_filter(img, k):
    return cv2.blur(img, (k, k))


def median_filter(img, k):
    return cv2.medianBlur(img, k)


def gaussian_filter(img, k):
    return cv2.GaussianBlur(img, (k, k), 0)


# ------------------------ Experiment 2: edges + sharpening ------------------------
PREWITT_X = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)
PREWITT_Y = PREWITT_X.T

# Sobel masks 
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]], dtype=np.float32)

# Laplacian mask (8-neighbour) 
LAPLACIAN = np.array([[1, 1, 1],
                      [1, -8, 1],
                      [1, 1, 1]], dtype=np.float32)


def _conv(img, kernel):
    return cv2.filter2D(img.astype(np.float32), -1, kernel)


def prewitt_edges(img):
    return np.sqrt(_conv(img, PREWITT_X) ** 2 + _conv(img, PREWITT_Y) ** 2)


def sobel_edges(img):
    return np.sqrt(_conv(img, SOBEL_X) ** 2 + _conv(img, SOBEL_Y) ** 2)


def laplacian_edges(img):
    return _conv(img, LAPLACIAN)        # signed response


EDGE_FUNCS = {
    "Prewitt": prewitt_edges,
    "Sobel": sobel_edges,
    "Laplacian": laplacian_edges,
}


def edge_to_display(edge):
    """Scale an edge response to 0..255 uint8 for showing."""
    mag = np.abs(edge)
    mx = mag.max()
    if mx == 0:
        return np.zeros_like(mag, dtype=np.uint8)
    return np.clip(mag / mx * 255.0, 0, 255).astype(np.uint8)


def sharpen_with_edges(img, edge, kind, weight=0.5):
    """
    Laplacian        : g = f - Laplacian(f)   (centre coefficient is negative)
    Prewitt / Sobel  : g = f + weight * normalised gradient magnitude
    """
    f = img.astype(np.float32)
    if kind == "Laplacian":
        out = f - edge
    else:
        out = f + weight * edge_to_display(edge).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
