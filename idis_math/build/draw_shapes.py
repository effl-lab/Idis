"""Geometric shapes on a white canvas (PIL for 2D, matplotlib for 3D).

  coarse label → pool of specific shapes (PLANE_POOLS, SOLID_POOLS)
"""

import io
import math
import random

from PIL import Image, ImageDraw

# ── Coarse → specific shape pools ──────────────────────────────────

PLANE_POOLS = {
    "triangle": ["triangle", "right_triangle"],
    "quadrilateral": ["square", "rectangle", "parallelogram", "trapezoid"],
    "circle_family": ["circle", "ellipse"],
    "line_angle_configuration": ["line_segment", "angle"],
}

SOLID_POOLS = {
    "prism_family": ["cube", "rectangular_prism"],
    "pyramid_family": ["pyramid"],
    "cylinder_family": ["cylinder"],
    "cone_family": ["cone"],
    "sphere_family": ["sphere", "hemisphere"],
}

# ── PIL 2D drawing ─────────────────────────────────────────────────

LINE = 4
PAD = 0.12


def _box(size):
    p = int(size * PAD)
    return p, p, size - p, size - p


def _draw_2d(shape: str, size: int) -> Image.Image:
    """Draw a 2D shape on a white square canvas using PIL."""
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = _box(size)
    cx, cy = size // 2, size // 2
    w, h = x1 - x0, y1 - y0

    if shape == "triangle":
        d.polygon([(cx, y0), (x0, y1), (x1, y1)], outline="black", width=LINE)
    elif shape == "right_triangle":
        d.polygon([(x0, y0), (x0, y1), (x1, y1)], outline="black", width=LINE)
    elif shape == "square":
        s = min(w, h)
        d.rectangle([cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2],
                    outline="black", width=LINE)
    elif shape == "rectangle":
        d.rectangle([x0, cy - h // 3, x1, cy + h // 3], outline="black", width=LINE)
    elif shape == "circle":
        r = min(w, h) // 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline="black", width=LINE)
    elif shape == "ellipse":
        d.ellipse([x0, cy - h // 3, x1, cy + h // 3], outline="black", width=LINE)
    elif shape == "parallelogram":
        off = w // 5
        d.polygon([(x0 + off, y0), (x1, y0), (x1 - off, y1), (x0, y1)],
                  outline="black", width=LINE)
    elif shape == "trapezoid":
        off = w // 5
        d.polygon([(x0 + off, y0), (x1 - off, y0), (x1, y1), (x0, y1)],
                  outline="black", width=LINE)
    elif shape == "line_segment":
        d.line([(x0, cy), (x1, cy)], fill="black", width=LINE)
    elif shape == "angle":
        d.line([(x0, y1), (x1, y1)], fill="black", width=LINE)
        d.line([(x0, y1), (x1, y0)], fill="black", width=LINE)
    elif shape == "pentagon":
        n = 5
        r = min(w, h) // 2
        pts = []
        for i in range(n):
            a = -math.pi / 2 + 2 * math.pi * i / n
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        d.polygon(pts, outline="black", width=LINE)
    elif shape == "hexagon":
        n = 6
        r = min(w, h) // 2
        pts = []
        for i in range(n):
            a = -math.pi / 2 + 2 * math.pi * i / n
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        d.polygon(pts, outline="black", width=LINE)
    else:
        # fallback: blank framed canvas
        d.rectangle([x0, y0, x1, y1], outline="black", width=LINE)

    return img


# ── Matplotlib 3D drawing ──────────────────────────────────────────

def _draw_3d(shape: str, size: int) -> Image.Image:
    """Draw a 3D shape using matplotlib and return as PIL Image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    fig = plt.figure(figsize=(4, 4), dpi=size // 4)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    if shape == "cube":
        _draw_cube(ax)
    elif shape == "rectangular_prism":
        _draw_rectangular_prism(ax)
    elif shape == "pyramid":
        _draw_pyramid(ax)
    elif shape == "cylinder":
        _draw_cylinder(ax, np)
    elif shape == "cone":
        _draw_cone(ax, np)
    elif shape == "sphere":
        _draw_sphere(ax, np)
    elif shape == "hemisphere":
        _draw_hemisphere(ax, np)
    else:
        _draw_cube(ax)  # fallback

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.view_init(elev=25, azim=45)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    # Resize to square canvas
    img = img.resize((size, size), Image.LANCZOS)
    return img


def _draw_cube(ax):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    v = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
    faces = [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
             [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
             [v[0], v[3], v[7], v[4]], [v[1], v[2], v[6], v[5]]]
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.1, facecolor="white",
                                         edgecolor="black", linewidth=1.5))


def _draw_rectangular_prism(ax):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    sx, sy, sz = 1.0, 0.6, 0.8
    v = [[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
         [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]]
    faces = [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
             [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
             [v[0], v[3], v[7], v[4]], [v[1], v[2], v[6], v[5]]]
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.1, facecolor="white",
                                         edgecolor="black", linewidth=1.5))


def _draw_pyramid(ax):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    base = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]]
    apex = [0, 0, 1]
    faces = [base]
    for i in range(4):
        faces.append([base[i], base[(i + 1) % 4], apex])
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.1, facecolor="white",
                                         edgecolor="black", linewidth=1.5))


def _draw_cylinder(ax, np):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    theta = np.linspace(0, 2 * np.pi, 40)
    z_bottom, z_top = -1.0, 1.0
    r = 0.8
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Side surface
    ax.plot_surface(
        np.array([x, x]),
        np.array([y, y]),
        np.array([[z_bottom] * len(theta), [z_top] * len(theta)]),
        alpha=0.1, color="white", edgecolor="black", linewidth=0.3
    )
    # Top and bottom circles
    ax.plot(x, y, z_bottom, color="black", linewidth=1.5)
    ax.plot(x, y, z_top, color="black", linewidth=1.5)


def _draw_cone(ax, np):
    theta = np.linspace(0, 2 * np.pi, 40)
    z = np.linspace(-1, 1, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    r = 0.8 * (1 - (z_grid + 1) / 2)  # radius decreases from bottom to top
    x = r * np.cos(theta_grid)
    y = r * np.sin(theta_grid)
    ax.plot_surface(x, y, z_grid, alpha=0.1, color="white",
                    edgecolor="black", linewidth=0.3)
    # Base circle
    ax.plot(0.8 * np.cos(theta), 0.8 * np.sin(theta), -1, color="black", linewidth=1.5)


def _draw_sphere(ax, np):
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    r = 0.9
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color="white",
                    edgecolor="black", linewidth=0.3)


def _draw_hemisphere(ax, np):
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi / 2, 15)
    r = 0.9
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color="white",
                    edgecolor="black", linewidth=0.3)
    # Base circle
    theta = np.linspace(0, 2 * np.pi, 40)
    ax.plot(r * np.cos(theta), r * np.sin(theta), 0, color="black", linewidth=1.5)


# ── Public API ─────────────────────────────────────────────────────

# All known pools
ALL_POOLS = {**PLANE_POOLS, **SOLID_POOLS}

# Which coarse labels use 3D rendering
SOLID_LABELS = set(SOLID_POOLS.keys())


def draw_coarse(coarse_label: str, size: int, rng: random.Random = None) -> Image.Image:
    """Draw a shape from the coarse label pool.

    Args:
        coarse_label: One of the coarse labels (e.g., "triangle", "prism_family")
        size: Canvas size (square, in pixels)
        rng: Random instance for reproducibility

    Returns:
        PIL Image with the drawn shape on white background
    """
    if rng is None:
        rng = random.Random()

    pool = ALL_POOLS.get(coarse_label)
    if pool is None:
        # Unknown label -> blank canvas
        return _draw_2d("other", size)

    specific = rng.choice(pool)

    if coarse_label in SOLID_LABELS:
        return _draw_3d(specific, size)
    else:
        return _draw_2d(specific, size)


def get_specific_shape(coarse_label: str, rng: random.Random = None) -> str:
    """Return a random specific shape name from the coarse pool."""
    if rng is None:
        rng = random.Random()
    pool = ALL_POOLS.get(coarse_label, ["other"])
    return rng.choice(pool)
