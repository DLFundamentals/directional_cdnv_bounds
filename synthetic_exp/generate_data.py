import os
import math
import random
from typing import Tuple, Dict
from PIL import Image, ImageDraw
import numpy as np

# ----------------------------
# Config
# ----------------------------
IMG_SIZE = 64
SHAPE_COLOR = (255, 255, 255)

COLORS = {
    "red": (220, 50, 50),
    "green": (50, 180, 80),
    "blue": (60, 120, 220),
    "purple": (160, 70, 200),
    "dark_brown": (130, 90, 40),
    "yellow": (240, 230, 70),
}

SHAPES = ["circle", "triangle", "square", "pentagon"]
STYLES = ["plus", "minus", "dots", "cross"]

SIZE_THRESHOLD = 0.55  # heuristic cutoff for small vs big

# ----------------------------
# Style patterns
# ----------------------------
def apply_style(draw: ImageDraw.Draw, style: str, color: Tuple[int, int, int]):
    step = 8

    if style == "dots":
        for x in range(0, IMG_SIZE, step):
            for y in range(0, IMG_SIZE, step):
                r = 1
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    elif style == "minus":
        for y in range(0, IMG_SIZE, step):
            draw.line((0, y, IMG_SIZE, y), fill=color, width=1)

    elif style == "plus":
        for x in range(0, IMG_SIZE, step):
            draw.line((x, 0, x, IMG_SIZE), fill=color, width=1)
        for y in range(0, IMG_SIZE, step):
            draw.line((0, y, IMG_SIZE, y), fill=color, width=1)

    elif style == "cross":
        for i in range(-IMG_SIZE, IMG_SIZE, step):
            draw.line((i, 0, i + IMG_SIZE, IMG_SIZE), fill=color, width=1)
            draw.line((i, IMG_SIZE, i + IMG_SIZE, 0), fill=color, width=1)

# ----------------------------
# Shape drawing helpers
# ----------------------------
def regular_polygon(cx, cy, r, n):
    return [
        (
            cx + r * math.cos(2 * math.pi * k / n),
            cy + r * math.sin(2 * math.pi * k / n),
        )
        for k in range(n)
    ]

def draw_shape(draw: ImageDraw.Draw, shape: str, size_frac: float):
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    r = int(size_frac * IMG_SIZE / 2)

    if shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=SHAPE_COLOR)

    elif shape == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=SHAPE_COLOR)

    elif shape == "triangle":
        pts = regular_polygon(cx, cy, r, 3)
        draw.polygon(pts, fill=SHAPE_COLOR)

    elif shape == "pentagon":
        pts = regular_polygon(cx, cy, r, 5)
        draw.polygon(pts, fill=SHAPE_COLOR)

# ----------------------------
# Single image generator
# ----------------------------
def generate_image(
    color_name: str,
    shape: str,
    style: str,
    size_frac: float,
):
    bg_color = COLORS[color_name]

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg_color)
    draw = ImageDraw.Draw(img)

    # background style overlay (slightly darker)
    style_color = tuple(max(0, c - 40) for c in bg_color)
    apply_style(draw, style, style_color)

    # draw shape
    draw_shape(draw, shape, size_frac)

    return img

# ----------------------------
# Dataset generation
# ----------------------------
def generate_dataset(
    out_dir="synthetic_data",
    samples_per_combo=100,
    seed=0,
):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    metadata = []

    idx = 0
    for color in COLORS:
        for shape in SHAPES:
            for style in STYLES:
                for _ in range(samples_per_combo):
                    size_frac = random.uniform(0.35, 0.75)
                    size_label = "big" if size_frac > SIZE_THRESHOLD else "small"

                    img = generate_image(color, shape, style, size_frac)

                    fname = f"{idx:06d}.png"
                    img.save(os.path.join(out_dir, fname))

                    metadata.append({
                        "file": fname,
                        "color": color,
                        "shape": shape,
                        "style": style,
                        "size_label": size_label,
                        "size_frac": round(size_frac, 3),
                    })

                    idx += 1

    return metadata

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    meta = generate_dataset(
        out_dir="synthetic_shapes",
        samples_per_combo=100,
        seed=42,
    )

    print(f"Generated {len(meta)} images")
    print(meta[0])

    # Save metadata
    import json
    with open("synthetic_shapes/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    