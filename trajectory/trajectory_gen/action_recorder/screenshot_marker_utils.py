from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw


DEFAULT_REGION_WIDTH = 200
DEFAULT_REGION_HEIGHT = 200
DEFAULT_HIGHLIGHT_RADIUS = 20
DEFAULT_HIGHLIGHT_WIDTH = 4


def build_click_marked_screenshot(
    screenshot: PILImage.Image,
    position: Tuple[int, int],
    region_width: int = DEFAULT_REGION_WIDTH,
    region_height: int = DEFAULT_REGION_HEIGHT,
    highlight_radius: int = DEFAULT_HIGHLIGHT_RADIUS,
    highlight_width: int = DEFAULT_HIGHLIGHT_WIDTH,
) -> tuple[PILImage.Image, PILImage.Image]:
    x, y = position
    width, height = screenshot.size

    half_w = region_width // 2
    half_h = region_height // 2
    left = max(x - half_w, 0)
    top = max(y - half_h, 0)
    right = min(left + region_width, width)
    bottom = min(top + region_height, height)

    region = screenshot.crop((left, top, right, bottom))
    marked = screenshot.copy()

    draw = PILImageDraw.Draw(marked)
    draw.ellipse(
        [x - highlight_radius, y - highlight_radius, x + highlight_radius, y + highlight_radius],
        outline="red",
        width=highlight_width,
    )
    line_length = highlight_radius * 2
    draw.line([x - line_length, y, x + line_length, y], fill="red", width=highlight_width)
    draw.line([x, y - line_length, x, y + line_length], fill="red", width=highlight_width)

    return marked, region


def save_click_marked_artifacts(
    input_image_path: str | Path,
    marked_output_path: str | Path,
    part_output_path: str | Path,
    position: Tuple[int, int],
    region_width: int = DEFAULT_REGION_WIDTH,
    region_height: int = DEFAULT_REGION_HEIGHT,
    highlight_radius: int = DEFAULT_HIGHLIGHT_RADIUS,
    highlight_width: int = DEFAULT_HIGHLIGHT_WIDTH,
) -> tuple[Path, Path]:
    input_path = Path(input_image_path)
    marked_path = Path(marked_output_path)
    part_path = Path(part_output_path)

    with PILImage.open(input_path) as image:
        source = image.convert("RGB") if image.mode != "RGB" else image.copy()

    marked, part = build_click_marked_screenshot(
        screenshot=source,
        position=position,
        region_width=region_width,
        region_height=region_height,
        highlight_radius=highlight_radius,
        highlight_width=highlight_width,
    )

    marked_path.parent.mkdir(parents=True, exist_ok=True)
    part_path.parent.mkdir(parents=True, exist_ok=True)
    marked.save(marked_path)
    part.save(part_path)
    return marked_path, part_path
