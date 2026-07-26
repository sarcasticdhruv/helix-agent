"""Generate the HELIX ASCII-art banner and social preview image.

Usage:
    python3.11 scripts/growth/generate_banner.py
"""
import pyfiglet
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
BG_COLOR = (13, 17, 23)      # GitHub dark background
FG_COLOR = (230, 237, 243)   # near-white


def render_banner(
    text: str,
    tagline: str,
    out_path: str,
    font_size: int = 22,
    canvas_size: tuple[int, int] | None = None,
) -> None:
    art_lines = pyfiglet.figlet_format(text, font="block").rstrip("\n").split("\n")
    font = ImageFont.truetype(FONT_PATH, font_size)
    tagline_font = ImageFont.truetype(FONT_PATH, font_size // 2)

    line_width = max(font.getbbox(line)[2] for line in art_lines)
    line_height = font.getbbox("Xg")[3] + 6
    padding = font_size * 2

    content_width = line_width + padding * 2
    content_height = line_height * len(art_lines) + padding * 3 + font_size

    width, height = canvas_size if canvas_size else (content_width, content_height)
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    x_offset = (width - line_width) // 2
    y = (height - content_height) // 2 + padding if canvas_size else padding
    for line in art_lines:
        draw.text((x_offset, y), line, font=font, fill=FG_COLOR)
        y += line_height

    tagline_bbox = draw.textbbox((0, 0), tagline, font=tagline_font)
    tagline_width = tagline_bbox[2] - tagline_bbox[0]
    draw.text(((width - tagline_width) // 2, y + padding // 2), tagline, font=tagline_font, fill=FG_COLOR)

    img.save(out_path)


if __name__ == "__main__":
    render_banner(
        "HELIX",
        "Production AI agents: budget limits, semantic caching, multi-agent teams",
        "assets/banner.png",
    )
