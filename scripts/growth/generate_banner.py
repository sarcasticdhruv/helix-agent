"""Generate the HELIX block-art banner and social preview image.

Usage:
    python3.11 scripts/growth/generate_banner.py
"""
import pyfiglet
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
FIGLET_FONT = "ansi_regular"  # solid block-character font, no outline/shadow glyphs to misalign
BG_COLOR = (13, 17, 23)      # GitHub dark background
FG_COLOR = (230, 237, 243)   # near-white


def _fit_tagline_font(
    draw: ImageDraw.ImageDraw, tagline: str, max_width: int, start_size: int, min_size: int = 16
) -> ImageFont.FreeTypeFont:
    size = start_size
    while size > min_size:
        font = ImageFont.truetype(FONT_PATH, size)
        bbox = draw.textbbox((0, 0), tagline, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 2
    return ImageFont.truetype(FONT_PATH, min_size)


def render_banner(
    text: str,
    tagline: str | None,
    out_path: str,
    font_size: int = 28,
    canvas_size: tuple[int, int] | None = None,
) -> None:
    art = pyfiglet.figlet_format(text, font=FIGLET_FONT)
    lines = art.split("\n")
    while lines and lines[-1].strip() == "":
        lines.pop()
    art_text = "\n".join(lines)

    font = ImageFont.truetype(FONT_PATH, font_size)
    probe_img = Image.new("RGB", (10, 10))
    probe_draw = ImageDraw.Draw(probe_img)
    bbox = probe_draw.multiline_textbbox((0, 0), art_text, font=font, spacing=0)
    art_width = bbox[2] - bbox[0]
    art_height = bbox[3] - bbox[1]
    padding = font_size * 2

    if canvas_size:
        width, height = canvas_size
    else:
        width, height = art_width + padding * 2, art_height + padding * 2

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    tagline_font = None
    tagline_height = 0
    gap = font_size
    if tagline:
        tagline_font = _fit_tagline_font(draw, tagline, width - padding * 2, font_size)
        tb = draw.textbbox((0, 0), tagline, font=tagline_font)
        tagline_height = tb[3] - tb[1]

    content_height = art_height + (gap + tagline_height if tagline else 0)
    top = (height - content_height) // 2 - bbox[1]
    draw.multiline_text(
        ((width - art_width) // 2 - bbox[0], top), art_text, font=font, fill=FG_COLOR, spacing=0
    )

    if tagline and tagline_font:
        tb = draw.textbbox((0, 0), tagline, font=tagline_font)
        tw = tb[2] - tb[0]
        draw.text(((width - tw) // 2, top + art_height + gap), tagline, font=tagline_font, fill=FG_COLOR)

    img.save(out_path)


if __name__ == "__main__":
    render_banner("HELIX", None, "assets/banner.png")
    render_banner(
        "HELIX",
        "Production AI agents: budget limits, semantic caching, multi-agent teams",
        "assets/social-preview.png",
        font_size=48,
        canvas_size=(1280, 640),
    )
