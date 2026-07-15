"""
Import a user-uploaded cover image and install it as the song's thumbnail.

The existing GET /api/download/<job_id>/thumbnail endpoint serves
outputs/<job_id>/thumbnail.{jpg,png} — we write to that same path so the
user's upload becomes the served thumbnail automatically.
"""

from pathlib import Path
from PIL import Image, ImageOps

ALLOWED_COVER_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
MAX_COVER_BYTES = 5 * 1024 * 1024   # 5 MB
MAX_COVER_DIM = 800                  # px; covers don't need to be bigger


def validate_cover_image(path: str) -> dict:
    """Open the file with Pillow to confirm it's a real image and grab dims."""
    try:
        with Image.open(path) as im:
            im.verify()  # raises on malformed
    except Exception as e:
        return {"ok": False, "error": f"Not a valid image: {e}", "meta": None}
    try:
        with Image.open(path) as im:
            w, h = im.size
            fmt = im.format
    except Exception as e:
        return {"ok": False, "error": f"Could not read image dimensions: {e}", "meta": None}
    if w < 64 or h < 64:
        return {"ok": False, "error": f"Image too small ({w}x{h}); minimum 64x64.", "meta": None}
    return {"ok": True, "error": None, "meta": {"width": w, "height": h, "format": fmt}}


def install_cover(src_path: str, job_dir: Path) -> str:
    """Resize to fit within MAX_COVER_DIM (preserves aspect ratio), strip metadata,
    save as PNG at outputs/<job_id>/thumbnail.png. Removes any stale .jpg variant.
    Returns the on-disk path written."""
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / "thumbnail.png"
    stale_jpg = job_dir / "thumbnail.jpg"

    with Image.open(src_path) as im:
        # Honor EXIF rotation, then convert to RGB for consistent output
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        # Resize preserving aspect ratio
        im.thumbnail((MAX_COVER_DIM, MAX_COVER_DIM), Image.LANCZOS)
        # Save as PNG (lossless, supports transparency)
        im.save(dest, format="PNG", optimize=True)

    # Remove stale .jpg so the download endpoint's preference order
    # (.jpg first, .png fallback) doesn't serve an old image.
    if stale_jpg.exists():
        try:
            stale_jpg.unlink()
        except Exception:
            pass

    return str(dest)
