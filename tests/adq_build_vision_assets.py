#!/usr/bin/env python3
"""Build deterministic raster cards for ADQ multimodal qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", optimize=False, compress_level=9)


def main() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run through ./dev adq-vision-assets; direct execution is unsupported.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--font", required=True, type=Path)
    args = parser.parse_args()
    if not args.font.is_file():
        raise SystemExit(f"font does not exist: {args.font}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    semantic = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(semantic)
    draw.polygon([(120, 600), (350, 140), (580, 600)], fill="#1565C0")
    draw.ellipse((720, 180, 1100, 560), fill="#7B1FA2")
    semantic_path = args.output_dir / "semantic_shapes.png"
    _save(semantic, semantic_path)

    spatial = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(spatial)
    draw.ellipse((70, 60, 270, 260), fill="#D32F2F")
    draw.rectangle((500, 300, 700, 500), fill="#388E3C")
    draw.polygon([(930, 730), (1090, 730), (1010, 540)], fill="#1976D2")
    spatial_path = args.output_dir / "spatial_shapes.png"
    _save(spatial, spatial_path)

    large_font = ImageFont.truetype(str(args.font), 128)
    medium_font = ImageFont.truetype(str(args.font), 86)
    ocr = Image.new("RGB", (1800, 1000), "white")
    draw = ImageDraw.Draw(ocr)
    draw.rectangle((45, 45, 1755, 955), outline="black", width=12)
    draw.text((125, 230), "ADQ VISION CODE", font=large_font, fill="black")
    draw.text((530, 520), "KX-7319", font=large_font, fill="#003366")
    ocr_path = args.output_dir / "ocr_code.png"
    _save(ocr, ocr_path)

    cards = []
    for label, region, number, background, filename in (
        ("CARD A", "NORTH", "4182", "#FFF59D", "multi_card_a.png"),
        ("CARD B", "SOUTH", "9367", "#FFCC80", "multi_card_b.png"),
    ):
        card = Image.new("RGB", (1200, 800), background)
        draw = ImageDraw.Draw(card)
        draw.rectangle((35, 35, 1165, 765), outline="black", width=10)
        draw.text((350, 105), label, font=large_font, fill="black")
        draw.text((410, 330), region, font=medium_font, fill="black")
        draw.text((445, 520), number, font=large_font, fill="#4A148C")
        path = args.output_dir / filename
        _save(card, path)
        cards.append(path)

    outputs = [semantic_path, spatial_path, ocr_path, *cards]
    manifest = {
        "format": "krasis_adq_vision_assets",
        "format_version": 1,
        "font": {"path": str(args.font.resolve()), "sha256": _sha256(args.font)},
        "assets": [
            {"path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in outputs
        ],
        "contracts": {
            "semantic_shapes.png": ["blue triangle", "purple circle"],
            "spatial_shapes.png": ["red circle top-left", "green square centre", "blue triangle bottom-right"],
            "ocr_code.png": ["ADQ VISION CODE", "KX-7319"],
            "multi_card_a.png+multi_card_b.png": ["CARD A NORTH 4182", "CARD B SOUTH 9367"],
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    for row in manifest["assets"]:
        print(f"{row['sha256']}  {row['path']}")


if __name__ == "__main__":
    main()
