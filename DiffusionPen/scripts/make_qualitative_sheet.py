import argparse
import math
import os

from PIL import Image, ImageDraw, ImageFont


def read_manifest(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            rows.append((parts[0], parts[1]))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Create a simple qualitative sheet from image-text pairs.")
    parser.add_argument("--manifest", type=str, required=True, help="TSV or split-style txt with image_path label")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--cols", type=int, default=3)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)[: args.limit]
    if not rows:
        raise ValueError("Manifest is empty")

    font = ImageFont.load_default()
    tile_w, tile_h = 320, 120
    caption_h = 28
    cols = max(1, args.cols)
    rows_n = math.ceil(len(rows) / cols)
    canvas = Image.new("RGB", (cols * tile_w, rows_n * (tile_h + caption_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (img_path, text) in enumerate(rows):
        r = idx // cols
        c = idx % cols
        x0 = c * tile_w
        y0 = r * (tile_h + caption_h)

        img = Image.open(img_path).convert("RGB")
        img.thumbnail((tile_w - 16, tile_h - 16))
        px = x0 + (tile_w - img.width) // 2
        py = y0 + (tile_h - img.height) // 2
        canvas.paste(img, (px, py))
        draw.rectangle([x0, y0, x0 + tile_w - 1, y0 + tile_h - 1], outline="black", width=1)
        draw.text((x0 + 8, y0 + tile_h + 6), text, fill="black", font=font)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
