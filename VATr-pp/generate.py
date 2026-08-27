import argparse
import os
from pathlib import Path
import unicodedata

import numpy as np
from PIL import Image
import torch

from models.model import VATr
from util.misc import add_vatr_args, VATRPP_BASE_ALPHABET, VATRPP_BASE_SPECIAL, SPANISH_EXTRA


def _load_style_folder(style_folder: str, height: int, max_width: int, num_examples: int):
    style_path = Path(style_folder)
    if not style_path.exists():
        raise FileNotFoundError(f"Style folder not found: {style_folder}")

    files = sorted([p for p in style_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not files:
        raise RuntimeError(f"No images found in style folder: {style_folder}")

    files = files[:num_examples]
    tensors = []
    widths = []

    for p in files:
        img = Image.open(p).convert("L")
        w, h = img.size
        if h <= 0:
            continue
        new_w = max(1, int(round(w * (height / float(h)))))
        img = img.resize((new_w, height), resample=Image.BICUBIC)

        arr = np.array(img, dtype=np.float32)
        widths.append(min(arr.shape[1], max_width))
        if arr.shape[1] < max_width:
            pad = np.ones((height, max_width), dtype=np.float32) * 255.0
            pad[:, : arr.shape[1]] = arr
            arr = pad
        else:
            arr = arr[:, :max_width]

        # ToTensor + Normalize((0.5,), (0.5,))
        t = torch.from_numpy(arr / 255.0).unsqueeze(0)  # [1,H,W]
        t = (t - 0.5) / 0.5
        tensors.append(t)

    if not tensors:
        raise RuntimeError(f"Failed to load any valid style images from: {style_folder}")

    style = torch.stack(tensors, dim=0)  # [N,1,H,W]
    style = style.squeeze(1)  # [N,H,W]
    style = style.unsqueeze(0)  # [1,N,H,W]
    return style, widths


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def _load_input_texts(args):
    if args.text_path:
        text_path = Path(args.text_path)
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found: {text_path}")
        texts = []
        with text_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = _normalize_text(line)
                if text:
                    texts.append(text)
        if not texts:
            raise RuntimeError(f"No non-empty lines found in: {text_path}")
        return texts

    if args.text is None:
        raise ValueError("Provide either --text or --text-path")
    return [_normalize_text(args.text)]


def _build_output_path(output: str, text: str, index: int, multi_output: bool) -> Path:
    base_output = Path(output)
    if multi_output:
        base_output.mkdir(parents=True, exist_ok=True)
        return base_output / f"{index + 1:04d}.png"

    base_output.parent.mkdir(parents=True, exist_ok=True)
    return base_output


@torch.no_grad()
def _render_single_text(model, device, style, args, text: str, out_path: Path):
    text_encode, text_len, _ = model.netconverter.encode([text.encode("utf-8")])
    text_encode = text_encode.to(device)

    with model._autocast():
        fake, _ = model.netG(style, text_encode)

    img = fake[0, 0].detach().float().cpu().numpy()
    crop_w = int(text_len[0].item()) * int(args.resolution)
    if crop_w > 0 and crop_w < img.shape[1]:
        img = img[:, :crop_w]
    img = (img + 1.0) * 0.5
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)


@torch.no_grad()
def generate_text_image(args):
    device = torch.device(args.device)

    model = VATr(args).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    style, _ = _load_style_folder(args.style_folder, args.img_height, args.max_width, args.num_examples)
    style = style.to(device)

    texts = _load_input_texts(args)
    multi_output = len(texts) > 1 or bool(args.text_path)
    manifest_lines = []

    for index, text in enumerate(texts):
        out_path = _build_output_path(args.output, text, index, multi_output)
        _render_single_text(model, device, style, args, text, out_path)
        if multi_output:
            manifest_lines.append(f"{out_path.name} {text}")
        print(f"Saved: {out_path}")

    if multi_output:
        manifest_path = Path(args.output) / "labels.txt"
        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        print(f"Saved manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["text"])

    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--style-folder", default="files/style_samples/00", type=str)
    parser.add_argument("--text", default=None, type=str)
    parser.add_argument("--text-path", default=None, type=str, help="UTF-8 text file, one sample per line")
    parser.add_argument("--output", required=True, type=str)

    parser = add_vatr_args(parser)
    args = parser.parse_args()

    # Inference-only: avoid creating D/W/OCR and their optimizers (smaller, loads with strict=False).
    args.infer_only = True
    if not hasattr(args, "max_width"):
        args.max_width = 192
    if args.preset == "vatrpp":
        args.alphabet = VATRPP_BASE_ALPHABET
        args.special_alphabet = VATRPP_BASE_SPECIAL + ''.join(c for c in SPANISH_EXTRA if c not in VATRPP_BASE_ALPHABET)

    if args.action == "text":
        generate_text_image(args)


if __name__ == "__main__":
    main()
