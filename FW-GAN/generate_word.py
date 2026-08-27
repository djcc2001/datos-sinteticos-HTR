#!/usr/bin/env python3
"""
Genera la imagen de una palabra en estilo manuscrito.
La palabra debe contener solo caracteres del alfabeto (espacio, a-z, ñ).
Si la palabra tiene caracteres no válidos, se filtran o se rechaza según opciones.
"""
import os
import argparse
import glob
from pathlib import Path
import cv2
import torch
from torch.utils.data import DataLoader

from lib.utils import yaml2config
from lib.alphabet import get_true_alphabet, sanitize_word
from lib.datasets import get_dataset, get_collect_fn
from networks import get_model


def infer_logdir_from_ckpt(ckpt_path):
    if not ckpt_path:
        return None
    ckpt = Path(ckpt_path)
    if ckpt.suffix == ".pth" and ckpt.parent.name == "ckpts":
        return str(ckpt.parent.parent)
    return None


def sanitize_input_word(word, true_alphabet, max_len, allow_filter=False):
    cleaned = sanitize_word(word, true_alphabet, max_length=max_len, ignore_case=True)
    if cleaned:
        return cleaned
    if allow_filter:
        cleaned = "".join(c for c in word.lower() if c in true_alphabet)
        if not cleaned:
            return None
        if len(cleaned) >= max_len:
            cleaned = cleaned[: max_len - 1]
        return cleaned
    return None


def load_words(args, true_alphabet, max_len):
    if args.input_txt:
        words = []
        with open(args.input_txt, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                cleaned = sanitize_input_word(raw, true_alphabet, max_len, allow_filter=args.allow_filter)
                if cleaned is None:
                    print(f"AVISO: línea ignorada en {args.input_txt}:{line_no}: {raw!r}")
                    continue
                words.append(cleaned)
        if not words:
            raise SystemExit("ERROR: No se encontraron palabras válidas en el .txt de entrada.")
        return words

    palabra = args.palabra or args.word
    if not palabra:
        try:
            palabra = input("Palabra a generar: ").strip()
        except EOFError:
            pass
    if not palabra:
        raise SystemExit("Debes indicar la palabra (argumento posicional, --word o --input_txt)")

    cleaned = sanitize_input_word(palabra, true_alphabet, max_len, allow_filter=args.allow_filter)
    if cleaned is None:
        print("ERROR: La palabra tiene caracteres no válidos o longitud no permitida.")
        print(f"Alfabeto: {repr(true_alphabet)}, longitud máx: {max_len}")
        print("Usa --allow_filter para filtrar caracteres no válidos.")
        raise SystemExit(1)
    if cleaned != palabra:
        print(f"AVISO: Se usó la palabra filtrada: {repr(cleaned)}")
    return [cleaned]


def safe_word_filename(word):
    safe = "".join(c for c in word if c.isalnum() or c in ("-", "_")).strip("_")
    return safe or "word"


def main():
    parser = argparse.ArgumentParser(description="Generar imagen de una palabra con FW-GAN")
    parser.add_argument("palabra", type=str, nargs="?", default=None,
                        help="Palabra a generar (ej. hola)")
    parser.add_argument("--word", type=str, default=None,
                        help="Palabra a generar (alternativa a argumento posicional)")
    parser.add_argument("--input_txt", type=str, default=None,
                        help="Archivo .txt con una palabra por línea.")
    parser.add_argument("--config", type=str, default="./configs/fw_gan_iam.yml")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Ruta directa a checkpoint (.pth). Tiene prioridad sobre config.")
    parser.add_argument("--logdir", type=str, default=None,
                        help="Carpeta del run con checkpoint (ej. runs/fw_gan_iam-02-03-12-31)")
    parser.add_argument("--out", type=str, default=None,
                        help="Ruta de salida de la imagen (default: <palabra>.png en cwd)")
    parser.add_argument("--writer_id", type=int, default=None,
                        help="ID de autor para el estilo (opcional; si no se pasa se usa uno del dataset)")
    parser.add_argument("--style_split", type=str, default=None,
                        help="Split desde donde tomar la muestra de estilo (train/val/test). Default: cfg.valid.dset_split")
    parser.add_argument("--no_style_guided", action="store_true",
                        help="Usar ruido aleatorio en lugar de estilo de referencia")
    parser.add_argument("--allow_filter", action="store_true",
                        help="Si la palabra tiene caracteres no válidos, filtrarlos en lugar de fallar")
    args = parser.parse_args()

    cfg = yaml2config(args.config)
    true_alphabet = get_true_alphabet(cfg.dataset)
    max_len = getattr(cfg.training, "max_word_len", 20)
    words_to_generate = load_words(args, true_alphabet, max_len)

    # Device
    try:
        if getattr(cfg, "device", "cuda") == "cuda" and torch.cuda.is_available():
            cfg["device"] = "cuda"
            print(f"[INFO] Usando GPU: {torch.cuda.get_device_name(0)}")
        else:
            cfg["device"] = "cpu"
            print("[INFO] Usando CPU.")
    except Exception:
        cfg["device"] = "cpu"
        print("[INFO] Usando CPU.")

    # Logdir y checkpoint
    config_name = os.path.basename(args.config)[:-4]
    ckpt_path = args.ckpt if args.ckpt else getattr(cfg, "ckpt", None)
    inferred_logdir = infer_logdir_from_ckpt(ckpt_path)
    if args.logdir:
        logdir = args.logdir
        if not os.path.isdir(logdir):
            print(f"ERROR: No existe la carpeta {logdir}")
            return 1
    elif inferred_logdir and os.path.isdir(inferred_logdir):
        logdir = inferred_logdir
        print(f"[INFO] Run inferido desde checkpoint: {logdir}")
    else:
        pattern = os.path.join("runs", config_name + "*")
        candidates = [d for d in glob.glob(pattern) if os.path.isdir(d)]
        if candidates:
            logdir = max(candidates, key=os.path.getmtime)
            print(f"[INFO] Run usado: {logdir}")
        else:
            logdir = os.path.join("runs", config_name)
            if not os.path.isdir(logdir):
                print("ERROR: No hay carpeta de run. Entrena antes o pasa --logdir.")
                return 1

    model = get_model(cfg.model)(cfg, logdir)
    if ckpt_path and os.path.exists(ckpt_path):
        model.load(ckpt_path, cfg.device)
    else:
        ckpt_last = os.path.join(logdir, "ckpts", "last.pth")
        if os.path.exists(ckpt_last):
            model.load(ckpt_last, cfg.device)
        else:
            print("AVISO: No se encontró checkpoint; se usan pesos aleatorios.")

    # Una muestra de referencia para el estilo
    style_split = args.style_split or cfg.valid.dset_split
    dset = get_dataset(cfg.valid.dset_name or cfg.dataset, style_split, label_converter=model.label_converter)
    collate = get_collect_fn(getattr(cfg.training, "sort_input", False))
    if args.writer_id is None:
        loader = DataLoader(dset, batch_size=4, shuffle=True, collate_fn=collate, num_workers=0)
        imgs, img_lens, _, _, _ = next(iter(loader))
        style_img = imgs[0:1]
        style_len = img_lens[0:1]
    else:
        # Search deterministically through the split to find the requested writer.
        loader = DataLoader(dset, batch_size=64, shuffle=False, collate_fn=collate, num_workers=0)
        style_img = None
        style_len = None
        for imgs, img_lens, _, _, wids in loader:
            matches = (wids == args.writer_id).nonzero(as_tuple=False)
            if len(matches) > 0:
                idx = int(matches[0].item())
                style_img = imgs[idx:idx + 1]
                style_len = img_lens[idx:idx + 1]
                break
        if style_img is None:
            print(f"ERROR: writer_id={args.writer_id} no encontrado en split '{style_split}'.")
            return 1

    if args.input_txt:
        out_dir = args.out or "generated_words"
        os.makedirs(out_dir, exist_ok=True)
        labels_path = os.path.join(out_dir, "labels.txt")
        label_lines = []
        for idx, word_to_use in enumerate(words_to_generate, start=1):
            img = model.generate_word_image(
                word_to_use,
                style_img,
                style_len,
                style_guided=not args.no_style_guided,
            )
            filename = f"{idx:04d}.png"
            out_path = os.path.join(out_dir, filename)
            cv2.imwrite(out_path, img)
            label_lines.append(f"{filename} {word_to_use}\n")
            print(f"[OK] [{idx}/{len(words_to_generate)}] {word_to_use!r} -> {out_path}")
        with open(labels_path, "w", encoding="utf-8") as f:
            f.writelines(label_lines)
        print(f"[OK] Etiquetas guardadas en: {labels_path}")
    else:
        word_to_use = words_to_generate[0]
        img = model.generate_word_image(
            word_to_use,
            style_img,
            style_len,
            style_guided=not args.no_style_guided,
        )

        out_path = args.out or (word_to_use + ".png")
        if os.path.isdir(out_path):
            out_path = os.path.join(out_path, word_to_use + ".png")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        cv2.imwrite(out_path, img)
        print(f"Imagen guardada: {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
