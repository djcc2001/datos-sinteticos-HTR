import os
import argparse
import random
import sys
from typing import Dict, List, Tuple
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.utils import yaml2config
from lib.datasets import get_dataset
from lib.alphabet import get_true_alphabet, sanitize_word, Alphabets
from networks import get_model


def select_device(cfg):
    """Selecciona GPU si está disponible, respetando el campo cfg.device."""
    try:
        if getattr(cfg, "device", "cuda") == "cuda" and torch.cuda.is_available():
            cfg["device"] = "cuda"
            print(f"[INFO] Usando GPU: {torch.cuda.get_device_name(0)}")
        else:
            cfg["device"] = "cpu"
            print("[INFO] CUDA no disponible. Usando CPU.")
    except Exception:
        cfg["device"] = "cpu"
        print("[INFO] Usando CPU.")
    return cfg


def build_style_pool(dset, max_per_writer: int = 16) -> Dict[int, List[Tuple[torch.Tensor, int]]]:
    """
    Recorre el dataset una vez y guarda algunas muestras por escritor.
    Cada entrada es (img_tensor, img_len), con img_tensor de forma [1, H, W].
    """
    style_pool: Dict[int, List[Tuple[torch.Tensor, int]]] = {}
    print("[INFO] Construyendo pool de estilos por escritor...")
    for img, _, wid in tqdm(dset, desc="Escaneando escritores"):
        wid_int = int(wid)
        if wid_int not in style_pool:
            style_pool[wid_int] = []
        if len(style_pool[wid_int]) >= max_per_writer:
            continue
        # img es tensor [1, H, W]
        img_len = img.size(-1)
        style_pool[wid_int].append((img, img_len))
    print(f"[INFO] Pool de estilos construido con {len(style_pool)} escritores.")
    return style_pool


def pick_style(style_pool, writer_opt: str):
    """
    Selecciona una imagen de estilo a partir del pool.

    - writer_opt == 'random' -> escritor aleatorio.
    - writer_opt es entero (en string) -> ese writer si existe.
    """
    if not style_pool:
        raise RuntimeError("El pool de estilos está vacío. ¿Se construyó correctamente el HDF5?")

    if writer_opt == "random":
        wid = random.choice(list(style_pool.keys()))
    else:
        try:
            wid_int = int(writer_opt)
        except ValueError:
            print(f"[WARN] Valor de --writer no reconocido ('{writer_opt}'); usando 'random'.")
            wid_int = random.choice(list(style_pool.keys()))
        if wid_int not in style_pool:
            print(f"[WARN] Writer {wid_int} no encontrado en el dataset; usando aleatorio.")
            wid_int = random.choice(list(style_pool.keys()))
        wid = wid_int

    samples = style_pool[wid]
    style_img, style_len = random.choice(samples)
    return wid, style_img, style_len


def upscale_to_64px(img):
    """
    El modelo genera imágenes de altura interna ~32 px.
    Para el OCR queremos salida a 64 px de alto:
      - reescalado bicúbico a H=64 manteniendo proporción de aspecto.
    """
    h, w = img.shape
    if h == 64:
        return img
    new_w = max(1, int(w * 64 / max(h, 1)))
    return cv2.resize(img, (new_w, 64), interpolation=cv2.INTER_CUBIC)


def main():
    parser = argparse.ArgumentParser(
        description="Generar imágenes manuscritas desde un .txt usando FW-GAN."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/fw_gan_iam.yml",
        help="Ruta al archivo de configuración YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Ruta al checkpoint entrenado (ej. checkpoints/fwgan.pth).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Fichero .txt con una palabra/frase por línea.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Carpeta de salida para las imágenes generadas.",
    )
    parser.add_argument(
        "--writer",
        type=str,
        default="random",
        help="ID de escritor concreto (entero) o 'random' para muestrear aleatoriamente estilos.",
    )
    parser.add_argument(
        "--no_style_guided",
        action="store_true",
        help="Si se indica, ignora el estilo del escritor y usa solo ruido (estilo puramente sintético).",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"No se encuentra el fichero de entrada: {args.input}")
    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"No se encuentra el checkpoint: {args.checkpoint}")

    os.makedirs(args.output, exist_ok=True)

    # 1) Cargar configuración y dispositivo
    cfg = yaml2config(args.config)
    cfg = select_device(cfg)

    # 2) Preparar modelo
    #    El logdir aquí solo se usa para compatibilidad; no entrenamos, solo inferencia.
    logdir = args.output
    model = get_model(cfg.model)(cfg, logdir)

    print(f"[INFO] Cargando checkpoint desde {args.checkpoint}")
    _ = model.load(args.checkpoint, cfg.device)

    # 3) Dataset solo para obtener estilos de escritores
    dset_name = cfg.valid.dset_name if cfg.valid.dset_name else cfg.dataset
    dset = get_dataset(dset_name, cfg.valid.dset_split, label_converter=model.label_converter)
    style_pool = build_style_pool(dset, max_per_writer=16)

    # 4) Preparar alfabeto real para filtrar texto
    true_alphabet = get_true_alphabet(cfg.dataset)
    print(f"[INFO] Alfabeto activo ({len(true_alphabet)} caracteres): {true_alphabet}")

    # 5) Leer palabras del .txt
    words: List[str] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            clean = sanitize_word(
                raw,
                true_alphabet=true_alphabet,
                max_length=cfg.training.max_word_len,
                ignore_case=False,
            )
            if clean is None:
                print(
                    f"[WARN] Línea ignorada (vacía, muy larga o fuera de alfabeto): '{raw}'"
                )
                continue
            words.append(clean)

    if not words:
        raise SystemExit(
            "No se encontraron palabras válidas en el fichero de entrada tras sanear con el alfabeto actual."
        )

    print(f"[INFO] Palabras a generar: {len(words)}")

    # 6) Generar una imagen por palabra
    for idx, word in enumerate(words):
        wid, style_img, style_len = pick_style(style_pool, args.writer)
        img_np = model.generate_word_image(
            word=word,
            style_img=style_img,
            style_img_len=style_len,
            style_guided=not args.no_style_guided,
        )

        # Normalizar a 64 px de alto para el OCR
        img_np = upscale_to_64px(img_np)

        # Nombre de fichero: índice + palabra saneada
        safe_word = "".join(
            c for c in word if c.isalnum() or c in ("-", "_")
        ).strip() or f"word{idx:04d}"
        filename = f"{idx:04d}_{safe_word}.png"
        out_path = os.path.join(args.output, filename)

        cv2.imwrite(out_path, img_np)
        print(f"[OK] [{idx+1}/{len(words)}] '{word}' -> {out_path} (writer {wid})")


if __name__ == "__main__":
    main()
