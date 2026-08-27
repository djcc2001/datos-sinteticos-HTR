import argparse
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

from lib.alphabet import Alphabets


IMG_H = 32
ALPHABET = Alphabets["custom"]
CHAR2IDX = {c: i for i, c in enumerate(ALPHABET)}


def normalize_height(img):
    h, w = img.shape
    if h == IMG_H:
        return img
    scale = IMG_H / float(h)
    new_w = max(2, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, IMG_H), interpolation=interp)


def encode_text(text):
    unknown = [c for c in text if c not in CHAR2IDX]
    if unknown:
        raise ValueError(f"Caracteres fuera del alfabeto: {unknown} en texto {text!r}")
    return np.asarray([CHAR2IDX[c] for c in text], dtype=np.int32)


def parse_split_file(split_file):
    samples = []
    split_path = Path(split_file)
    with split_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rel_img_path, label = line.split(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"Línea inválida en {split_file}:{line_no}: {line!r}") from exc

            img_path = Path(rel_img_path)
            if not img_path.is_absolute():
                img_path = Path.cwd() / img_path
            if not img_path.exists():
                raise FileNotFoundError(f"Imagen ausente en {split_file}:{line_no}: {img_path}")

            writer_id = int(Path(rel_img_path).parts[-2]) if Path(rel_img_path).parts[-2].isdigit() else 0
            samples.append((str(img_path), label, writer_id))
    if not samples:
        raise RuntimeError(f"No se encontraron muestras en {split_file}")
    return samples


def build_hdf5(out_path, split_file):
    imgs_cat = []
    img_seek = []
    img_lens = []
    lbs_cat = []
    lb_seek = []
    lb_lens = []
    wids = []

    img_cursor = 0
    lb_cursor = 0

    for img_path, label, writer_id in parse_split_file(split_file):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"No se pudo leer la imagen: {img_path}")

        img = normalize_height(img).astype(np.uint8)
        lb = encode_text(label)
        width = img.shape[1]

        imgs_cat.append(img)
        img_seek.append(img_cursor)
        img_lens.append(width)
        img_cursor += width

        lbs_cat.append(lb)
        lb_seek.append(lb_cursor)
        lb_lens.append(len(lb))
        lb_cursor += len(lb)
        wids.append(writer_id)

    imgs_cat = np.concatenate(imgs_cat, axis=1)
    lbs_cat = np.concatenate(lbs_cat)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("imgs", data=imgs_cat, dtype=np.uint8)
        f.create_dataset("img_seek_idxs", data=np.asarray(img_seek, dtype=np.int64))
        f.create_dataset("img_lens", data=np.asarray(img_lens, dtype=np.int32))
        f.create_dataset("lbs", data=lbs_cat, dtype=np.int32)
        f.create_dataset("lb_seek_idxs", data=np.asarray(lb_seek, dtype=np.int64))
        f.create_dataset("lb_lens", data=np.asarray(lb_lens, dtype=np.int32))
        f.create_dataset("wids", data=np.asarray(wids, dtype=np.int32))

    print(f"Saved: {out_path} ({len(wids)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construye HDF5 a partir de splits/*.txt")
    parser.add_argument("--train_split", default="splits/train.txt")
    parser.add_argument("--val_split", default="splits/val.txt")
    parser.add_argument("--test_split", default="splits/test.txt")
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    build_hdf5(os.path.join(args.out_dir, "train.hdf5"), args.train_split)
    build_hdf5(os.path.join(args.out_dir, "val.hdf5"), args.val_split)
    build_hdf5(os.path.join(args.out_dir, "test.hdf5"), args.test_split)
