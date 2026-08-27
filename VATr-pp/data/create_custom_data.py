import argparse
import collections
import pickle
from pathlib import Path
import unicodedata

import cv2
import numpy as np
from PIL import Image


def resize_keep_height(image: np.ndarray, target_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image shape")
    scale = target_height / float(h)
    target_width = max(1, int(round(w * scale)))
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    return resized


def normalize_label(label: str) -> str:
    return unicodedata.normalize("NFC", str(label).strip())


def parse_split_file(split_file: Path):
    samples = []
    with split_file.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if " " not in line:
                raise ValueError(f"Malformed line at {split_file}:{line_no}: '{line}'")

            image_rel, label = line.split(" ", 1)
            label = normalize_label(label)
            if not image_rel or not label:
                continue

            samples.append((image_rel.strip(), label))
    return samples


def resolve_image_path(repo_root: Path, image_rel: str) -> Path:
    image_path = (repo_root / image_rel).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_rel}")
    return image_path


def infer_author_id(image_rel: str) -> str:
    parts = Path(image_rel).parts
    if len(parts) < 2:
        raise ValueError(f"Could not infer author folder from path: {image_rel}")
    return parts[-2]


def build_pickle_split(repo_root: Path, split_name: str, split_file: Path, target_height: int, image_counter_start: int):
    author_samples = collections.defaultdict(list)
    image_counter = image_counter_start

    for image_rel, label in parse_split_file(split_file):
        image_path = resolve_image_path(repo_root, image_rel)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size <= 1:
            continue

        author_id = infer_author_id(image_rel)
        image = resize_keep_height(image, target_height)
        author_samples[author_id].append(
            {
                "img": Image.fromarray(image),
                "label": label,
                "image_id": image_counter,
                "original_image_id": image_path.stem,
                "img_path": image_rel,
                "split": split_name,
            }
        )
        image_counter += 1

    return dict(author_samples), image_counter


def validate_no_leakage(repo_root: Path, split_to_samples: dict):
    seen_paths = {}
    for split_name, split_file in split_to_samples.items():
        for image_rel, _ in parse_split_file(split_file):
            norm_path = str(resolve_image_path(repo_root, image_rel))
            prev = seen_paths.get(norm_path)
            if prev is not None and prev != split_name:
                raise ValueError(
                    f"Data leakage detected: {image_rel} appears in both '{prev}' and '{split_name}'"
                )
            seen_paths[norm_path] = split_name


def main():
    parser = argparse.ArgumentParser(description="Create VATr++ pickle dataset from splits/*.txt with UTF-8 labels.")
    parser.add_argument("--root-dir", type=str, default=".")
    parser.add_argument("--train-split", type=str, default="splits/train.txt")
    parser.add_argument("--val-split", type=str, default="splits/val.txt")
    parser.add_argument("--test-split", type=str, default="splits/test.txt")
    parser.add_argument("--output", type=str, default="files/CUSTOM-32.pickle")
    parser.add_argument("--height", type=int, default=32)
    args = parser.parse_args()

    repo_root = Path(args.root_dir).resolve()
    split_files = {
        "train": (repo_root / args.train_split).resolve(),
        "val": (repo_root / args.val_split).resolve(),
        "test": (repo_root / args.test_split).resolve(),
    }
    for split_name, split_file in split_files.items():
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found for {split_name}: {split_file}")

    validate_no_leakage(repo_root, split_files)

    dataset = {"train": {}, "val": {}, "test": {}}
    image_counter = 0

    for split_name, split_file in split_files.items():
        subset_data, image_counter = build_pickle_split(repo_root, split_name, split_file, args.height, image_counter)
        dataset[split_name] = subset_data

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved: {output_path}")
    for split_name in ("train", "val", "test"):
        author_count = len(dataset[split_name])
        image_count = sum(len(v) for v in dataset[split_name].values())
        print(f"{split_name}: authors={author_count} images={image_count}")

    train_authors = set(dataset["train"])
    val_authors = set(dataset["val"])
    test_authors = set(dataset["test"])
    print(f"author_overlap(train,val)={len(train_authors & val_authors)}")
    print(f"author_overlap(train,test)={len(train_authors & test_authors)}")
    print(f"author_overlap(val,test)={len(val_authors & test_authors)}")


if __name__ == "__main__":
    main()
