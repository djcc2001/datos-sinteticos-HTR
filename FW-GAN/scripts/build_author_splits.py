#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def load_split(split_file, author_id):
    rows = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rel_path, label = line.split(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"Línea inválida en {split_file}:{line_no}: {line!r}") from exc
            parts = Path(rel_path).parts
            if len(parts) < 2 or parts[-2] != str(author_id):
                continue
            rows.append((rel_path, label))
    return rows


def stratified_train_val_split(rows, train_ratio, seed):
    if len(rows) < 2:
        return rows, []

    rng = random.Random(seed)
    pos = [row for row in rows if ("ñ" in row[1] or "Ñ" in row[1])]
    neg = [row for row in rows if row not in pos]
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_total = len(rows)
    n_val_total = max(1, int(round(n_total * (1.0 - train_ratio))))
    n_val_total = min(n_val_total, n_total - 1)

    def safe_val_count(group, proposed):
        if len(group) <= 1:
            return 0
        return min(max(proposed, 0), len(group) - 1)

    if pos:
        pos_val = max(1, int(round(len(pos) / n_total * n_val_total)))
        pos_val = safe_val_count(pos, pos_val)
    else:
        pos_val = 0

    neg_val = n_val_total - pos_val
    neg_val = safe_val_count(neg, neg_val)

    while (pos_val + neg_val) < n_val_total:
        if len(neg) - neg_val > 1:
            neg_val += 1
        elif len(pos) - pos_val > 1:
            pos_val += 1
        else:
            break

    val_rows = pos[:pos_val] + neg[:neg_val]
    train_rows = pos[pos_val:] + neg[neg_val:]
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def write_rows(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rel_path, label in rows:
            f.write(f"{rel_path} {label}\n")


def main():
    parser = argparse.ArgumentParser(description="Construye splits reproducibles para fine-tuning por autor.")
    parser.add_argument("--author_id", type=int, required=True)
    parser.add_argument("--train_split", default="splits/train.txt")
    parser.add_argument("--val_split", default="splits/val.txt")
    parser.add_argument("--test_split", default="splits/test.txt")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--out_dir", default="splits/authors")
    args = parser.parse_args()

    author_key = str(args.author_id)
    phase1_train_rows = load_split(args.train_split, author_key)
    phase1_val_rows = load_split(args.val_split, author_key)
    phase1_test_rows = load_split(args.test_split, author_key)

    if not phase1_train_rows:
        raise RuntimeError(f"No se encontraron muestras del autor {author_key} en {args.train_split}")

    ft_train_rows, ft_val_rows = stratified_train_val_split(
        phase1_train_rows,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    write_rows(ft_train_rows, out_dir / f"{author_key}_ft_train.txt")
    write_rows(ft_val_rows, out_dir / f"{author_key}_ft_val.txt")
    write_rows(phase1_val_rows, out_dir / f"{author_key}_eval_val.txt")
    write_rows(phase1_test_rows, out_dir / f"{author_key}_eval_test.txt")

    def count_enye(rows):
        return sum(1 for _, label in rows if ("ñ" in label or "Ñ" in label))

    print(
        f"Autor {author_key}: ft_train={len(ft_train_rows)} (ñ={count_enye(ft_train_rows)}), "
        f"ft_val={len(ft_val_rows)} (ñ={count_enye(ft_val_rows)}), "
        f"eval_val={len(phase1_val_rows)}, eval_test={len(phase1_test_rows)}"
    )


if __name__ == "__main__":
    main()
