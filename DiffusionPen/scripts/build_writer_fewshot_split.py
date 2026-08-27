import argparse
import os
import random


def read_split(path: str):
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


def filter_writer(rows, writer_id: int):
    marker = f"/{writer_id}/"
    return [(img, text) for img, text in rows if marker in img.replace("\\", "/")]


def write_manifest(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for img, text in rows:
            f.write(f"{img} {text}\n")


def write_words(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for _, text in rows:
            f.write(f"{text}\n")


def main():
    parser = argparse.ArgumentParser(description="Build a few-shot style manifest and evaluation word list for one writer.")
    parser.add_argument("--writer_id", type=int, required=True)
    parser.add_argument("--train_split", type=str, default="splits/train.txt")
    parser.add_argument("--val_split", type=str, default="splits/val.txt")
    parser.add_argument("--test_split", type=str, default="splits/test.txt")
    parser.add_argument("--num_refs", type=int, default=5)
    parser.add_argument("--num_eval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_rows = filter_writer(read_split(args.train_split), args.writer_id)
    val_rows = filter_writer(read_split(args.val_split), args.writer_id)
    test_rows = filter_writer(read_split(args.test_split), args.writer_id)

    if len(train_rows) == 0:
        raise ValueError(f"No training rows found for writer {args.writer_id}")

    if len(train_rows) >= args.num_refs:
        ref_rows = random.sample(train_rows, k=args.num_refs)
    else:
        ref_rows = random.choices(train_rows, k=args.num_refs)

    eval_pool = val_rows + test_rows
    if len(eval_pool) == 0:
        eval_pool = [row for row in train_rows if row not in ref_rows]
    if len(eval_pool) == 0:
        eval_pool = train_rows

    if len(eval_pool) >= args.num_eval:
        eval_rows = random.sample(eval_pool, k=args.num_eval)
    else:
        eval_rows = eval_pool[:]

    refs_path = os.path.join(args.output_dir, f"writer_{args.writer_id}_refs_k{args.num_refs}.txt")
    eval_manifest_path = os.path.join(args.output_dir, f"writer_{args.writer_id}_eval_manifest.txt")
    eval_words_path = os.path.join(args.output_dir, f"writer_{args.writer_id}_eval_words.txt")

    write_manifest(refs_path, ref_rows)
    write_manifest(eval_manifest_path, eval_rows)
    write_words(eval_words_path, eval_rows)

    print(f"writer={args.writer_id} refs={len(ref_rows)} eval={len(eval_rows)}")
    print(f"refs_manifest={refs_path}")
    print(f"eval_manifest={eval_manifest_path}")
    print(f"eval_words={eval_words_path}")


if __name__ == "__main__":
    main()
