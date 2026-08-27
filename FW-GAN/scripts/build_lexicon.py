import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.alphabet import Alphabets


def main():
    parser = argparse.ArgumentParser(description="Genera un léxico único desde splits/train.txt")
    parser.add_argument("--input", default="splits/train.txt")
    parser.add_argument("--output", default="data/spanish_words.txt")
    parser.add_argument("--alphabet", default="custom")
    parser.add_argument("--max_length", type=int, default=24)
    args = parser.parse_args()

    allowed = set(Alphabets[args.alphabet])
    words = set()

    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                _, label = line.split(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"Línea inválida en {args.input}:{line_no}: {line!r}") from exc

            if len(label) < 2 or len(label) >= args.max_length:
                continue
            unknown = [c for c in label if c not in allowed]
            if unknown:
                raise ValueError(f"Caracteres fuera de alfabeto en {args.input}:{line_no}: {unknown}")
            words.add(label.lower())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for word in sorted(words):
            f.write(f"{word}\n")

    print(f"Lexicon generado en {output} con {len(words)} entradas")


if __name__ == "__main__":
    main()
