import argparse
import csv
from collections import Counter


def levenshtein_alignment(ref: str, hyp: str):
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            options = [
                (dp[i - 1][j] + 1, "del"),
                (dp[i][j - 1] + 1, "ins"),
                (dp[i - 1][j - 1] + cost, "eq" if cost == 0 else "sub"),
            ]
            dp[i][j], bt[i][j] = min(options, key=lambda x: x[0])

    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        op = bt[i][j]
        if op in {"eq", "sub"}:
            pairs.append((ref[i - 1], hyp[j - 1], op))
            i -= 1
            j -= 1
        elif op == "del":
            pairs.append((ref[i - 1], "", op))
            i -= 1
        elif op == "ins":
            pairs.append(("", hyp[j - 1], op))
            j -= 1
        else:
            break
    pairs.reverse()
    return dp[n][m], pairs


def load_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"image_path", "gt_text", "pred_text"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain TSV headers: image_path, gt_text, pred_text")
        for row in reader:
            rows.append(row)
    return rows


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def main():
    parser = argparse.ArgumentParser(description="Compute character-level metrics, including n/ñ confusion.")
    parser.add_argument("--predictions_tsv", type=str, required=True)
    args = parser.parse_args()

    rows = load_rows(args.predictions_tsv)
    total_chars = 0
    correct_chars = 0
    total_edit = 0
    total_words = 0
    exact_words = 0
    words_with_enye = 0
    exact_words_with_enye = 0
    confusion = Counter()

    for row in rows:
        gt = row["gt_text"]
        pred = row["pred_text"]
        total_words += 1
        if gt == pred:
            exact_words += 1

        edit_distance, aligned = levenshtein_alignment(gt, pred)
        total_edit += edit_distance
        total_chars += len(gt)
        correct_chars += sum(1 for g, p, op in aligned if op == "eq")

        has_enye = any(ch in "ñÑ" for ch in gt)
        if has_enye:
            words_with_enye += 1
            if gt == pred:
                exact_words_with_enye += 1

        for g, p, _ in aligned:
            if g in "nNñÑ" and p in "nNñÑ" and g != p:
                confusion[(g, p)] += 1

    print("metric\tvalue")
    print(f"num_samples\t{total_words}")
    print(f"word_accuracy\t{pct(exact_words, total_words):.4f}")
    print(f"char_accuracy\t{pct(correct_chars, total_chars):.4f}")
    print(f"CER\t{(total_edit / total_chars) if total_chars else 0.0:.6f}")
    print(f"word_accuracy_with_ñ\t{pct(exact_words_with_enye, words_with_enye):.4f}")
    print(f"samples_with_ñ\t{words_with_enye}")
    print(f"n_to_ñ\t{confusion[('n', 'ñ')] + confusion[('N', 'Ñ')]}")
    print(f"ñ_to_n\t{confusion[('ñ', 'n')] + confusion[('Ñ', 'N')]}")
    print(f"n_to_Ñ\t{confusion[('n', 'Ñ')] + confusion[('N', 'ñ')]}")
    print(f"ñ_to_N\t{confusion[('ñ', 'N')] + confusion[('Ñ', 'n')]}")


if __name__ == "__main__":
    main()
