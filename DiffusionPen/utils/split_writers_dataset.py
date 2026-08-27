import json
import os
import unicodedata
from typing import Iterable

from utils.word_dataset import WordLineDataset


def _normalize_label(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    # Preserve full Unicode content while dropping hidden control chars.
    return "".join(ch for ch in text if ch.isprintable())


class SplitWritersDataset(WordLineDataset):
    """
    Dataset driven strictly by manifest files with the format:
        relative/or/absolute/image_path label text

    The writer id is inferred from the parent folder name of each image path.
    No directory crawling is performed, which keeps train/val/test fully
    controlled by the provided split files.
    """

    CACHE_VERSION = 3

    def __init__(
        self,
        basefolder,
        subset,
        segmentation_level,
        fixed_size,
        tokenizer,
        text_encoder,
        feat_extractor,
        transforms,
        args,
    ):
        super().__init__(
            basefolder,
            subset,
            segmentation_level,
            fixed_size,
            tokenizer,
            text_encoder,
            feat_extractor,
            transforms,
            character_classes=None,
            args=args,
        )
        self.setname = "CUSTOM_SPLITS"
        self.args = args
        self.writer_id_map = {}
        self.index_to_writer = {}
        self.__finalize__()

    def __finalize__(self):
        data = self.main_loader(self.subset, self.segmentation_level)
        self.data = data

        self.initial_writer_ids = [d[2] for d in data]
        writer_ids = sorted({d[2] for d in data})
        self.writer_ids = writer_ids
        self.wclasses = len(writer_ids)
        print("Number of writers", self.wclasses)

        if self.character_classes is None:
            res = set()
            for _, transcr, _, _ in data:
                res.update(list(transcr))
                self.max_transcr_len = max(self.max_transcr_len, len(transcr))
            res = sorted(list(res))
            if " " not in res:
                res.append(" ")
            print("Character classes: {} ({} different characters)".format(res, len(res)))
            print("Max transcription length: {}".format(self.max_transcr_len))
            self.character_classes = res

        self._build_writer_indices()

    def _resolve_split_path(self, subset: str) -> str:
        split_map = {
            "train": getattr(self.args, "train_split", None),
            "val": getattr(self.args, "val_split", None),
            "validation": getattr(self.args, "val_split", None),
            "test": getattr(self.args, "test_split", None),
        }
        split_path = split_map.get(subset)
        if not split_path:
            raise ValueError(f"Split path not configured for subset '{subset}'")
        if not os.path.isabs(split_path):
            split_path = os.path.join(os.getcwd(), split_path)
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")
        return split_path

    def _save_writer_dict(self, subset: str) -> None:
        path = getattr(self.args, "writer_map_path", None)
        if not path:
            path = f"./writers_dict_{subset}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.writer_id_map, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _resolve_image_path(basefolder: str, raw_path: str) -> str:
        if os.path.isabs(raw_path):
            return raw_path
        return os.path.normpath(os.path.join(os.getcwd(), raw_path))

    @staticmethod
    def _infer_writer_id(img_path: str) -> int:
        writer_folder = os.path.basename(os.path.dirname(img_path))
        try:
            return int(writer_folder)
        except ValueError as exc:
            raise ValueError(
                f"Could not infer integer writer id from parent folder '{writer_folder}' for '{img_path}'"
            ) from exc

    @staticmethod
    def _iter_manifest_rows(split_path: str) -> Iterable[tuple[str, str]]:
        with open(split_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    raise ValueError(
                        f"Invalid split row at {split_path}:{lineno}. Expected 'image_path label_text'."
                    )
                yield parts[0], parts[1]

    def main_loader(self, subset, segmentation_level) -> list:
        split_path = self._resolve_split_path(subset)
        data_raw = []
        missing_images = 0
        skipped_empty = 0

        for raw_img_path, raw_text in self._iter_manifest_rows(split_path):
            img_path = self._resolve_image_path(self.basefolder, raw_img_path)
            if not os.path.isfile(img_path):
                missing_images += 1
                continue

            text = _normalize_label(raw_text)
            if len(text) == 0:
                skipped_empty += 1
                continue

            writer_id = self._infer_writer_id(img_path)
            data_raw.append((img_path, text, writer_id, img_path))

        if not data_raw:
            raise ValueError(f"No usable samples found in split '{split_path}'")

        present_writers = sorted({wid for _, _, wid, _ in data_raw})
        self.writer_id_map = {wid: idx for idx, wid in enumerate(present_writers)}
        self.index_to_writer = {idx: wid for wid, idx in self.writer_id_map.items()}
        self._save_writer_dict(subset)

        data = [
            (img_path0, text, self.writer_id_map[wid], img_path)
            for img_path0, text, wid, img_path in data_raw
        ]

        print(f"subset={subset} split={split_path} len data={len(data)} writers={len(present_writers)}")
        if missing_images > 0:
            print("missing images", missing_images)
        if skipped_empty > 0:
            print("skipped empty labels", skipped_empty)

        return data
