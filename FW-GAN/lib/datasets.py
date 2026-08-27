import os
import h5py
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

from lib.path_config import data_roots, data_paths, split_files


def _recalc_len(length, scale):
    tmp = length % scale
    return length + scale - tmp if tmp != 0 else length


def _extract_writer_id(path_str):
    parts = Path(path_str).parts
    if len(parts) < 2:
        return 0
    parent = parts[-2]
    return int(parent) if str(parent).isdigit() else 0


def resolve_split_file(dataset_name, split, split_file=None):
    if split_file is not None:
        return str(Path(split_file))
    tag = "_".join(dataset_name.split("_")[:2])
    split_key = str(split).lower()
    if tag in split_files and split_key in split_files[tag]:
        return split_files[tag][split_key]
    return None


class BatchCollatorMixin:
    @staticmethod
    def collect_fn(batch):
        imgs, lbs, wids, lb_lens, img_lens, pad_img_lens = [], [], [], [], [], []

        for img, lb, wid in batch:
            if isinstance(img, torch.Tensor):
                if img.dim() == 3 and img.size(0) == 1:
                    img = img[0]
                img = img.numpy()
            imgs.append(img)
            lbs.append(lb)
            wids.append(wid)
            lb_lens.append(len(lb))
            img_lens.append(img.shape[-1])
            pad_img_lens.append(_recalc_len(img.shape[-1], max(img.shape[-2] // 2, 1)))

        bz = len(lb_lens)
        img_height = imgs[0].shape[-2]
        max_img_len = max(pad_img_lens)
        pad_imgs = np.ones((bz, 1, img_height, max_img_len), dtype=np.float32)
        for i, (img, img_len) in enumerate(zip(imgs, img_lens)):
            pad_imgs[i, 0, :, :img_len] = img

        max_lb_len = max(lb_lens)
        pad_lbs = np.zeros((bz, max_lb_len), dtype=np.int64)
        for i, (lb, lb_len) in enumerate(zip(lbs, lb_lens)):
            pad_lbs[i, :lb_len] = lb

        imgs = torch.from_numpy(pad_imgs).float()
        img_lens = torch.tensor(img_lens, dtype=torch.int32)
        lbs = torch.from_numpy(pad_lbs).long()
        lb_lens = torch.tensor(lb_lens, dtype=torch.int32)
        wids = torch.tensor(wids, dtype=torch.long)
        return imgs, img_lens, lbs, lb_lens, wids

    @staticmethod
    def sort_collect_fn(batch):
        imgs, lbs, wids = zip(*batch)
        img_lens = np.array([img.size(-1) for img in imgs]).astype(np.int32)
        idx = np.argsort(img_lens)[::-1]
        imgs = [imgs[i] for i in idx]
        lbs = [lbs[i] for i in idx]
        wids = [wids[i] for i in idx]
        return BatchCollatorMixin.collect_fn(list(zip(imgs, lbs, wids)))

    @staticmethod
    def merge_batch(batch1, batch2, device):
        imgs1, img_lens1, lbs1, lb_lens1, wids1 = batch1
        imgs2, img_lens2, lbs2, lb_lens2, wids2 = batch2
        bz1, bz2 = imgs1.size(0), imgs2.size(0)

        max_img_len = max(imgs1.size(-1), imgs2.size(-1))
        pad_imgs = torch.ones((bz1 + bz2, imgs1.size(1), imgs1.size(2), max_img_len), dtype=torch.float32, device=device)
        pad_imgs[:bz1, :, :, :imgs1.size(-1)] = imgs1.to(device)
        pad_imgs[bz1:, :, :, :imgs2.size(-1)] = imgs2.to(device)

        max_lb_len = max(lb_lens1.max(), lb_lens2.max()).item()
        pad_lbs = torch.zeros((bz1 + bz2, max_lb_len), dtype=torch.long, device=device)
        pad_lbs[:bz1, :lbs1.size(-1)] = lbs1.to(device)
        pad_lbs[bz1:, :lbs2.size(-1)] = lbs2.to(device)

        merge_img_lens = torch.cat([img_lens1, img_lens2]).to(device)
        merge_lb_lens = torch.cat([lb_lens1, lb_lens2]).to(device)
        merge_wids = torch.cat([wids1, wids2]).long().to(device)
        return pad_imgs, merge_img_lens, pad_lbs, merge_lb_lens, merge_wids


class Hdf5Dataset(Dataset, BatchCollatorMixin):
    def __init__(self, root, split, transforms=None):
        super(Hdf5Dataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self._load_h5py(split)

    def _load_h5py(self, split):
        self.file_path = os.path.join(self.root, split)
        with h5py.File(self.file_path, "r") as h5f:
            self.imgs = h5f["imgs"][:]
            self.lbs = h5f["lbs"][:]
            self.img_seek_idxs = h5f["img_seek_idxs"][:]
            self.lb_seek_idxs = h5f["lb_seek_idxs"][:]
            self.img_lens = h5f["img_lens"][:]
            self.lb_lens = h5f["lb_lens"][:]
            self.wids = h5f["wids"][:] if "wids" in h5f else np.zeros((len(self.img_lens),), dtype=np.int64)

    def __getitem__(self, idx):
        img_seek_idx, img_len = self.img_seek_idxs[idx], self.img_lens[idx]
        lb_seek_idx, lb_len = self.lb_seek_idxs[idx], self.lb_lens[idx]
        img = self.imgs[:, img_seek_idx: img_seek_idx + img_len]
        lb = self.lbs[lb_seek_idx: lb_seek_idx + lb_len].astype(np.int64)
        wid = int(self.wids[idx])

        img = Image.fromarray(img, mode="L")
        if self.transforms is not None:
            img = self.transforms(img)
        return img, lb, wid

    def __len__(self):
        return int(len(self.img_lens))


class FileListDataset(Dataset, BatchCollatorMixin):
    def __init__(self, split_file, transforms=None, strict=True):
        super(FileListDataset, self).__init__()
        self.split_file = Path(split_file)
        self.transforms = transforms
        self.strict = strict
        self.samples = self._load_samples()

    def _load_samples(self):
        if not self.split_file.exists():
            raise FileNotFoundError(f"No existe el split file: {self.split_file}")

        samples = []
        with self.split_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    rel_path, label = line.split(maxsplit=1)
                except ValueError as exc:
                    raise ValueError(f"Línea inválida en {self.split_file}:{line_no}: {line!r}") from exc

                img_path = Path(rel_path)
                if not img_path.is_absolute():
                    img_path = Path.cwd() / img_path
                if not img_path.exists():
                    msg = f"Imagen ausente en {self.split_file}:{line_no}: {img_path}"
                    if self.strict:
                        raise FileNotFoundError(msg)
                    continue

                samples.append(
                    {
                        "img_path": str(img_path),
                        "label": label,
                        "writer_id": _extract_writer_id(rel_path),
                    }
                )
        if not samples:
            raise RuntimeError(f"No se encontraron muestras válidas en {self.split_file}")
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("L")
        if self.transforms is not None:
            img = self.transforms(img)
        label = np.fromiter((ord(c) for c in sample["label"]), dtype=np.int64)
        return img, label, sample["writer_id"]

    def __len__(self):
        return len(self.samples)


class EncodedFileListDataset(FileListDataset):
    def __init__(self, split_file, transforms=None, strict=True, label_converter=None):
        if label_converter is None:
            raise ValueError("EncodedFileListDataset requiere label_converter para codificar labels.")
        self.label_converter = label_converter
        super().__init__(split_file, transforms=transforms, strict=strict)
        self._validate_labels()

    def _validate_labels(self):
        for sample in self.samples:
            self.label_converter.encode(sample["label"])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("L")
        if self.transforms is not None:
            img = self.transforms(img)
        label = np.asarray(self.label_converter.encode(sample["label"]), dtype=np.int64)
        return img, label, sample["writer_id"]


def get_dataset(name, split, label_converter=None, split_file=None):
    tag = "_".join(name.split("_")[:2])
    transforms = Compose([ToTensor(), Normalize([0.5], [0.5])])
    resolved_split_file = resolve_split_file(name, split, split_file=split_file)

    if resolved_split_file is not None:
        return EncodedFileListDataset(
            resolved_split_file,
            transforms=transforms,
            strict=True,
            label_converter=label_converter,
        )

    h5_name = data_paths[name][split]
    return Hdf5Dataset(data_roots[tag], h5_name, transforms=transforms)


def get_collect_fn(sort_input=False):
    return BatchCollatorMixin.sort_collect_fn if sort_input else BatchCollatorMixin.collect_fn


def get_max_image_width(dset):
    max_image_width = 0
    for img, _, _ in dset:
        max_image_width = max(max_image_width, img.size(-1))
    return max_image_width
