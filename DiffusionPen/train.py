import argparse
import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
from transformers import CanineModel, CanineTokenizer

from feature_extractor import ImageEncoder
from unet import UNetModel
from utils.iam_writers_dataset import IAMWritersDataset
from utils.spanish_dataset import SpanishWritersDataset
from utils.split_writers_dataset import SplitWritersDataset
from torch.utils.data import ConcatDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(save_path: str):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)


def save_grid(images: torch.Tensor, path: str, nrow: int = 4):
    """
    images: float tensor in [0,1], shape [B,3,H,W]
    """
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=0)
    torchvision.utils.save_image(grid, path)


class EMA:
    def __init__(self, decay: float, warmup_steps: int = 0):
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.steps = 0

    @torch.no_grad()
    def update(self, ema_model: nn.Module, model: nn.Module):
        self.steps += 1
        if self.steps <= self.warmup_steps:
            ema_model.load_state_dict(model.state_dict(), strict=True)
            return

        d = self.decay
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            if not v.is_floating_point():
                v.copy_(msd[k])
                continue
            v.mul_(d).add_(msd[k].to(v.dtype), alpha=1.0 - d)


@dataclass
class Batch:
    images: torch.Tensor
    texts: list
    writer_ids: torch.Tensor
    style_images: torch.Tensor


def to_batch(data, device: torch.device) -> Batch:
    images = data[0].to(device, non_blocking=True)
    texts = list(data[1])
    writer_ids = data[2].to(device, non_blocking=True)
    style_images = data[3].to(device, non_blocking=True)
    return Batch(images=images, texts=texts, writer_ids=writer_ids, style_images=style_images)


def build_schedulers(stable_dif_path: str, sampling_steps: int):
    ddim = DDIMScheduler.from_pretrained(stable_dif_path, subfolder="scheduler")
    train_scheduler = DDPMScheduler.from_config(ddim.config)
    sample_scheduler = DDIMScheduler.from_config(ddim.config)
    sample_scheduler.set_timesteps(sampling_steps)
    return train_scheduler, sample_scheduler


def build_dataloaders(args, transform):
    if args.dataset == "iam":
        ds_train = IAMWritersDataset(args.dataset_root, "train", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        ds_val = IAMWritersDataset(args.dataset_root, "val", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        ds_test = IAMWritersDataset(args.dataset_root, "test", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
    elif args.dataset == "spanish":
        ds_train = SpanishWritersDataset(args.dataset_root, "train", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        ds_val = SpanishWritersDataset(args.dataset_root, "val", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        ds_test = SpanishWritersDataset(args.dataset_root, "test", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
    elif args.dataset == "custom_splits":
        ds_train = SplitWritersDataset(args.dataset_root, "train", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        ds_val = SplitWritersDataset(args.dataset_root, "val", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        ds_test = SplitWritersDataset(args.dataset_root, "test", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
    elif args.dataset == "mixed":
        ds_train = ConcatDataset(
            [
                IAMWritersDataset(args.dataset_root, "train", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args),
                SpanishWritersDataset(args.dataset_root, "train", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args),
            ]
        )
        ds_val = ConcatDataset(
            [
                IAMWritersDataset(args.dataset_root, "val", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args),
                SpanishWritersDataset(args.dataset_root, "val", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args),
            ]
        )
        ds_test = ConcatDataset(
            [
                IAMWritersDataset(args.dataset_root, "test", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args),
                SpanishWritersDataset(args.dataset_root, "test", "word", fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args),
            ]
        )
    else:
        raise ValueError("--dataset must be 'iam', 'spanish', 'custom_splits', or 'mixed'")

    def seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    def mk_loader(ds, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=args.micro_batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            drop_last=shuffle,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    return mk_loader(ds_train, True), mk_loader(ds_val, False), mk_loader(ds_test, False)


@torch.no_grad()
def encode_style(style_extractor: nn.Module, style_images: torch.Tensor) -> torch.Tensor:
    # style_images: [B,S,3,64,256]
    b, s, c, h, w = style_images.shape
    feats = style_extractor(style_images.reshape(b * s, c, h, w))
    return feats.reshape(b, s, -1)


def compute_loss(
    *,
    unet: nn.Module,
    vae: nn.Module | None,
    tokenizer: CanineTokenizer,
    train_scheduler: DDPMScheduler,
    style_extractor: nn.Module,
    batch: Batch,
    device: torch.device,
    args,
):
    images = batch.images
    style_images = batch.style_images

    if args.text_dropout_prob > 0 and random.random() < args.text_dropout_prob:
        texts = [""] * len(batch.texts)
    else:
        texts = batch.texts

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=args.max_text_len,
    ).to(device)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            text_ctx = args._text_encoder(**tokenized).last_hidden_state

    if args.latent:
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * args.vae_scale_factor
        model_input = latents
    else:
        model_input = images

    noise = torch.randn_like(model_input)
    t = torch.randint(
        0,
        train_scheduler.config.num_train_timesteps,
        (model_input.shape[0],),
        device=device,
        dtype=torch.long,
    )
    noisy = train_scheduler.add_noise(model_input, noise, t)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
        style_features = encode_style(style_extractor, style_images)
    if args.style_dropout_prob > 0 and random.random() < args.style_dropout_prob:
        style_features = torch.zeros_like(style_features)

    pred = unet(noisy, timesteps=t, context=text_ctx, y=batch.writer_ids, style_extractor=style_features)
    loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")
    return loss


@torch.no_grad()
def evaluate(
    *,
    ema_unet: nn.Module,
    vae: nn.Module | None,
    tokenizer: CanineTokenizer,
    train_scheduler: DDPMScheduler,
    style_extractor: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args,
):
    ema_unet.eval()
    losses = []
    for data in loader:
        batch = to_batch(data, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            loss = compute_loss(
                unet=ema_unet,
                vae=vae,
                tokenizer=tokenizer,
                train_scheduler=train_scheduler,
                style_extractor=style_extractor,
                batch=batch,
                device=device,
                args=args,
            )
        losses.append(float(loss.item()))
    ema_unet.train()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def sample_preview(
    *,
    ema_unet: nn.Module,
    vae: nn.Module,
    tokenizer: CanineTokenizer,
    sample_scheduler: DDIMScheduler,
    style_extractor: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args,
    out_path: str,
):
    ema_unet.eval()
    data = next(iter(loader))
    batch = to_batch(data, device)

    n = min(args.preview_n, batch.images.shape[0])
    texts = batch.texts[:n]
    style_images = batch.style_images[:n]

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=args.max_text_len,
    ).to(device)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            text_ctx = args._text_encoder(**tokenized).last_hidden_state
    uncond = None
    uncond_ctx = None
    if args.cfg_scale != 1.0:
        uncond = tokenizer(
            [""] * n,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=args.max_text_len,
        ).to(device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                uncond_ctx = args._text_encoder(**uncond).last_hidden_state

    style_features = encode_style(style_extractor, style_images)

    latents = torch.randn((n, 4, 64 // 8, 256 // 8), device=device)
    for step_t in sample_scheduler.timesteps:
        t = torch.full((n,), int(step_t.item()), device=device, dtype=torch.long)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            if uncond_ctx is not None:
                eps_c = ema_unet(latents, timesteps=t, context=text_ctx, y=batch.writer_ids[:n], style_extractor=style_features)
                eps_u = ema_unet(latents, timesteps=t, context=uncond_ctx, y=batch.writer_ids[:n], style_extractor=style_features)
                eps = eps_u + args.cfg_scale * (eps_c - eps_u)
            else:
                eps = ema_unet(latents, timesteps=t, context=text_ctx, y=batch.writer_ids[:n], style_extractor=style_features)
        latents = sample_scheduler.step(eps, step_t, latents).prev_sample

    latents = latents / args.vae_scale_factor
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
        images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    save_grid(images.float().cpu(), out_path, nrow=min(4, n))
    ema_unet.train()


def save_checkpoint(save_path: str, unet: nn.Module, ema_unet: nn.Module, optimizer, scaler, epoch: int, global_step: int):
    ckpt = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "unet": unet.state_dict(),
        "ema_unet": ema_unet.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(ckpt, os.path.join(save_path, "models", "ckpt.pt"))
    # Compatibility artifacts (used by older sampling scripts).
    torch.save(unet.state_dict(), os.path.join(save_path, "models", "unet.pt"))
    torch.save(ema_unet.state_dict(), os.path.join(save_path, "models", "ema_unet.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_path, "models", "optim.pt"))


def load_checkpoint(path: str, unet: nn.Module, ema_unet: nn.Module, optimizer, scaler, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    unet_state = dict(ckpt["unet"])
    ema_state = dict(ckpt["ema_unet"])

    for key in ("label_emb.weight",):
        if key in unet_state and key in unet.state_dict():
            if unet_state[key].shape != unet.state_dict()[key].shape:
                print(f"Skipping incompatible tensor for {key}: {tuple(unet_state[key].shape)} -> {tuple(unet.state_dict()[key].shape)}")
                del unet_state[key]
        if key in ema_state and key in ema_unet.state_dict():
            if ema_state[key].shape != ema_unet.state_dict()[key].shape:
                del ema_state[key]

    unet.load_state_dict(unet_state, strict=False)
    ema_unet.load_state_dict(ema_state, strict=False)

    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as e:
        print(f"Skipping optimizer state reload: {type(e).__name__}: {e}")

    if scaler is not None and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"Skipping scaler state reload: {type(e).__name__}: {e}")

    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="custom_splits", help="custom_splits|spanish|iam|mixed")
    parser.add_argument("--dataset_root", type=str, default="./dataset")
    parser.add_argument("--save_path", type=str, default="./diffusionpen_colab_runs")
    parser.add_argument("--train_split", type=str, default="./splits/train.txt")
    parser.add_argument("--val_split", type=str, default="./splits/val.txt")
    parser.add_argument("--test_split", type=str, default="./splits/test.txt")
    parser.add_argument("--writer_map_path", type=str, default="./writers_dict_train.json")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Per-step batch size (fits in VRAM).")
    parser.add_argument("--target_batch_size", type=int, default=32, help="Effective batch size via grad accumulation.")
    parser.add_argument("--auto_micro_batch", action="store_true", default=True, help="Auto-reduce micro-batch on CUDA OOM.")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--max_text_len", type=int, default=64)
    parser.add_argument("--text_dropout_prob", type=float, default=0.1)
    parser.add_argument("--style_dropout_prob", type=float, default=0.1)
    parser.add_argument("--cfg_scale", type=float, default=1.0)

    parser.add_argument("--latent", action="store_true", default=True)
    parser.add_argument("--stable_dif_path", type=str, default="./stable-diffusion-v1-5")
    parser.add_argument("--vae_scale_factor", type=float, default=0.18215)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--sample_every_epochs", type=int, default=5)
    parser.add_argument("--preview_n", type=int, default=8)
    parser.add_argument("--num_style_samples", type=int, default=5)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_warmup_steps", type=int, default=500)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--save_every_epochs", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None, help="Path to ckpt.pt")
    parser.add_argument("--log_every_steps", type=int, default=50)

    # UNet
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--emb_dim", type=int, default=320)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--grad_checkpoint", action="store_true", default=True)
    # Kept for compatibility with the original UNet implementation.
    parser.add_argument("--interpolation", action="store_true", default=False)
    parser.add_argument("--mix_rate", type=float, default=None)

    # Style encoder
    parser.add_argument("--style_path", type=str, default="./style_models/mixed_spanish_mobilenetv2_100.pth")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    ensure_dirs(args.save_path)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, val_loader, test_loader = build_dataloaders(args, transform)

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c").to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    # stash for helper fns (avoid threading through every call)
    args._text_encoder = text_encoder

    # Style encoder (frozen)
    style_extractor = ImageEncoder(model_name="mobilenetv2_100", num_classes=0, pretrained=True, trainable=True)
    try:
        state_dict = torch.load(args.style_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(args.style_path, map_location=device)
    model_dict = style_extractor.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(state_dict)
    style_extractor.load_state_dict(model_dict)
    style_extractor = style_extractor.to(device)
    style_extractor.eval()
    style_extractor.requires_grad_(False)

    # Noise schedulers
    train_scheduler, sample_scheduler = build_schedulers(args.stable_dif_path, args.sampling_steps)

    # VAE (frozen)
    vae = None
    if args.latent:
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(device)
        vae.eval()
        vae.requires_grad_(False)

    # UNet
    # Note: num_classes is only used if style_extractor is None; we still set it for completeness.
    num_classes = int(getattr(train_loader.dataset, "wclasses", 1))
    unet = UNetModel(
        image_size=(64, 256),
        in_channels=args.channels,
        model_channels=args.emb_dim,
        out_channels=args.channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=args.num_heads,
        num_classes=num_classes,
        context_dim=args.emb_dim,
        vocab_size=0,
        text_encoder=None,
        use_checkpoint=args.grad_checkpoint,
        args=args,
    ).to(device)
    unet.train()

    ema_unet = UNetModel(
        image_size=(64, 256),
        in_channels=args.channels,
        model_channels=args.emb_dim,
        out_channels=args.channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=args.num_heads,
        num_classes=num_classes,
        context_dim=args.emb_dim,
        vocab_size=0,
        text_encoder=None,
        use_checkpoint=args.grad_checkpoint,
        args=args,
    ).to(device)
    ema_unet.load_state_dict(unet.state_dict(), strict=True)
    ema_unet.eval()
    ema_unet.requires_grad_(False)

    optimizer = optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    ema = EMA(decay=args.ema_decay, warmup_steps=args.ema_warmup_steps)

    def _try_one_step() -> None:
        data = next(iter(train_loader))
        batch = to_batch(data, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.amp and device.type == "cuda")):
            loss = compute_loss(
                unet=unet,
                vae=vae,
                tokenizer=tokenizer,
                train_scheduler=train_scheduler,
                style_extractor=style_extractor,
                batch=batch,
                device=device,
                args=args,
            )
        scaler.scale(loss).backward()
        optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda" and args.auto_micro_batch:
        # If we OOM on Colab/T4, auto-reduce the micro-batch and rebuild loaders.
        while True:
            try:
                _try_one_step()
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                new_bs = max(1, args.micro_batch_size // 2)
                if new_bs == args.micro_batch_size:
                    raise
                print(f"CUDA OOM at micro_batch_size={args.micro_batch_size}; retrying with micro_batch_size={new_bs}")
                args.micro_batch_size = new_bs
                train_loader, val_loader, test_loader = build_dataloaders(args, transform)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                new_bs = max(1, args.micro_batch_size // 2)
                if new_bs == args.micro_batch_size:
                    raise
                print(f"CUDA OOM at micro_batch_size={args.micro_batch_size}; retrying with micro_batch_size={new_bs}")
                args.micro_batch_size = new_bs
                train_loader, val_loader, test_loader = build_dataloaders(args, transform)

    grad_accum_steps = max(1, int(np.ceil(args.target_batch_size / args.micro_batch_size)))
    effective_bsz = args.micro_batch_size * grad_accum_steps
    print(f"micro_batch_size={args.micro_batch_size} grad_accum_steps={grad_accum_steps} effective_batch_size={effective_bsz}")

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(args.resume, unet, ema_unet, optimizer, scaler, device)
        print(f"Resumed from {args.resume}: start_epoch={start_epoch} global_step={global_step}")

    log_path = os.path.join(args.save_path, "logs", "train_log.jsonl")

    optimizer.zero_grad(set_to_none=True)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(start_epoch, args.epochs):
        unet.train()
        t0 = time.time()
        running = []

        for step, data in enumerate(train_loader):
            batch = to_batch(data, device)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.amp and device.type == "cuda")):
                    loss = compute_loss(
                        unet=unet,
                        vae=vae,
                        tokenizer=tokenizer,
                        train_scheduler=train_scheduler,
                        style_extractor=style_extractor,
                        batch=batch,
                        device=device,
                        args=args,
                    )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if device.type != "cuda" or "out of memory" not in str(e).lower():
                    raise
                print("CUDA OOM during training step; skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            if not torch.isfinite(loss):
                print(f"Non-finite loss at step={global_step}: {loss.item()}. Skipping step.")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            loss_to_backprop = loss / grad_accum_steps
            scaler.scale(loss_to_backprop).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(ema_unet, unet)
                global_step += 1

                running.append(float(loss.item()))
                if global_step % args.log_every_steps == 0:
                    msg = f"epoch={epoch} step={global_step} loss={np.mean(running[-args.log_every_steps:]):.6f}"
                    print(msg)

        train_loss = float(np.mean(running)) if running else float("nan")
        val_loss = evaluate(
            ema_unet=ema_unet,
            vae=vae,
            tokenizer=tokenizer,
            train_scheduler=train_scheduler,
            style_extractor=style_extractor,
            loader=val_loader,
            device=device,
            args=args,
        )

        record = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "effective_batch_size": int(effective_bsz),
            "micro_batch_size": int(args.micro_batch_size),
            "grad_accum_steps": int(grad_accum_steps),
            "seconds": float(time.time() - t0),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={record['seconds']:.1f}s")

        save_checkpoint(args.save_path, unet, ema_unet, optimizer, scaler, epoch, global_step)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(ema_unet.state_dict(), os.path.join(args.save_path, "models", "best_ema_unet.pt"))
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()

        if args.save_every_epochs > 0 and ((epoch + 1) % args.save_every_epochs == 0):
            torch.save(
                {
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "unet": unet.state_dict(),
                    "ema_unet": ema_unet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.save_path, "models", f"ckpt_epoch_{epoch + 1:04d}.pt"),
            )

        if args.latent and (epoch % args.sample_every_epochs == 0 or epoch == args.epochs - 1):
            out_img = os.path.join(args.save_path, "images", f"preview_val_epoch_{epoch:04d}.png")
            sample_preview(
                ema_unet=ema_unet,
                vae=vae,
                tokenizer=tokenizer,
                sample_scheduler=sample_scheduler,
                style_extractor=style_extractor,
                loader=val_loader,
                device=device,
                args=args,
                out_path=out_img,
            )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping at epoch={epoch} with best_val_loss={best_val_loss:.6f}")
            break

    # Final test loss on EMA weights (no sampling).
    test_loss = evaluate(
        ema_unet=ema_unet,
        vae=vae,
        tokenizer=tokenizer,
        train_scheduler=train_scheduler,
        style_extractor=style_extractor,
        loader=test_loader,
        device=device,
        args=args,
    )
    print(f"Final test_loss={test_loss:.6f}")


if __name__ == "__main__":
    main()
