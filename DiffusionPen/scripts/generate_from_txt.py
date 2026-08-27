import os
import sys
import argparse
import random
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineModel, CanineTokenizer

# Ensure project-local modules are imported instead of similarly named site-packages.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from unet import UNetModel
from feature_extractor import ImageEncoder
from utils.auxilary_functions import image_resize_PIL, centered_PIL
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_image(img, target_h=64, target_w=256):
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.height != target_h:
        img = image_resize_PIL(img, height=target_h)
    if img.width > target_w:
        img = image_resize_PIL(img, width=target_w)
    img = centered_PIL(img, (target_h, target_w), border_value=255.0)
    return img


def load_writer_map(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Writer map not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        wr_dict = json.load(f)
    return {int(k): int(v) for k, v in wr_dict.items()}


def infer_num_classes_from_checkpoint(checkpoint_path, device):
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "unet" in state:
        state = state["unet"]
    if "label_emb.weight" in state:
        return int(state["label_emb.weight"].shape[0])
    if "module.label_emb.weight" in state:
        return int(state["module.label_emb.weight"].shape[0])
    for key, tensor in state.items():
        if key.endswith("label_emb.weight"):
            return int(tensor.shape[0])
    raise KeyError("Could not infer num_classes from checkpoint (missing label_emb.weight).")


def resolve_writer_map_path(requested_path, expected_num_classes):
    candidate_paths = [requested_path]
    repo_root = PROJECT_ROOT
    candidate_paths.extend([
        os.path.join(repo_root, "writers_dict_train.json"),
        os.path.join(repo_root, "writers_dict_spanish_train.json"),
    ])

    seen = set()
    for path in candidate_paths:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        if not os.path.isfile(norm):
            continue
        wm = load_writer_map(norm)
        if len(wm) == expected_num_classes:
            return norm, wm

    raise ValueError(
        f"No writer_map compatible with checkpoint classes ({expected_num_classes}). "
        f"Checked: {', '.join(seen)}"
    )


def select_writer_id(writer_arg, writer_map):
    reverse_map = {v: k for k, v in writer_map.items()}
    if writer_arg == "random":
        return int(random.choice(list(reverse_map.keys())))
    try:
        writer_val = int(writer_arg)
    except ValueError:
        raise ValueError("--writer must be 'random' or an integer")
    if writer_val in writer_map:
        return int(writer_map[writer_val])
    if writer_val in reverse_map:
        return writer_val
    raise ValueError(f"Writer {writer_val} not found in writer map")


def load_style_images(dataset_root, writer_idx, writer_map, num_styles, transform):
    reverse_map = {v: k for k, v in writer_map.items()}
    writer_id = reverse_map[writer_idx]
    writer_dir = os.path.join(dataset_root, str(writer_id))
    labels_path = os.path.join(writer_dir, "labels.txt")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"labels.txt not found for writer {writer_id}")

    with open(labels_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split(maxsplit=1)[0] for line in f if line.strip()]
    if len(lines) == 0:
        raise ValueError(f"No samples found for writer {writer_id}")
    if len(lines) >= num_styles:
        chosen = random.sample(lines, k=num_styles)
    else:
        chosen = random.choices(lines, k=num_styles)

    style_tensors = []
    for fname in chosen:
        img_path = os.path.join(writer_dir, fname)
        img = Image.open(img_path)
        img = normalize_image(img)
        style_tensors.append(transform(img))
    return style_tensors


def load_style_images_from_writer_folder(dataset_root, writer_id, num_styles, transform):
    writer_dir = os.path.join(dataset_root, str(writer_id))
    labels_path = os.path.join(writer_dir, "labels.txt")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"labels.txt not found for writer folder {writer_id}")

    with open(labels_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split(maxsplit=1)[0] for line in f if line.strip()]
    if len(lines) == 0:
        raise ValueError(f"No samples found for writer folder {writer_id}")
    if len(lines) >= num_styles:
        chosen = random.sample(lines, k=num_styles)
    else:
        chosen = random.choices(lines, k=num_styles)

    style_tensors = []
    for fname in chosen:
        img_path = os.path.join(writer_dir, fname)
        img = Image.open(img_path)
        img = normalize_image(img)
        style_tensors.append(transform(img))
    return style_tensors


def load_style_images_from_manifest(manifest_path, num_styles, transform):
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 1:
                continue
            rows.append(parts[0])
    if len(rows) == 0:
        raise ValueError(f"No style rows found in manifest {manifest_path}")
    if len(rows) >= num_styles:
        chosen = random.sample(rows, k=num_styles)
    else:
        chosen = random.choices(rows, k=num_styles)

    style_tensors = []
    for img_path in chosen:
        img = Image.open(img_path)
        img = normalize_image(img)
        style_tensors.append(transform(img))
    return style_tensors


def load_checkpoint(model, checkpoint_path, device):
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "unet" in state:
        state = state["unet"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate handwritten words from a .txt file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained UNet checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input .txt file (one word per line)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for generated images")
    parser.add_argument("--dataset_root", type=str, default="./dataset")
    parser.add_argument("--writer_map", type=str, default="./writers_dict_spanish_train.json")
    parser.add_argument("--writer", type=str, default="random", help="Writer id or 'random'")
    parser.add_argument("--style_manifest", type=str, default=None, help="Optional txt manifest with image_path text rows for explicit few-shot style refs.")
    parser.add_argument("--num_style_samples", type=int, default=5)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--stable_dif_path", type=str, default="./stable-diffusion-v1-5")
    parser.add_argument("--style_path", type=str, default="./style_models/mixed_spanish_mobilenetv2_100.pth")
    parser.add_argument("--max_text_len", type=int, default=64)
    parser.add_argument("--color", type=bool, default=True)
    # Keep compatibility with UNetModel constructor used in training.
    parser.add_argument("--interpolation", type=bool, default=False)
    parser.add_argument("--mix_rate", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    expected_num_classes = infer_num_classes_from_checkpoint(args.checkpoint, device)
    writer_map = None
    writer_map_path = None
    writer_idx = 0
    writer_folder_id = None
    try:
        writer_map_path, writer_map = resolve_writer_map_path(args.writer_map, expected_num_classes)
        print(f"Using writer map: {writer_map_path} (writers={len(writer_map)})")
        writer_idx = select_writer_id(args.writer, writer_map)
    except Exception as e:
        # Mixed training may not have a meaningful writer embedding table. In that case we can still
        # condition on style images from a writer folder directly.
        if args.writer == "random":
            raise ValueError("When no compatible --writer_map is found, --writer must be an integer writer folder id.") from e
        writer_folder_id = int(args.writer)
        writer_idx = 0
        print(f"Warning: no compatible writer_map found ({type(e).__name__}); using writer folder id={writer_folder_id} directly.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c").to(device)
    text_encoder.eval()

    unet = UNetModel(
        image_size=(64, 256),
        in_channels=4,
        model_channels=320,
        out_channels=4,
        num_res_blocks=1,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=4,
        num_classes=expected_num_classes,
        context_dim=320,
        vocab_size=0,
        text_encoder=text_encoder,
        args=args,
    ).to(device)
    unet.eval()
    load_checkpoint(unet, args.checkpoint, device)

    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(device)
    vae.eval()
    vae.requires_grad_(False)

    scheduler = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    scheduler.set_timesteps(args.sampling_steps)

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

    if args.style_manifest is not None:
        style_tensors = load_style_images_from_manifest(args.style_manifest, args.num_style_samples, transform)
    elif writer_map is not None:
        style_tensors = load_style_images(args.dataset_root, writer_idx, writer_map, args.num_style_samples, transform)
    else:
        style_tensors = load_style_images_from_writer_folder(args.dataset_root, writer_folder_id, args.num_style_samples, transform)
    style_images = torch.stack(style_tensors).to(device)
    style_features = style_extractor(style_images).unsqueeze(0)

    with open(args.input, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    labels_path = os.path.join(args.output, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as labels_file:
        for idx, word in enumerate(words, start=1):
            text_features = tokenizer([word], padding="max_length", truncation=True, return_tensors="pt", max_length=args.max_text_len).to(device)
            with torch.inference_mode():
                text_ctx = text_encoder(**text_features).last_hidden_state

            uncond_features = None
            uncond_ctx = None
            if args.cfg_scale != 1.0:
                uncond_features = tokenizer([""], padding="max_length", truncation=True, return_tensors="pt", max_length=args.max_text_len).to(device)
                with torch.inference_mode():
                    uncond_ctx = text_encoder(**uncond_features).last_hidden_state

            x = torch.randn((1, 4, 64 // 8, 256 // 8)).to(device)
            labels = torch.tensor([writer_idx]).long().to(device)

            for time in scheduler.timesteps:
                t = torch.tensor([time.item()], device=device, dtype=torch.long)
                with torch.no_grad():
                    if args.cfg_scale != 1.0 and uncond_ctx is not None:
                        cond = unet(x, t, text_ctx, labels, style_extractor=style_features)
                        uncond = unet(x, t, uncond_ctx, labels, style_extractor=style_features)
                        noisy_residual = uncond + args.cfg_scale * (cond - uncond)
                    else:
                        noisy_residual = unet(x, t, text_ctx, labels, style_extractor=style_features)
                x = scheduler.step(noisy_residual, time, x).prev_sample

            latents = 1 / 0.18215 * x
            image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.squeeze(0).cpu()

            pil = transforms.ToPILImage()(image)
            if not args.color:
                pil = pil.convert("L")

            file_name = f"{idx:04d}.png"
            out_path = os.path.join(args.output, file_name)
            pil.save(out_path)
            labels_file.write(f"{file_name} {word}\n")


if __name__ == "__main__":
    main()
