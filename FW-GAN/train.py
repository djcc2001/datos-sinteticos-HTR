import os
from datetime import datetime
import argparse
from pathlib import Path

# Reduce CUDA allocator fragmentation (must be set before importing torch).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from lib.utils import yaml2config
from networks import get_model
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter


def infer_resume_logdir(ckpt_path):
    if not ckpt_path:
        return None
    ckpt = Path(ckpt_path).resolve()
    if ckpt.name.endswith(".pth") and ckpt.parent.name == "ckpts":
        return str(ckpt.parent.parent)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./configs/fw_gan_global_es.yml",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--ckpt",
        nargs="?",
        type=str,
        default=None,
        help="Checkpoint a cargar para reanudar o hacer fine-tuning. Sobrescribe cfg.ckpt.",
    )
    parser.add_argument(
        "--reset_epoch",
        action="store_true",
        help="Al cargar checkpoint, reinicia el contador de épocas a 1 para fine-tuning.",
    )

    args = parser.parse_args()
    print(f"Config file: {args.config}")

    cfg = yaml2config(args.config)
    if args.ckpt:
        cfg["ckpt"] = args.ckpt
    resume_logdir = infer_resume_logdir(getattr(cfg, "ckpt", None))
    if resume_logdir and os.path.isdir(resume_logdir):
        logdir = resume_logdir
        print(f"[INFO] Reanudando en el mismo run: {logdir}")
    else:
        run_id = datetime.strftime(datetime.now(), '%m-%d-%H-%M')
        logdir = os.path.join("runs", os.path.basename(args.config)[:-4] + '-' + str(run_id))
    print(logdir)

    # Device selection (single GPU).
    try:
        if getattr(cfg, 'device', 'cuda') == 'cuda' and torch.cuda.is_available():
            cfg['device'] = 'cuda'
            print(f"[INFO] Usando GPU: {torch.cuda.get_device_name(0)}")
        else:
            cfg['device'] = 'cpu'
            print("[INFO] CUDA no disponible. Usando CPU.")
    except Exception:
        cfg['device'] = 'cpu'
        print("[INFO] Usando CPU.")

    # Performance defaults for CUDA (safe for training stability).
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model = get_model(cfg.model)(cfg, logdir)

    # Check and load checkpoint
    epoch_done = 1
    if cfg.ckpt and os.path.exists(cfg.ckpt):
        print(f"Loading checkpoint from {cfg.ckpt}")
        loaded_epoch = model.load(cfg.ckpt, cfg.device)
        reset_epoch = bool(args.reset_epoch) or bool(getattr(cfg.training, "reset_epoch_on_load", False))
        if reset_epoch:
            epoch_done = 1
            print(f"[INFO] Checkpoint cargado para fine-tuning. Reiniciando contador de épocas desde {epoch_done} (ckpt epoch={loaded_epoch}).")
        else:
            # Continue from the *next* epoch after the checkpoint epoch.
            try:
                epoch_done = int(loaded_epoch) + 1
            except Exception:
                epoch_done = 1
            print(f"[INFO] Reanudando desde epoch {epoch_done} (ckpt epoch={loaded_epoch}).")
    else:
        print("No valid checkpoint found, starting from scratch.")

    model.train(epoch_done=epoch_done)
