import os
import cv2
import random
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from munch import Munch
from itertools import chain
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.nn import CTCLoss, CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from fid_kid.fid_kid import calculate_kid_fid
from networks.utils import _info, set_requires_grad, get_scheduler, idx_to_words, words_to_images, rand_clip
from networks.BigGAN_networks import Generator, Discriminator, HFDiscriminator
from networks.module import Recognizer, WriterIdentifier, StyleEncoder, SharedBackbone
from lib.datasets import get_dataset, get_collect_fn, Hdf5Dataset, resolve_split_file
from lib.alphabet import (
    strLabelConverter,
    get_lexicon,
    get_true_alphabet,
    Alphabets,
    get_alphabet_stats,
    validate_split_alphabet,
)
from lib.utils import draw_image, get_logger, AverageMeterManager, option_to_string, pad
from networks.rand_dist import prepare_z_dist, prepare_y_dist
from networks.loss import FDL_loss

# wandb.login()  # Use environment variable or config file
# wandb.init(project="FW-GAN", name="FW-GAN-training", resume="allow", sync_tensorboard=True)


class BaseModel(object):
    def __init__(self, opt, log_root='./kaggle/working/'):
        self.opt = opt
        self.device = torch.device(opt.device)
        self.models = Munch()
        self.models_ema = Munch()
        self.log_root = log_root
        self.logger = None
        self.writer = None
        alphabet_key = 'custom'
        #alphabet_key = 'rimes_word' if opt.dataset.startswith('rimes') else 'all'
        self.alphabet = Alphabets[alphabet_key]
        self.label_converter = strLabelConverter(alphabet_key)
        self.collect_fn = get_collect_fn(opt.training.sort_input)
        self._seed_everything(int(getattr(opt, "seed", 123456)))
        self.frozen_modules = set(str(name) for name in getattr(self.opt.training, "freeze_modules", []) or [])
        self._validate_active_splits()
        # Config hygiene (do NOT edit config files; fix in-memory only):
        # Some configs mistakenly point ckpt_dir to a checkpoint file path.
        try:
            ckpt_dir = getattr(self.opt.training, "ckpt_dir", None)
            if isinstance(ckpt_dir, str) and ckpt_dir.endswith(".pth"):
                self.opt.training.ckpt_dir = "ckpts"
        except Exception:
            pass

    @staticmethod
    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    @staticmethod
    def _seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_dataset(self, split=None, section="training", dset_name=None, split_file=None):
        section_cfg = getattr(self.opt, section, Munch())
        dset_name = dset_name or getattr(section_cfg, "dset_name", None) or self.opt.dataset
        split_name = split or getattr(section_cfg, "dset_split", None)
        split_file = split_file or getattr(section_cfg, "split_file", None)
        return get_dataset(
            dset_name,
            split_name,
            label_converter=self.label_converter,
            split_file=split_file,
        )

    def _validate_active_splits(self):
        alphabet_stats = get_alphabet_stats("custom")
        if alphabet_stats["has_duplicates"]:
            raise RuntimeError("El alfabeto activo contiene caracteres duplicados; corrígelo antes de entrenar.")

        for section in ("training", "valid", "test"):
            section_cfg = getattr(self.opt, section, None)
            if section_cfg is None:
                continue
            dset_name = getattr(section_cfg, "dset_name", None) or self.opt.dataset
            split_name = getattr(section_cfg, "dset_split", None)
            split_file = getattr(section_cfg, "split_file", None)
            resolved = resolve_split_file(dset_name, split_name, split_file=split_file)
            if resolved is None or not os.path.exists(resolved):
                continue
            stats = validate_split_alphabet(resolved, alphabet_key="custom")
            if stats["unknown"]:
                unknown_desc = ", ".join(f"{repr(ch)}x{count}" for ch, count in sorted(stats["unknown"].items()))
                raise ValueError(f"Split inválido para {section}: {resolved}. Caracteres fuera de alfabeto: {unknown_desc}")

    def _set_modules_requires_grad(self, module_names, requires_grad):
        for name in module_names:
            if name not in self.models:
                continue
            effective_requires_grad = bool(requires_grad) and (name not in self.frozen_modules)
            set_requires_grad(self.models[name], effective_requires_grad)

    def _make_loader(self, dataset, batch_size, shuffle, drop_last=False):
        num_workers = getattr(self.opt.training, "num_workers", 4)
        num_workers = 4 if num_workers is None else int(num_workers)
        generator = torch.Generator()
        generator.manual_seed(int(getattr(self.opt, "seed", 123456)))
        dl_kwargs = dict(
            collate_fn=self.collect_fn,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            worker_init_fn=self._seed_worker,
            generator=generator,
        )
        if num_workers > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = int(getattr(self.opt.training, "prefetch_factor", 2) or 2)
        return DataLoader(dataset, **dl_kwargs)

    def print(self, info):
        if self.logger is None:
            print(info)
        else:
            self.logger.info(info)

    def create_logger(self):
        if self.logger or self.writer:
            return

        os.makedirs(self.log_root, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_root)
        opt_str = option_to_string(self.opt)
        with open(os.path.join(self.log_root, 'config.txt'), 'w') as f:
            f.writelines(opt_str)
        self.logger = get_logger(self.log_root)

    def _reset_writer(self):
        try:
            if self.writer is not None:
                self.writer.close()
        except Exception:
            pass
        self.writer = None
        os.makedirs(self.log_root, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_root)

    def _safe_add_scalar(self, tag, value, step):
        if not self.writer:
            return
        try:
            os.makedirs(self.log_root, exist_ok=True)
            self.writer.add_scalar(tag, value, step)
        except Exception as e:
            self.print(f"[WARN] TensorBoard add_scalar falló ({tag}): {e}. Reintentando una vez.")
            try:
                self._reset_writer()
                self.writer.add_scalar(tag, value, step)
            except Exception as e2:
                self.print(f"[WARN] TensorBoard deshabilitado temporalmente: {e2}")
                try:
                    self.writer.close()
                except Exception:
                    pass
                self.writer = None

    def _safe_add_image(self, tag, image, step):
        if not self.writer:
            return
        try:
            os.makedirs(self.log_root, exist_ok=True)
            self.writer.add_image(tag, image, step)
        except Exception as e:
            self.print(f"[WARN] TensorBoard add_image falló ({tag}): {e}. Reintentando una vez.")
            try:
                self._reset_writer()
                self.writer.add_image(tag, image, step)
            except Exception as e2:
                self.print(f"[WARN] TensorBoard deshabilitado temporalmente: {e2}")
                try:
                    self.writer.close()
                except Exception:
                    pass
                self.writer = None

    def info(self, extra=None):
        self.print("RUNDIR: {}".format(self.log_root))
        opt_str = option_to_string(self.opt)
        self.print(opt_str)
        for model in self.models.values():
            self.print(_info(model, ret=True))
        if extra is not None:
            self.print(extra)
        self.print('=' * 20)

    def save(self, tag='best', epoch_done=0, **kwargs):
        ckpt = {}
        if len(self.models_ema.values()) == 0:
            for model in self.models.values():
                ckpt[type(model).__name__] = model.state_dict()
        else:
            for model in self.models_ema.values():
                ckpt[type(model).__name__] = model.state_dict()

        for key, val in kwargs.items():
            ckpt[key] = val

        ckpt['Epoch'] = epoch_done
        ckpt_save_path = os.path.join(self.log_root, self.opt.training.ckpt_dir, tag + '.pth')
        torch.save(ckpt, ckpt_save_path)

    def load(self, ckpt, map_location=None, modules=None):
        if modules is None:
            modules = []
        elif not isinstance(modules, list):
            modules = [modules]

        print('load checkpoint from ', ckpt)
        if map_location is None:
            ckpt = torch.load(ckpt)
        else:
            ckpt = torch.load(ckpt, map_location=map_location, weights_only = False)

        if len(modules) == 0:
            for model in self.models.values():
                model.load_state_dict(ckpt[type(model).__name__])
        else:
            for model in modules:
                model.load_state_dict(ckpt[type(model).__name__])

        return ckpt['Epoch']

    def set_mode(self, mode='eval'):
        for model in self.models.values():
            if mode == 'eval':
                model.eval()
            elif mode == 'train':
                model.train()
            else:
                raise NotImplementedError()

    def validate(self):
        yield NotImplementedError()

    def train(self):
        yield NotImplementedError()


class AdversarialModel(BaseModel):
    def __init__(self, opt, log_root='./kaggle/working/'):
        super(AdversarialModel, self).__init__(opt, log_root)

        device = self.device
        # Ensure config n_class matches the active alphabet (characters) and CTC (characters + blank).
        n_char = int(len(self.alphabet))
        try:
            if int(getattr(opt.training, "n_class", n_char)) != n_char:
                self.print(f"[WARN] training.n_class={opt.training.n_class} != len(alphabet)={n_char}; ajustando.")
            opt.training.n_class = n_char
        except Exception:
            pass
        for key in ("GenModel", "DiscModel", "HFDiscModel"):
            try:
                if int(getattr(getattr(opt, key), "n_class", n_char)) != n_char:
                    self.print(f"[WARN] {key}.n_class != len(alphabet); ajustando a {n_char}.")
                getattr(opt, key).n_class = n_char
            except Exception:
                pass
        try:
            expected_ctc = n_char + 1
            if int(getattr(opt.OcrModel, "n_class", expected_ctc)) != expected_ctc:
                self.print(f"[WARN] OcrModel.n_class={opt.OcrModel.n_class} != len(alphabet)+1={expected_ctc}; ajustando.")
            opt.OcrModel.n_class = expected_ctc
        except Exception:
            pass

        self.lexicon = get_lexicon(
            self.opt.training.lexicon,
            get_true_alphabet(opt.dataset),
            max_length=self.opt.training.max_word_len,
        )
        # Fallback: build lexicon directly from the explicit training split.
        if not self.lexicon:
            try:
                train_dset = self._get_dataset(section="training")
                self.lexicon = sorted({sample["label"] for sample in getattr(train_dset, "samples", []) if 1 <= len(sample["label"]) <= int(self.opt.training.max_word_len)})
            except Exception:
                pass
        if not self.lexicon:
            raise RuntimeError("Lexicon vacío: revisa training.lexicon o genera data/spanish_words.txt desde splits/train.txt.")
        self.max_valid_image_width = self.opt.char_width * self.opt.training.max_word_len
        self.noise_dim = self.opt.GenModel.style_dim - self.opt.EncModel.style_dim

        generator = Generator(**opt.GenModel).to(device)
        style_encoder = StyleEncoder(**opt.EncModel).to(device)
        # Ensure WriterIdentifier output covers writer IDs present in the active dataset.
        try:
            train_dset = self._get_dataset(section="training")
            if hasattr(train_dset, "samples"):
                max_wid = max(int(sample["writer_id"]) for sample in train_dset.samples)
                if getattr(opt.WidModel, "n_writer", 0) <= max_wid:
                    opt.WidModel.n_writer = max_wid + 1
        except Exception:
            pass
        writer_identifier = WriterIdentifier(**opt.WidModel).to(device)
        discriminator = Discriminator(**opt.DiscModel).to(device)
        hf_discriminator = HFDiscriminator(**opt.HFDiscModel).to(device)
        recognizer = Recognizer(**opt.OcrModel).to(device)
        shared_backbone = SharedBackbone(**opt.SharedBackbone).to(device)
        
        self.models = Munch(
            G=generator,
            D=discriminator,
            HF_D=hf_discriminator,
            R=recognizer,
            E=style_encoder,
            W=writer_identifier,
            S=shared_backbone
        )

        # OCR head uses (alphabet + blank). Convention here: blank is the last index.
        blank_idx = int(getattr(opt.OcrModel, "n_class", 0)) - 1
        if blank_idx < 0:
            blank_idx = 0
        self.ctc_blank_idx = blank_idx
        self.ctc_loss = CTCLoss(blank=self.ctc_blank_idx, zero_infinity=True, reduction='mean')
        self.classify_loss = CrossEntropyLoss()
        # FDL is expensive; expose safe defaults via config for low-VRAM GPUs.
        fdl_num_proj = int(getattr(opt.training, "fdl_num_proj", 256) or 256)
        fdl_upscale = int(getattr(opt.training, "fdl_upscale_factor", 2) or 2)
        fdl_chunk = int(getattr(opt.training, "fdl_chunk_size", 32) or 32)
        self.fdl_loss_fn = FDL_loss(
            backbone=self.models.S,
            num_proj=fdl_num_proj,
            upscale_factor=fdl_upscale,
            chunk_size=fdl_chunk,
        ).to(device)
        if self.frozen_modules:
            self.print(f"[INFO] Módulos congelados para fine-tuning: {sorted(self.frozen_modules)}")

    def train(self, epoch_done):
        self.create_logger()
        self.info()
        
        
        def KLloss(mu, logvar):
            return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        opt = self.opt

        # Optional torch.compile (disabled by default; dynamic shapes can be tricky).
        if bool(getattr(opt.training, "torch_compile", False)) and hasattr(torch, "compile"):
            try:
                for k in ["G", "D", "HF_D", "R", "E", "W", "S"]:
                    if k in self.models:
                        self.models[k] = torch.compile(self.models[k], mode="reduce-overhead", dynamic=True)
                self.print("[INFO] torch.compile habilitado.")
            except Exception as e:
                self.print(f"[WARN] torch.compile falló; continuando sin compile. Error: {e}")

        # Auto-adjust batch size on CUDA to avoid OOM while honoring batch_size=32 when possible.
        if self.device.type == "cuda":
            target_bs = int(getattr(opt.training, "batch_size", 32) or 32)
            bs = target_bs
            while bs >= 1:
                try:
                    probe_loader = DataLoader(
                        self._get_dataset(section="training"),
                        batch_size=bs,
                        shuffle=True,
                        collate_fn=self.collect_fn,
                        num_workers=0,
                        drop_last=True,
                    )
                    imgs, img_lens, lbs, lb_lens, wids = next(iter(probe_loader))
                    imgs, img_lens = imgs.to(self.device), img_lens.to(self.device)
                    with torch.no_grad():
                        z = prepare_z_dist(bs, opt.GenModel.style_dim, self.device, seed=self.opt.seed)
                        y = prepare_y_dist(bs, len(self.lexicon), self.device, seed=self.opt.seed)
                        z.sample_()
                        y.sample_()
                        sampled_words = idx_to_words(y, self.lexicon, self.opt.training.capitalize_ratio)
                        fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
                        fake_lbs, fake_lb_lens = fake_lbs.to(self.device), fake_lb_lens.to(self.device)
                        fake_imgs = self.models.G(z, fake_lbs, fake_lb_lens)
                        fake_img_lens = fake_lb_lens * self.opt.char_width
                        _ = self.models.D(fake_imgs, fake_img_lens, fake_lb_lens)
                        _ = self.models.R(fake_imgs)
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.print(f"[WARN] CUDA OOM con batch_size={bs}; probando batch_size={bs//2}.")
                        bs //= 2
                        torch.cuda.empty_cache()
                        continue
                    raise
            if bs < 1:
                raise RuntimeError("No se pudo encontrar batch_size válido en CUDA (OOM).")
            current_bs = int(getattr(opt.training, "batch_size", bs) or bs)
            if bs != current_bs:
                self.print(f"[INFO] batch_size ajustado automáticamente: {target_bs} -> {bs}")
                opt.training.batch_size = bs

        self.z = prepare_z_dist(opt.training.batch_size, opt.GenModel.style_dim, self.device,
                                seed=self.opt.seed)
        self.y = prepare_y_dist(opt.training.batch_size, len(self.lexicon), self.device, seed=self.opt.seed)

        self.eval_z = prepare_z_dist(opt.training.eval_batch_size, opt.GenModel.style_dim, self.device,
                                     seed=self.opt.seed)
        self.eval_y = prepare_y_dist(opt.training.eval_batch_size, len(self.lexicon), self.device,
                                     seed=self.opt.seed)

        self.train_loader = self._make_loader(
            self._get_dataset(section="training"),
            batch_size=opt.training.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.tst_loader = self._make_loader(
            self._get_dataset(section="valid"),
            batch_size=opt.training.eval_batch_size // 2,
            shuffle=True,
            drop_last=False,
        )

        self.tst_loader2 = self._make_loader(
            self._get_dataset(section="valid"),
            batch_size=opt.training.eval_batch_size // 2,
            shuffle=True,
            drop_last=False,
        )
        g_params = [p for p in chain(self.models.G.parameters(), self.models.E.parameters()) if p.requires_grad]
        d_params = [p for p in chain(
            self.models.D.parameters(),
            self.models.HF_D.parameters(),
            self.models.R.parameters(),
            self.models.W.parameters(),
            self.models.S.parameters(),
        ) if p.requires_grad]
        if not g_params:
            raise RuntimeError("No hay parámetros entrenables para el optimizador G.")
        if not d_params:
            raise RuntimeError("No hay parámetros entrenables para el optimizador D.")

        self.optimizers = Munch(
            G=torch.optim.Adam(g_params, lr=opt.training.lr, betas=(opt.training.adam_b1, opt.training.adam_b2)),
            D=torch.optim.Adam(d_params, lr=opt.training.lr, betas=(opt.training.adam_b1, opt.training.adam_b2)),
        )

        self.lr_schedulers = Munch(
            G=get_scheduler(self.optimizers.G, opt.training),
            D=get_scheduler(self.optimizers.D, opt.training)
        )

        self.averager_meters = AverageMeterManager(['g_total', 'd_total',
                                                    'adv_loss', 'adv_loss_hf', 'fake_disc_loss',
                                                    'real_disc_loss', 'hf_fake_disc_loss', 'hf_real_disc_loss', 'info_loss',
                                                    'fake_ctc_loss', 'real_ctc_loss',
                                                    'fake_wid_loss', 'real_wid_loss',
                                                    'kl_loss', 'fdl_loss', 'gp_ctc', 'gp_info', 'gp_wid', 'r1_pen'])
        device = self.device
        grad_clip = float(getattr(self.opt.training, "grad_clip_norm", 5.0))
        use_amp = bool(getattr(self.opt.training, "amp", False)) and (self.device.type == "cuda")
        use_hf_discriminator = bool(getattr(self.opt.training, "use_hf_discriminator", True))
        # torch.cuda.amp.* is deprecated in recent PyTorch; use torch.amp.* instead.
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast_ctx = torch.amp.autocast
        autocast_device = self.device.type

        def _reduce_lr(factor=0.5, min_lr=1e-6):
            for optim in self.optimizers.values():
                for pg in optim.param_groups:
                    pg["lr"] = max(float(pg["lr"]) * factor, min_lr)

        ctc_len_scale = int(getattr(self.models.R, "len_scale", 8))
        best_kid = np.inf
        best_monitor = None
        early_stop_bad_epochs = 0
        iter_count = 0
        epoch = int(epoch_done)
        max_epoch = int(self.opt.training.epochs)
        while epoch < max_epoch:
            restart_epoch = False
            for i, (imgs, img_lens, lbs, lb_lens, wids) in enumerate(self.train_loader):
                try:
                    #############################
                    # Prepare inputs & Network Forward
                    #############################
                    self.set_mode('train')
                    real_imgs = imgs.to(device, non_blocking=True)
                    real_img_lens = img_lens.to(device, non_blocking=True)
                    real_wids = wids.to(device, non_blocking=True)
                    real_lbs = lbs.to(device, non_blocking=True)
                    real_lb_lens = lb_lens.to(device, non_blocking=True)

                    #############################
                    # Optimizing Recognizer & Writer Identifier & Discriminator
                    #############################
                    self.optimizers.D.zero_grad(set_to_none=True)
                    self._set_modules_requires_grad(["G", "E"], False)
                    self._set_modules_requires_grad(["R", "D", "W", "S"], True)
                    self._set_modules_requires_grad(["HF_D"], use_hf_discriminator)

                    with autocast_ctx(autocast_device, enabled=use_amp):
                        # CTC loss on real
                        real_ctc = self.models.R(real_imgs)
                        real_ctc_lens = (real_img_lens // ctc_len_scale).clamp_min(1).clamp_max(real_ctc.size(0))
                        real_ctc_loss = self.ctc_loss(real_ctc, real_lbs, real_ctc_lens, real_lb_lens)
                        self.averager_meters.update('real_ctc_loss', real_ctc_loss.item())

                        # Writer ID on random clip
                        clip_imgs, clip_img_lens = rand_clip(real_imgs, real_img_lens)
                        real_wid_logits = self.models.W(clip_imgs, clip_img_lens, self.models.S)
                        real_wid_loss = self.classify_loss(real_wid_logits, real_wids)
                        self.averager_meters.update('real_wid_loss', real_wid_loss.item())

                    real_aux_loss = real_ctc_loss + real_wid_loss
                    real_aux_loss_for_backward = None
                    if real_ctc_loss.requires_grad and real_wid_loss.requires_grad:
                        real_aux_loss_for_backward = real_aux_loss
                    elif real_ctc_loss.requires_grad:
                        real_aux_loss_for_backward = real_ctc_loss
                    elif real_wid_loss.requires_grad:
                        real_aux_loss_for_backward = real_wid_loss

                    if torch.isfinite(real_aux_loss):
                        if real_aux_loss_for_backward is not None:
                            scaler.scale(real_aux_loss_for_backward).backward()
                    else:
                        self.print("[WARN] Pérdida real (CTC/WID) no finita; reduciendo LR y saltando step de D.")
                        _reduce_lr()
                        continue

                    del real_ctc
                    del real_wid_logits
                    del clip_imgs
                    del clip_img_lens

                    with torch.no_grad():
                        self.y.sample_()
                        sampled_words = idx_to_words(self.y, self.lexicon, self.opt.training.capitalize_ratio)
                        fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
                        fake_lbs = fake_lbs.to(device, non_blocking=True).detach()
                        fake_lb_lens = fake_lb_lens.to(device, non_blocking=True).detach()

                        self.z.sample_()
                        with autocast_ctx(autocast_device, enabled=use_amp):
                            fake_imgs = self.models.G(self.z, fake_lbs, fake_lb_lens)

                        with autocast_ctx(autocast_device, enabled=use_amp):
                            enc_styles, _, _ = self.models.E(real_imgs, real_img_lens, self.models.S, vae_mode=True)
                        noises = torch.randn(
                            (real_imgs.size(0), self.opt.GenModel.style_dim - self.opt.EncModel.style_dim),
                            device=device,
                            dtype=torch.float32,
                        )
                        enc_z = torch.cat([noises, enc_styles], dim=-1)
                        with autocast_ctx(autocast_device, enabled=use_amp):
                            style_imgs = self.models.G(enc_z, fake_lbs, fake_lb_lens)
                        fake_img_lens = fake_lb_lens * self.opt.char_width
                        style_img_lens = fake_lb_lens * self.opt.char_width

                    # Hinge losses on fake (two forwards to lower peak memory)
                    with autocast_ctx(autocast_device, enabled=use_amp):
                        fake_disc_a = self.models.D(fake_imgs.detach(), fake_img_lens, fake_lb_lens)
                        fake_disc_b = self.models.D(style_imgs.detach(), style_img_lens, fake_lb_lens)
                        fake_disc_loss = 0.5 * (torch.mean(F.relu(1.0 + fake_disc_a)) + torch.mean(F.relu(1.0 + fake_disc_b)))

                    # Optional R1 regularization (real images)
                    r1_pen = torch.tensor(0.0, device=device)
                    r1_gamma = float(getattr(self.opt.training, "r1_gamma", 0.0) or 0.0)
                    r1_every = int(getattr(self.opt.training, "r1_every", 16) or 16)
                    if r1_gamma > 0 and (iter_count % max(r1_every, 1) == 0):
                        # Keep R1 in fp32 for stability.
                        real_imgs_r1 = real_imgs.detach().float().requires_grad_(True)
                        with autocast_ctx(autocast_device, enabled=False):
                            real_disc_r1 = self.models.D(real_imgs_r1, real_img_lens, real_lb_lens)
                            hf_real_disc_r1 = self.models.HF_D(real_imgs_r1, real_img_lens, real_lb_lens) if use_hf_discriminator else 0.0
                        r1_grads = torch.autograd.grad(
                            outputs=(real_disc_r1.sum() + (hf_real_disc_r1.sum() if use_hf_discriminator else 0.0)),
                            inputs=real_imgs_r1,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        r1_pen = 0.5 * r1_gamma * r1_grads.pow(2).flatten(1).sum(1).mean()
                        real_disc = real_disc_r1
                        hf_real_disc = hf_real_disc_r1
                    else:
                        with autocast_ctx(autocast_device, enabled=use_amp):
                            real_disc = self.models.D(real_imgs, real_img_lens, real_lb_lens)
                            hf_real_disc = self.models.HF_D(real_imgs, real_img_lens, real_lb_lens) if use_hf_discriminator else None
                    with autocast_ctx(autocast_device, enabled=use_amp):
                        real_disc_loss = torch.mean(F.relu(1.0 - real_disc))

                        if use_hf_discriminator:
                            hf_fake_a = self.models.HF_D(fake_imgs.detach(), fake_img_lens, fake_lb_lens)
                            hf_fake_b = self.models.HF_D(style_imgs.detach(), style_img_lens, fake_lb_lens)
                            hf_fake_disc_loss = 0.5 * (torch.mean(F.relu(1.0 + hf_fake_a)) + torch.mean(F.relu(1.0 + hf_fake_b)))
                            hf_real_disc_loss = torch.mean(F.relu(1.0 - hf_real_disc))
                        else:
                            hf_fake_disc_loss = torch.tensor(0.0, device=device)
                            hf_real_disc_loss = torch.tensor(0.0, device=device)

                        disc_loss = (real_disc_loss + fake_disc_loss + hf_real_disc_loss + hf_fake_disc_loss) + r1_pen
                    self.averager_meters.update('r1_pen', float(r1_pen.detach().item()))
                    self.averager_meters.update('real_disc_loss', real_disc_loss.item())
                    self.averager_meters.update('fake_disc_loss', fake_disc_loss.item())
                    self.averager_meters.update('hf_real_disc_loss', hf_real_disc_loss.item())
                    self.averager_meters.update('hf_fake_disc_loss', hf_fake_disc_loss.item())

                    d_total = real_aux_loss.detach() + disc_loss.detach()
                    self.averager_meters.update('d_total', float(d_total.detach().item()))
                    if torch.isfinite(disc_loss):
                        scaler.scale(disc_loss).backward()
                        scaler.unscale_(self.optimizers.D)
                        torch.nn.utils.clip_grad_norm_(self.optimizers.D.param_groups[0]["params"], grad_clip)
                        scaler.step(self.optimizers.D)
                        scaler.update()
                    else:
                        self.print("[WARN] D loss no finita; reduciendo LR y saltando step.")
                        _reduce_lr()

                    #############################
                    # Optimizing Generator
                    #############################
                    if iter_count % self.opt.training.num_critic_train == 0:
                        self.optimizers.G.zero_grad(set_to_none=True)
                        self._set_modules_requires_grad(["D", "R", "W", "S"], False)
                        self._set_modules_requires_grad(["HF_D"], False)
                        self._set_modules_requires_grad(["G", "E"], True)

                        self.y.sample_()
                        sampled_words = idx_to_words(self.y, self.lexicon, self.opt.training.capitalize_ratio)
                        fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
                        fake_lbs = fake_lbs.to(device, non_blocking=True).detach()
                        fake_lb_lens = fake_lb_lens.to(device, non_blocking=True).detach()
                        fake_img_lens = fake_lb_lens * self.opt.char_width

                        self.z.sample_()
                        with autocast_ctx(autocast_device, enabled=use_amp):
                            fake_imgs = self.models.G(self.z, fake_lbs, fake_lb_lens)

                        with autocast_ctx(autocast_device, enabled=use_amp):
                            enc_styles, enc_mu, enc_logvar = self.models.E(real_imgs, real_img_lens, self.models.S, vae_mode=True)
                        noises = torch.randn(
                            (real_imgs.size(0), self.opt.GenModel.style_dim - self.opt.EncModel.style_dim),
                            device=device,
                            dtype=torch.float32,
                        )
                        enc_z = torch.cat([noises, enc_styles], dim=-1)
                        with autocast_ctx(autocast_device, enabled=use_amp):
                            style_imgs = self.models.G(enc_z, fake_lbs, fake_lb_lens)
                        style_img_lens = fake_lb_lens * self.opt.char_width

                        with autocast_ctx(autocast_device, enabled=use_amp):
                            recn_imgs = self.models.G(enc_z, real_lbs, real_lb_lens)

                        # Adversarial
                        with autocast_ctx(autocast_device, enabled=use_amp):
                            disc_fake = self.models.D(fake_imgs, fake_img_lens, fake_lb_lens)
                            disc_style = self.models.D(style_imgs, style_img_lens, fake_lb_lens)
                            adv_loss = -0.5 * (torch.mean(disc_fake) + torch.mean(disc_style))

                            if use_hf_discriminator:
                                hf_fake = self.models.HF_D(fake_imgs, fake_img_lens, fake_lb_lens)
                                hf_style = self.models.HF_D(style_imgs, style_img_lens, fake_lb_lens)
                                adv_loss_hf = -0.5 * (torch.mean(hf_fake) + torch.mean(hf_style))
                            else:
                                adv_loss_hf = torch.tensor(0.0, device=device)

                        # CTC
                        with autocast_ctx(autocast_device, enabled=use_amp):
                            ctc_fake = self.models.R(fake_imgs)
                            ctc_style = self.models.R(style_imgs)
                            ctc_lens_fake = (fake_img_lens // ctc_len_scale).clamp_min(1).clamp_max(ctc_fake.size(0))
                            ctc_lens_style = (style_img_lens // ctc_len_scale).clamp_min(1).clamp_max(ctc_style.size(0))
                            fake_ctc_loss = 0.5 * (
                                self.ctc_loss(ctc_fake, fake_lbs, ctc_lens_fake, fake_lb_lens)
                                + self.ctc_loss(ctc_style, fake_lbs, ctc_lens_style, fake_lb_lens)
                            )

                        # Info
                        with autocast_ctx(autocast_device, enabled=use_amp):
                            styles = self.models.E(fake_imgs, fake_img_lens, self.models.S)
                            info_loss = torch.mean(torch.abs(styles - self.z[:, -self.opt.EncModel.style_dim:].detach()))

                        # Writer-ID
                        with autocast_ctx(autocast_device, enabled=use_amp):
                            recn_wid_logits = self.models.W(style_imgs, style_img_lens, self.models.S)
                            fake_wid_loss = self.classify_loss(recn_wid_logits, real_wids)

                        # FDL (align widths to avoid shape mismatch in backbone/FFT)
                        fdl_loss = torch.tensor(0.0, device=device)
                        fdl_every = int(getattr(self.opt.training, "fdl_every", 1) or 1)
                        lambda_fdl = float(getattr(self.opt.training, "lambda_fdl", 1.0) or 1.0)
                        if lambda_fdl > 0 and (iter_count % max(fdl_every, 1) == 0):
                            w = min(int(real_imgs.size(-1)), int(recn_imgs.size(-1)))
                            if w >= 2:
                                with autocast_ctx(autocast_device, enabled=False):
                                    fdl_loss = self.fdl_loss_fn(real_imgs.float()[..., :w], recn_imgs.float()[..., :w])

                        # KL
                        kl_loss = KLloss(enc_mu, enc_logvar)

                        # Gradient balance (no 2nd-order graphs)
                        grad_adv_fake = torch.autograd.grad(adv_loss, fake_imgs, create_graph=False, retain_graph=True)[0]
                        grad_adv_style = torch.autograd.grad(adv_loss, style_imgs, create_graph=False, retain_graph=True)[0]
                        grad_ctc_fake = torch.autograd.grad(fake_ctc_loss, fake_imgs, create_graph=False, retain_graph=True)[0]
                        grad_ctc_style = torch.autograd.grad(fake_ctc_loss, style_imgs, create_graph=False, retain_graph=True)[0]
                        grad_fake_info = torch.autograd.grad(info_loss, fake_imgs, create_graph=False, retain_graph=True)[0]
                        grad_fake_wid = torch.autograd.grad(fake_wid_loss, style_imgs, create_graph=False, retain_graph=True)[0]

                        eps = 1e-8
                        std_grad_adv_all = torch.std(torch.cat([grad_adv_fake.reshape(-1), grad_adv_style.reshape(-1)]))
                        std_grad_adv_fake = torch.std(grad_adv_fake)
                        std_grad_adv_style = torch.std(grad_adv_style)
                        std_grad_ctc_all = torch.std(torch.cat([grad_ctc_fake.reshape(-1), grad_ctc_style.reshape(-1)]))

                        gp_ctc = (torch.div(std_grad_adv_all, std_grad_ctc_all + eps).detach() + 1).clamp_max(100)
                        gp_info = (torch.div(std_grad_adv_fake, torch.std(grad_fake_info) + eps).detach() + 1).clamp_max(50)
                        gp_wid = (torch.div(std_grad_adv_style, torch.std(grad_fake_wid) + eps).detach() + 1).clamp_max(10)

                        self.averager_meters.update('gp_ctc', gp_ctc.item())
                        self.averager_meters.update('gp_info', gp_info.item())
                        self.averager_meters.update('gp_wid', gp_wid.item())

                        g_loss = (2 * adv_loss + adv_loss_hf +
                                  gp_ctc * fake_ctc_loss +
                                  gp_info * info_loss +
                                  gp_wid * fake_wid_loss +
                                  (lambda_fdl * fdl_loss) +
                                  self.opt.training.lambda_kl * kl_loss)
                        self.averager_meters.update('g_total', float(g_loss.detach().item()))

                        if torch.isfinite(g_loss):
                            scaler.scale(g_loss).backward()
                            scaler.unscale_(self.optimizers.G)
                            self.averager_meters.update('adv_loss', adv_loss.item())
                            self.averager_meters.update('adv_loss_hf', adv_loss_hf.item())
                            self.averager_meters.update('fake_ctc_loss', fake_ctc_loss.item())
                            self.averager_meters.update('info_loss', info_loss.item())
                            self.averager_meters.update('fake_wid_loss', fake_wid_loss.item())
                            self.averager_meters.update('fdl_loss', fdl_loss.item())
                            self.averager_meters.update('kl_loss', kl_loss.item())
                            torch.nn.utils.clip_grad_norm_(self.optimizers.G.param_groups[0]["params"], grad_clip)
                            scaler.step(self.optimizers.G)
                            scaler.update()
                        else:
                            self.print("[WARN] G loss no finita; reduciendo LR y saltando step.")
                            _reduce_lr()

                    # periodic logging
                    if iter_count % self.opt.training.print_iter_val == 0:
                        meter_vals = self.averager_meters.eval_all()
                        self.averager_meters.reset_all()
                        info = "[%3d|%3d]-[%4d|%4d] G:%.4f G-HF:%.4f D-fake:%.4f D-real:%.4f " \
                               "HF-fake:%.4f HF-real:%.4f CTC-fake:%.4f CTC-real:%.4f " \
                               "Wid-fake:%.4f Wid-real:%.4f Recn-z:%.4f FDL:%.4f Kl:%.4f" \
                               % (epoch, self.opt.training.epochs,
                                  iter_count % len(self.train_loader), len(self.train_loader),
                                  meter_vals['adv_loss'], meter_vals['adv_loss_hf'],
                                  meter_vals['fake_disc_loss'], meter_vals['real_disc_loss'],
                                  meter_vals['hf_fake_disc_loss'], meter_vals['hf_real_disc_loss'],
                                  meter_vals['fake_ctc_loss'], meter_vals['real_ctc_loss'],
                                  meter_vals['fake_wid_loss'], meter_vals['real_wid_loss'],
                                  meter_vals['info_loss'], meter_vals['fdl_loss'], meter_vals['kl_loss'])
                        self.print(info)
                        if self.writer:
                            for key, val in meter_vals.items():
                                self._safe_add_scalar('loss/%s' % key, val, iter_count + 1)

                    # sampling
                    if (iter_count + 1) % self.opt.training.sample_iter_val == 0:
                        sample_root = os.path.join(self.log_root, self.opt.training.sample_dir)
                        if not os.path.exists(sample_root):
                            os.makedirs(sample_root)
                        self.sample_images(iter_count + 1)

                    iter_count += 1

                except torch.OutOfMemoryError as e:
                    if self.device.type != "cuda":
                        raise
                    self.print(f"[WARN] CUDA OOM en iter {iter_count} (bs={opt.training.batch_size}). Reduciendo batch_size y reintentando epoch.")
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    new_bs = max(1, int(opt.training.batch_size) // 2)
                    if new_bs >= int(opt.training.batch_size):
                        raise RuntimeError(
                            "La configuración actual no cabe ni con batch_size=1. "
                            "Reduce memoria: desactiva HF_D/FDL/R1, baja canales o usa una config low-VRAM."
                        ) from e
                    opt.training.batch_size = new_bs
                    self.z = prepare_z_dist(new_bs, opt.GenModel.style_dim, self.device, seed=self.opt.seed)
                    self.y = prepare_y_dist(new_bs, len(self.lexicon), self.device, seed=self.opt.seed)
                    self.train_loader = self._make_loader(
                        self._get_dataset(section="training"),
                        batch_size=new_bs,
                        shuffle=True,
                        drop_last=True,
                    )
                    restart_epoch = True
                    break

            if restart_epoch:
                continue

            # --- Validation (no-leakage split) ---
            valid_every = int(getattr(self.opt.valid, "every", 1) or 1)
            if (epoch % max(valid_every, 1)) == 0:
                try:
                    val_scores = self.validate_recognition(split=self.opt.valid.dset_split)
                    if self.writer:
                        for k, v in val_scores.items():
                            self._safe_add_scalar(f"valid/{k}", v, epoch)
                    self.print(
                        f"[VAL {epoch:03d}] ctc_loss={val_scores['ctc_loss']:.4f} "
                        f"cer={val_scores['cer']:.4f} char_acc={val_scores['char_acc']:.4f} "
                        f"wid_acc={val_scores['wid_acc']:.4f} style_cos={val_scores['style_cosine']:.4f} "
                        f"word_acc={val_scores['word_acc']:.4f} enye_word_acc={val_scores['enye_word_acc']:.4f} "
                        f"n->ñ={val_scores['conf_n_to_enye']:.4f} ñ->n={val_scores['conf_enye_to_n']:.4f}"
                    )

                    monitor_metric = str(getattr(self.opt.valid, "monitor_metric", "word_acc"))
                    monitor_mode = str(getattr(self.opt.valid, "monitor_mode", "max")).lower()
                    monitor_value = val_scores.get(monitor_metric)
                    if monitor_value is not None and np.isfinite(monitor_value):
                        improved = (
                            best_monitor is None
                            or (monitor_mode == "min" and monitor_value < best_monitor)
                            or (monitor_mode != "min" and monitor_value > best_monitor)
                        )
                        if improved:
                            best_monitor = float(monitor_value)
                            early_stop_bad_epochs = 0
                            self.save(
                                "best_recognition",
                                epoch,
                                monitor_metric=monitor_metric,
                                monitor_value=best_monitor,
                            )
                        else:
                            early_stop_bad_epochs += 1
                except Exception as e:
                    self.print(f"[WARN] Validación CTC falló: {e}")

            if epoch:
                ckpt_root = os.path.join(self.log_root, self.opt.training.ckpt_dir)
                if not os.path.exists(ckpt_root):
                    os.makedirs(ckpt_root)

                self.save('last', epoch)
                if epoch >= self.opt.training.start_save_epoch_val and \
                        epoch % self.opt.training.save_epoch_val == 0:
                    self.print('Calculate FID_KID')
                    scores = self.validate()
                    fid, kid = scores['FID'], scores['KID']
                    self.print('FID:{} KID:{}'.format(fid, kid))

                    if kid < best_kid:
                        best_kid = kid
                        self.save('best', epoch, KID=kid, FID=fid)
                    if self.writer:
                        self._safe_add_scalar('valid/FID', fid, epoch)
                        self._safe_add_scalar('valid/KID', kid, epoch)

            early_stop_cfg = getattr(self.opt.valid, "early_stop", None)
            if early_stop_cfg and bool(getattr(early_stop_cfg, "enabled", False)):
                patience = int(getattr(early_stop_cfg, "patience", 0) or 0)
                if patience > 0 and early_stop_bad_epochs >= patience:
                    self.print(
                        f"[INFO] Early stopping activado en epoch {epoch}: sin mejora en "
                        f"{getattr(self.opt.valid, 'monitor_metric', 'word_acc')} durante {early_stop_bad_epochs} épocas."
                    )
                    break

            for scheduler in self.lr_schedulers.values():
                scheduler.step(epoch)

            epoch += 1

    @staticmethod
    def _ctc_greedy_decode(argmax_seq, seq_len, blank_idx: int):
        # argmax_seq: (T,) int tensor on CPU
        out = []
        prev = None
        T = int(seq_len)
        for t in range(T):
            v = int(argmax_seq[t])
            if v == blank_idx:
                prev = v
                continue
            if prev is not None and v == prev:
                continue
            out.append(v)
            prev = v
        return out

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        if a == b:
            return 0
        if len(a) == 0:
            return len(b)
        if len(b) == 0:
            return len(a)
        # DP over the shorter string for memory.
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i]
            for j, cb in enumerate(b, start=1):
                ins = cur[j - 1] + 1
                dele = prev[j] + 1
                sub = prev[j - 1] + (0 if ca == cb else 1)
                cur.append(min(ins, dele, sub))
            prev = cur
        return prev[-1]

    @staticmethod
    def _align_strings(pred: str, tgt: str):
        dp = [[0] * (len(tgt) + 1) for _ in range(len(pred) + 1)]
        for i in range(len(pred) + 1):
            dp[i][0] = i
        for j in range(len(tgt) + 1):
            dp[0][j] = j
        for i in range(1, len(pred) + 1):
            for j in range(1, len(tgt) + 1):
                cost = 0 if pred[i - 1] == tgt[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

        aligned_pred = []
        aligned_tgt = []
        i, j = len(pred), len(tgt)
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                cost = 0 if pred[i - 1] == tgt[j - 1] else 1
                if dp[i][j] == dp[i - 1][j - 1] + cost:
                    aligned_pred.append(pred[i - 1])
                    aligned_tgt.append(tgt[j - 1])
                    i -= 1
                    j -= 1
                    continue
            if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                aligned_pred.append(pred[i - 1])
                aligned_tgt.append("")
                i -= 1
            else:
                aligned_pred.append("")
                aligned_tgt.append(tgt[j - 1])
                j -= 1
        return aligned_pred[::-1], aligned_tgt[::-1]

    def validate_recognition(self, split="val"):
        """
        Fast validation metrics on real images:
          - ctc_loss (Recognizer)
          - cer (character error rate, greedy CTC)
          - wid_acc (WriterIdentifier accuracy)
        """
        self.set_mode("eval")
        device = self.device

        dset = self._get_dataset(split=split, section="valid")
        dloader = self._make_loader(
            dset,
            batch_size=int(getattr(self.opt.valid, "batch_size", 8) or 8),
            shuffle=False,
            drop_last=False,
        )

        blank = int(self.ctc_blank_idx)
        len_scale = int(getattr(self.models.R, "len_scale", 8))
        max_batches = int(getattr(self.opt.valid, "max_batches", 0) or 0)
        compute_style_metrics = bool(getattr(self.opt.valid, "compute_style_metrics", False))

        total_loss = 0.0
        total_cer = 0.0
        total_wid_correct = 0
        total_samples = 0
        total_char_correct = 0
        total_char_count = 0
        total_word_correct = 0
        total_enye_samples = 0
        total_enye_correct = 0
        total_enye_char_correct = 0
        total_enye_char_count = 0
        n_to_enye = 0
        enye_to_n = 0
        target_n = 0
        target_enye = 0
        total_style_cos = 0.0
        total_style_wid_correct = 0

        with torch.no_grad():
            for bi, (imgs, img_lens, lbs, lb_lens, wids) in enumerate(dloader):
                imgs = imgs.to(device, non_blocking=True)
                img_lens = img_lens.to(device, non_blocking=True)
                lbs = lbs.to(device, non_blocking=True)
                lb_lens = lb_lens.to(device, non_blocking=True)
                wids = wids.to(device, non_blocking=True)

                log_probs = self.models.R(imgs)  # (T, N, C)
                in_lens = (img_lens // len_scale).clamp_min(1).clamp_max(log_probs.size(0))
                loss = self.ctc_loss(log_probs, lbs, in_lens, lb_lens)

                # Writer-ID accuracy (use full images for a stable metric).
                wid_logits = self.models.W(imgs, img_lens, self.models.S)
                wid_pred = wid_logits.argmax(dim=1)
                total_wid_correct += int((wid_pred == wids).sum().item())

                if compute_style_metrics:
                    # Style preservation proxy: reconstruct same text from its own style and
                    # compare style embeddings and writer prediction.
                    enc_styles = self.models.E(imgs, img_lens, self.models.S)
                    noise_dim = int(self.opt.GenModel.style_dim - self.opt.EncModel.style_dim)
                    if noise_dim > 0:
                        zeros = torch.zeros((imgs.size(0), noise_dim), device=device, dtype=enc_styles.dtype)
                        enc_z = torch.cat([zeros, enc_styles], dim=-1)
                    else:
                        enc_z = enc_styles
                    recn_imgs = self.models.G(enc_z, lbs, lb_lens)
                    recn_img_lens = lb_lens * self.opt.char_width
                    recn_styles = self.models.E(recn_imgs, recn_img_lens, self.models.S)
                    total_style_cos += float(F.cosine_similarity(enc_styles.float(), recn_styles.float(), dim=1).sum().item())
                    recn_wid_logits = self.models.W(recn_imgs, recn_img_lens, self.models.S)
                    total_style_wid_correct += int((recn_wid_logits.argmax(dim=1) == wids).sum().item())

                # CER via greedy CTC decode.
                pred_ids = log_probs.argmax(dim=2).transpose(0, 1).cpu()  # (N, T)
                in_lens_cpu = in_lens.cpu()
                tgt_texts = self.label_converter.decode(lbs.cpu(), lb_lens.cpu(), raw=True)
                for n in range(pred_ids.size(0)):
                    seq = self._ctc_greedy_decode(pred_ids[n], int(in_lens_cpu[n].item()), blank)
                    pred_text = "".join(self.alphabet[i] for i in seq if 0 <= i < len(self.alphabet))
                    tgt_text = tgt_texts[n]
                    denom = max(len(tgt_text), 1)
                    total_cer += self._levenshtein(pred_text, tgt_text) / float(denom)
                    total_word_correct += int(pred_text == tgt_text)
                    has_enye = ("ñ" in tgt_text) or ("Ñ" in tgt_text)
                    if has_enye:
                        total_enye_samples += 1
                        total_enye_correct += int(pred_text == tgt_text)

                    aligned_pred, aligned_tgt = self._align_strings(pred_text, tgt_text)
                    for pred_ch, tgt_ch in zip(aligned_pred, aligned_tgt):
                        if tgt_ch:
                            total_char_count += 1
                            total_char_correct += int(pred_ch == tgt_ch)
                        if tgt_ch == "n":
                            target_n += 1
                            if pred_ch == "ñ":
                                n_to_enye += 1
                        elif tgt_ch == "ñ":
                            target_enye += 1
                            total_enye_char_count += 1
                            total_enye_char_correct += int(pred_ch == tgt_ch)
                            if pred_ch == "n":
                                enye_to_n += 1

                bsz = int(imgs.size(0))
                total_loss += float(loss.item()) * bsz
                total_samples += bsz

                if max_batches > 0 and (bi + 1) >= max_batches:
                    break

        if total_samples == 0:
            return {
                "ctc_loss": float("nan"),
                "cer": float("nan"),
                "char_acc": float("nan"),
                "wid_acc": float("nan"),
                "word_acc": float("nan"),
                "enye_word_acc": float("nan"),
                "enye_char_acc": float("nan"),
                "conf_n_to_enye": float("nan"),
                "conf_enye_to_n": float("nan"),
                "style_cosine": float("nan"),
                "style_wid_acc": float("nan"),
            }
        return {
            "ctc_loss": total_loss / total_samples,
            "cer": total_cer / total_samples,
            "char_acc": (total_char_correct / total_char_count) if total_char_count > 0 else float("nan"),
            "wid_acc": total_wid_correct / total_samples,
            "word_acc": total_word_correct / total_samples,
            "enye_word_acc": (total_enye_correct / total_enye_samples) if total_enye_samples > 0 else float("nan"),
            "enye_char_acc": (total_enye_char_correct / total_enye_char_count) if total_enye_char_count > 0 else float("nan"),
            "conf_n_to_enye": (n_to_enye / target_n) if target_n > 0 else 0.0,
            "conf_enye_to_n": (enye_to_n / target_enye) if target_enye > 0 else 0.0,
            "style_cosine": (total_style_cos / total_samples) if compute_style_metrics else float("nan"),
            "style_wid_acc": (total_style_wid_correct / total_samples) if compute_style_metrics else float("nan"),
        }

    def sample_images(self, iteration_done=0):
        self.set_mode('eval')

        device = self.device
        batchA = next(iter(self.tst_loader))
        batchB = next(iter(self.tst_loader2))
        batch = Hdf5Dataset.merge_batch(batchA, batchB, device)
        imgs, img_lens, lbs, lb_lens, wids = batch

        real_imgs, real_img_lens = imgs.to(device), img_lens.to(device)
        real_lbs, real_lb_lens = lbs.to(device), lb_lens.to(device)

        with torch.no_grad():
            self.eval_z.sample_()
            recn_imgs = None
            if 'E' in self.models:
                enc_styles = self.models.E(real_imgs, real_img_lens, self.models.S)
                noises = torch.randn((real_imgs.size(0), self.opt.GenModel.style_dim
                                      - self.opt.EncModel.style_dim)).float().to(device)
                enc_z = torch.cat([noises, enc_styles], dim=-1)
                recn_imgs = self.models.G(enc_z, real_lbs, real_lb_lens)

            fake_real_imgs = self.models.G(self.eval_z, real_lbs, real_lb_lens)

            self.eval_y.sample_()
            sampled_words = idx_to_words(self.eval_y, self.lexicon, self.opt.training.capitalize_ratio)
            sampled_words[-2] = sampled_words[-1]
            fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
            fake_lbs, fake_lb_lens = fake_lbs.to(device), fake_lb_lens.to(device)
            fake_imgs = self.models.G(self.eval_z, fake_lbs, fake_lb_lens)

            max_img_len = max([real_imgs.size(-1), fake_real_imgs.size(-1), fake_imgs.size(-1)])
            img_shape = [real_imgs.size(2), max_img_len, real_imgs.size(1)]

            # Padding to the right should match the background (~+1 after Normalize([0.5],[0.5])).
            real_imgs = F.pad(real_imgs, [0, max_img_len - real_imgs.size(-1), 0, 0], value=1.)
            fake_real_imgs = F.pad(fake_real_imgs, [0, max_img_len - fake_real_imgs.size(-1), 0, 0], value=1.)
            fake_imgs = F.pad(fake_imgs, [0, max_img_len - fake_imgs.size(-1), 0, 0], value=1.)
            recn_imgs = F.pad(recn_imgs, [0, max_img_len - recn_imgs.size(-1), 0, 0], value=1.) \
                        if recn_imgs is not None else None

            real_words = self.label_converter.decode(real_lbs, real_lb_lens)
            real_labels = words_to_images(real_words, *img_shape)
            rand_labels = words_to_images(sampled_words, *img_shape)

            try:
                sample_img_list = [real_labels.cpu(), real_imgs.cpu(), fake_real_imgs.cpu(),
                                   fake_imgs.cpu(), rand_labels.cpu()]
                if recn_imgs is not None:
                    sample_img_list.insert(2, recn_imgs.cpu())
                sample_imgs = torch.cat(sample_img_list, dim=2).repeat(1, 3, 1, 1)
                res_img = draw_image(1 - sample_imgs.data, nrow=self.opt.training.sample_nrow, normalize=True)
                save_path = os.path.join(self.log_root, self.opt.training.sample_dir,
                                         'iter_{}.png'.format(iteration_done))
                im = Image.fromarray(res_img)
                im.save(save_path)
                if self.writer:
                    self._safe_add_image('Image', res_img.transpose((2, 0, 1)), iteration_done)
            except RuntimeError as e:
                print(e)

    def image_generator(self, source_dloader, style_guided=True):
        device = self.device

        with torch.no_grad():
            for style_imgs, style_img_lens, style_lbs, style_lb_lens, style_wids in source_dloader:
                content_lbs, content_lb_lens = style_lbs.to(device), style_lb_lens.to(device)

                if style_guided:
                    enc_styles = self.models.E(style_imgs.to(device), style_img_lens.to(device),
                                               self.models.S)
                    noises = torch.randn((style_imgs.size(0), self.opt.GenModel.style_dim
                                          - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                else:
                    enc_z = torch.randn(style_imgs.size(0), self.opt.GenModel.style_dim).to(device)

                fake_imgs = self.models.G(enc_z, content_lbs.long(), content_lb_lens.long())
                fake_img_lens = content_lb_lens * self.opt.char_width
                yield fake_imgs, fake_img_lens, content_lbs, content_lb_lens, style_wids.to(device)


    def validate(self, guided=True):
        self.set_mode('eval')
        dset_name = self.opt.valid.dset_name if self.opt.valid.dset_name else self.opt.dataset
        dset = self._get_dataset(section="valid", dset_name=dset_name)
        dloader = DataLoader(
            dset,
            collate_fn=self.collect_fn,
            batch_size=self.opt.valid.batch_size,
            shuffle=False,
            num_workers=4
        )
        # style images are resized
        source_dset_name = self.opt.valid.dset_name.strip('_org') if self.opt.valid.dset_name else dset_name
        source_dloader = DataLoader(
            self._get_dataset(section="valid", dset_name=source_dset_name),
            collate_fn=self.collect_fn,
            batch_size=self.opt.valid.batch_size,
            shuffle=False,
            num_workers=4
        )
        generator = self.image_generator(source_dloader, guided)
        fid_kid = calculate_kid_fid(self.opt.valid, dloader, generator, self.max_valid_image_width, self.device)
        return fid_kid

    def eval_interp(self):
        self.set_mode('eval')

        with torch.no_grad():
            interp_num = self.opt.test.interp_num
            nrow, ncol = 1, interp_num
            while True:
                text = input('input text: ')
                if len(text) == 0:
                    break

                fake_lbs = self.label_converter.encode(text)
                fake_lbs = torch.LongTensor(fake_lbs)
                fake_lb_lens = torch.IntTensor([len(text)])

                style0 = torch.randn((1, self.opt.GenModel.style_dim))
                style1 = torch.randn(style0.size())
                noise = torch.randn((1, self.noise_dim)).repeat(interp_num, 1).to(self.device)

                styles = [torch.lerp(style0, style1, i / (interp_num - 1)) for i in range(interp_num)]
                styles = torch.cat(styles, dim=0).float().to(self.device)
                styles = torch.cat([noise, styles], dim=1).to(self.device)

                fake_lbs, fake_lb_lens = fake_lbs.repeat(nrow * ncol, 1).to(self.device),\
                                         fake_lb_lens.repeat(nrow * ncol).to(self.device)
                gen_imgs = self.models.G(styles, fake_lbs, fake_lb_lens)
                gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127
                plt.figure()
                for i in range(nrow * ncol):
                    plt.subplot(nrow, ncol, i + 1)
                    plt.imshow(gen_imgs[i], cmap='gray')
                    plt.axis('off')
                plt.tight_layout()
                plt.show()

    def image_generator_custom(self, source_dloader, style_guided=False, use_sampled_words=True):
        device = self.device
        opt = self.opt
        
        if use_sampled_words:
            max_batch_size = self.opt.valid.batch_size
        
        with torch.no_grad():
            for (
                style_imgs,
                style_img_lens,
                style_lbs,
                style_lb_lens,
                style_wids,
            ) in source_dloader:
                batch_size = style_imgs.size(0)
                
                # Get style information
                if style_guided:
                    enc_styles = self.models.E(
                        style_imgs.to(device),
                        style_img_lens.to(device),
                        self.models.S,
                    )
                    noises = torch.randn((batch_size, self.opt.GenModel.style_dim - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                else:
                    enc_z = torch.randn(batch_size, self.opt.GenModel.style_dim).to(device)
                
                if use_sampled_words:
                    self.temp_y_dist.sample_()
                    sampled_words = idx_to_words(
                        self.temp_y_dist[:batch_size],
                        self.lexicon, 
                        self.opt.training.capitalize_ratio
                    )
                    
                    fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
                    content_lbs = torch.LongTensor(fake_lbs).to(device)
                    content_lb_lens = torch.IntTensor(fake_lb_lens).to(device)
                else:
                    content_lbs, content_lb_lens = style_lbs.to(device), style_lb_lens.to(device)
    
                # Generate images
                fake_imgs = self.models.G(enc_z, content_lbs.long(), content_lb_lens.long())
                fake_img_lens = content_lb_lens * self.opt.char_width
                
                yield fake_imgs, fake_img_lens, content_lbs, content_lb_lens, style_wids.to(device)

    def image_generator_custom_CER(self, source_dloader, style_guided=False, use_sampled_words=True):
        device = self.device
        opt = self.opt
    
        with torch.no_grad():
            for (
                style_imgs,
                style_img_lens,
                style_lbs,
                style_lb_lens,
                style_wids,
            ) in source_dloader:
                batch_size = style_imgs.size(0)
    
                # Get style information
                if style_guided:
                    enc_styles = self.models.E(
                        style_imgs.to(device),
                        style_img_lens.to(device),
                        self.models.S,
                    )
                    noises = torch.randn((batch_size, self.opt.GenModel.style_dim - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                else:
                    enc_z = torch.randn(batch_size, self.opt.GenModel.style_dim).to(device)

                if use_sampled_words:
                    self.temp_y_dist.sample_()
                    sampled_words = idx_to_words(
                        self.temp_y_dist[:batch_size],
                        self.lexicon, 
                        self.opt.training.capitalize_ratio
                    )
                    fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
                    content_lbs = torch.LongTensor(fake_lbs).to(device)
                    content_lb_lens = torch.IntTensor(fake_lb_lens).to(device)
                else:
                    content_lbs, content_lb_lens = style_lbs.to(device), style_lb_lens.to(device)
    
                fake_imgs = self.models.G(enc_z, content_lbs.long(), content_lb_lens.long())
                fake_img_lens = content_lb_lens * self.opt.char_width
    
                yield fake_imgs, fake_img_lens, content_lbs, content_lb_lens, style_wids.to(device)

    def gen_random_images(self, guided=True, total=25000):
        import cv2
        from tqdm import tqdm
        self.set_mode("eval")
        dset_name = self.opt.valid.dset_name if self.opt.valid.dset_name else self.opt.dataset
        dset = self._get_dataset(section="valid", dset_name=dset_name)
        dloader = DataLoader(
            dset,
            collate_fn=self.collect_fn,
            batch_size=self.opt.valid.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True
        )
        def create_source_dloader():
            return DataLoader(
                self._get_dataset(section="valid", dset_name=self.opt.valid.dset_name),
                collate_fn=self.collect_fn,
                batch_size=self.opt.valid.batch_size,
                shuffle=False,
                num_workers=4,
                drop_last=True
            )
    
        fake_base = os.path.join("/kaggle/working/", "test-fake")
        os.makedirs(fake_base, exist_ok=True)
    
        # Prepare the random distribution
        max_batch_size = self.opt.valid.batch_size
        self.temp_y_dist = prepare_y_dist(max_batch_size, len(self.lexicon), self.device, seed=self.opt.seed)
    
        idx = 0
        with tqdm(total=total, desc="Generating images") as pbar:
            while idx < total:
                source_dloader = create_source_dloader()
                generator2 = self.image_generator_custom_CER(source_dloader, guided, use_sampled_words=True)
                try:
                    for batch in generator2:
                        if idx >= total:
                            break
                        imgs, img_lens, lb, lb_len, w = batch
                        lb_len = lb_len * self.opt.char_width
    
                        for i in range(imgs.shape[0]):
                            if idx >= total:
                                break
                            lbs = self.label_converter.decode(lb[i])
                            image = imgs[i, :, :, :lb_len[i]]
                            image = 255 * ((image[0] + 1) / 2)
                            image = image.cpu().numpy()
                            cv2.imwrite(
                                "/kaggle/working/test-fake/fw" + str(idx) + ".png",
                                image,
                            )
                            with open(
                                "/kaggle/working/test-fake/fw.txt",
                                "a",
                            ) as f:
                                label_str = ''.join(lbs[:lb_len[i] // self.opt.char_width])
                                f.write(f"fw{idx}.png\t{label_str}\n")
                            idx += 1
                            pbar.update(1)
                except StopIteration:
                    continue 

    def gen_fakes(self, guided=True, use_random_lexicon=False):
        import json
        import cv2
        self.set_mode('eval')
        
        real_root = os.path.join(self.log_root, 'reals_images')
        fake_root = os.path.join(self.log_root, 'fakes_images')
        os.makedirs(real_root, exist_ok=True)
        os.makedirs(fake_root, exist_ok=True)
        
        dset_name = self.opt.valid.dset_name if self.opt.valid.dset_name else self.opt.dataset
        dset = self._get_dataset(section="valid", dset_name=dset_name)
        source_dset_name = dset_name.strip('_org') if '_org' in dset_name else dset_name
        
        dloader = DataLoader(
            dset,
            collate_fn=self.collect_fn,
            batch_size=self.opt.valid.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        source_dloader = DataLoader(
            self._get_dataset(section="valid", dset_name=source_dset_name),
            collate_fn=self.collect_fn,
            batch_size=self.opt.valid.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        real_count = 0
        fake_count = 0
        author_count = {}
        real_transcriptions = {}
        
        # Process real images
        for real_imgs, real_img_lens, real_lbs, real_lb_lens, wids in tqdm(dloader, desc="Saving real images"):
            real_imgs = real_imgs.to(self.device)
            batch_size = real_imgs.size(0)
            
            real_texts = self.label_converter.decode(real_lbs, real_lb_lens)
            
            for i in range(batch_size):
                wid = wids[i].item()
                img_len = real_img_lens[i].item()
                text = real_texts[i]
                
                author_dir = os.path.join(real_root, f'{wid}')
                os.makedirs(author_dir, exist_ok=True)
                
                if wid not in author_count:
                    author_count[wid] = 0
                
                image = real_imgs[i, :, :, :img_len]
                image = 255 * ((image[0] + 1) / 2)

                image = pad(image, img_len, lenlb=len(text))
                
                img_filename = f'{author_count[wid]:04d}.png'
                img_path = os.path.join(author_dir, img_filename)
                cv2.imwrite(img_path, image)
                
                rel_path = f"{wid}/{img_filename}"
                real_transcriptions[rel_path] = text
                
                author_count[wid] += 1
                real_count += 1
        
        with open(os.path.join(real_root, 'transcriptions.json'), 'w') as f:
            json.dump(real_transcriptions, f, indent=2)
        
        author_count = {}
        fake_transcriptions = {}
        
        if use_random_lexicon:
            max_batch_size = self.opt.valid.batch_size
            self.temp_y_dist = prepare_y_dist(max_batch_size, len(self.lexicon), self.device, seed=self.opt.seed)
            generator = self.image_generator_custom(source_dloader, guided, use_sampled_words=True)
        else:
            generator = self.image_generator(source_dloader, guided)
            
            
        # Process fake images
        for fake_imgs, fake_img_lens, fake_lbs, fake_lb_lens, wids in tqdm(generator, desc="Saving fake images"):
            batch_size = fake_imgs.size(0)
            
            fake_texts = self.label_converter.decode(fake_lbs, fake_lb_lens)
            
            for i in range(batch_size):
                wid = wids[i].item()
                img_len = fake_img_lens[i].item()
                text = fake_texts[i]
                
                author_dir = os.path.join(fake_root, f'{wid}')
                os.makedirs(author_dir, exist_ok=True)
                
                if wid not in author_count:
                    author_count[wid] = 0
                
                image = fake_imgs[i, :, :, :img_len]
                image = 255 * ((image[0] + 1) / 2)

                image = pad(image, img_len, lenlb=len(text))
                
                img_filename = f'{author_count[wid]:04d}.png'
                img_path = os.path.join(author_dir, img_filename)
                cv2.imwrite(img_path, image)
                
                rel_path = f"{wid}/{img_filename}"
                fake_transcriptions[rel_path] = text
                
                author_count[wid] += 1
                fake_count += 1
        
        with open(os.path.join(fake_root, 'transcriptions.json'), 'w') as f:
            json.dump(fake_transcriptions, f, indent=2)
        
        sampling_mode = "random lexicon" if use_random_lexicon else "original text"
        self.print(f"Saved {real_count} real images from {len(os.listdir(real_root))-1} authors")  # -1 for transcriptions.json
        self.print(f"Saved {fake_count} fake images from {len(os.listdir(fake_root))-1} authors using {sampling_mode}")
        self.print(f"Created transcription files for CER calculation")
        
        return real_root, fake_root

    def _preprocess_sentences(self, sentences):
        
        if sentences is None:
            return [["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]]
        
        processed_sentences = []
        for sentence in sentences:
            if isinstance(sentence, str):
                # Split string by spaces and filter out empty strings
                words = [word.strip() for word in sentence.split(' ') if word.strip()]
                processed_sentences.append(words)
            elif isinstance(sentence, list):
                # Already a list of words
                processed_sentences.append(sentence)
            else:
                raise ValueError(f"Sentence must be a string or list of words, got {type(sentence)}")
        
        return processed_sentences

    def save_images_from_sentence(self, save_root=None, sentences=None):
        # Default sentence if none provided
        if sentences is None:
            sentences = ["The quick brown fox jumps over the lazy dog"]

        sentences = self._preprocess_sentences(sentences)
    
        self.set_mode('eval')
        device = self.device
    
        if save_root is None:
            save_root = os.path.join(self.log_root, 'sentence_images')
        os.makedirs(save_root, exist_ok=True)
    
        dataset = self._get_dataset(
            section="valid",
            dset_name=self.opt.valid.dset_name,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=5,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collect_fn,
            drop_last=False
        )
    
        wid_sample_counts = {}
        total_generated = 0
        
        for batch in tqdm(dataloader, desc="Processing writer styles"):
            imgs, img_lens, lbs, lb_lens, wids = batch
            style_imgs, style_img_lens = imgs.to(device), img_lens.to(device)
            
            with torch.no_grad():
                if 'E' in self.models:
                    enc_styles = self.models.E(style_imgs, style_img_lens, self.models.S)
                    noises = torch.randn((style_imgs.size(0), self.opt.GenModel.style_dim
                                          - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                else:
                    enc_z = torch.randn(style_imgs.size(0), self.opt.GenModel.style_dim).to(device)
                
                for sentence in sentences:
                    batch_words = []
                    for word in sentence:
                        batch_words.extend([word] * style_imgs.size(0))
                    
                    fake_lbs, fake_lb_lens = self.label_converter.encode(batch_words)
                    fake_lbs, fake_lb_lens = torch.LongTensor(fake_lbs).to(device), torch.IntTensor(fake_lb_lens).to(device)
                    
                    for i in range(style_imgs.size(0)):
                        wid = wids[i].item()
                        
                        if wid not in wid_sample_counts:
                            wid_sample_counts[wid] = 0
                        
                        wid_dir = os.path.join(save_root, f'wid{wid}')
                        os.makedirs(wid_dir, exist_ok=True)
                        
                        fake_imgs_list = []
                        
                        fake_imgs_list.append(style_imgs[i:i+1][:, :, :, :style_img_lens[i]])
                        
                        padding = torch.ones(
                            1,
                            style_imgs.size(1),
                            style_imgs.size(2),
                            16,
                        ).to(device)
                        fake_imgs_list.append(padding)
                        
                        for j in range(len(sentence)):
                            word_idx = i + j * style_imgs.size(0)
                            word_lbs = fake_lbs[word_idx:word_idx+1]
                            word_lb_lens = fake_lb_lens[word_idx:word_idx+1]
                            
                            fake_img = self.models.G(enc_z[i:i+1], word_lbs, word_lb_lens)
                            fake_img = fake_img[:, :, :, :word_lb_lens * self.opt.char_width]
                            fake_imgs_list.append(fake_img)
                            
                            if j < len(sentence) - 1:
                                padding = torch.ones(
                                    1,
                                    fake_img.size(1),
                                    fake_img.size(2),
                                    16,
                                ).to(device)
                                fake_imgs_list.append(padding)
                        
                        fake_imgs = torch.cat(fake_imgs_list, dim=3)
                        
                        img = fake_imgs[0][0]  
                        img = 255 * ((img + 1) / 2)  
                        img = img.cpu().numpy()
                        
                        sentence_str = '_'.join(sentence)
                        path = os.path.join(wid_dir, f'{wid_sample_counts[wid]:04d}_{sentence_str}.png')
                        cv2.imwrite(path, img)
                        
                        # Update counters
                        wid_sample_counts[wid] += 1
                        total_generated += 1
    
        self.print(f"Generated {total_generated} images across {len(wid_sample_counts)} writer IDs")
        self.print(f"Images saved to {save_root}")
        
        wid_counts_str = ", ".join([f"wid{wid}: {count}" for wid, count in wid_sample_counts.items()])
        self.print(f"Sample counts per writer ID: {wid_counts_str}")
        
        return save_root

    def save_images_from_reference_labels(self, save_root=None, max_samples_per_writer=None):
        
        self.set_mode('eval')
        device = self.device
    
        if save_root is None:
            save_root = os.path.join(self.log_root, 'reconstructed_images_by_wid')
        os.makedirs(save_root, exist_ok=True)
    
        dataset = self._get_dataset(
            section="valid",
            dset_name=self.opt.valid.dset_name,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=5,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collect_fn,
            drop_last=False
        )
    
        wid_sample_counts = {}
        total_generated = 0
        
        for batch in tqdm(dataloader, desc="Reconstructing images from reference labels"):
            imgs, img_lens, lbs, lb_lens, wids = batch
            style_imgs, style_img_lens = imgs.to(device), img_lens.to(device)
            lbs, lb_lens = lbs.to(device), lb_lens.to(device)
            
            original_labels = lbs.cpu().numpy()
            original_label_lens = lb_lens.cpu().numpy()
            
            with torch.no_grad():
                if 'E' in self.models:
                    enc_styles = self.models.E(style_imgs, style_img_lens, self.models.S)
                    noises = torch.randn((style_imgs.size(0), self.opt.GenModel.style_dim
                                          - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                else:
                    enc_z = torch.randn(style_imgs.size(0), self.opt.GenModel.style_dim).to(device)
                
                for i in range(style_imgs.size(0)):
                    wid = wids[i].item()
                    
                    if max_samples_per_writer is not None:
                        if wid in wid_sample_counts and wid_sample_counts[wid] >= max_samples_per_writer:
                            continue
                    
                    if wid not in wid_sample_counts:
                        wid_sample_counts[wid] = 0
                    
                    wid_dir = os.path.join(save_root, f'wid{wid}')
                    os.makedirs(wid_dir, exist_ok=True)
                    
                    original_label = original_labels[i][:original_label_lens[i]]
                    decoded_text = self.label_converter.decode(original_label)
                    
                    img_list = []
                    
                    ref_img = style_imgs[i:i+1][:, :, :, :style_img_lens[i]]
                    img_list.append(ref_img)
                    
                    padding = torch.ones(1, ref_img.size(1), ref_img.size(2), 16).to(device)
                    img_list.append(padding)
                    
                    fake_img = self.models.G(enc_z[i:i+1], lbs[i:i+1], lb_lens[i:i+1])
                    fake_img = fake_img[:, :, :, :lb_lens[i] * self.opt.char_width]
                    img_list.append(fake_img)
                    
                    combined_img = torch.cat(img_list, dim=3)
                    
                    img = combined_img[0][0]  
                    img = 255 * ((img + 1) / 2)  
                    img = img.cpu().numpy()
                    
                    clean_text = "".join(c for c in decoded_text if c.isalnum() or c in (' ', '-', '_')).strip()
                    clean_text = clean_text.replace(' ', '_')
                    path = os.path.join(wid_dir, f'{wid_sample_counts[wid]:04d}_ref_vs_gen_{clean_text}.png')
                    cv2.imwrite(path, img)
                    
                    # Update counters
                    wid_sample_counts[wid] += 1
                    total_generated += 1
    
        self.print(f"Generated {total_generated} reconstructed images across {len(wid_sample_counts)} writer IDs")
        self.print(f"Images saved to {save_root}")
        
        # Print sample counts per writer ID
        wid_counts_str = ", ".join([f"wid{wid}: {count}" for wid, count in wid_sample_counts.items()])
        self.print(f"Sample counts per writer ID: {wid_counts_str}")
        
        return save_root

    def generate_word_image(self, word, style_img, style_img_len, style_guided=True):
        """
        Generate a single word image using an optional reference style sample.
        Returns: uint8 numpy array (H, W).
        """
        self.set_mode('eval')
        device = self.device

        if style_img.dim() == 3:
            style_img = style_img.unsqueeze(0)
        style_img = style_img.to(device)

        if isinstance(style_img_len, (int, float)):
            style_img_len = torch.tensor([style_img_len], dtype=torch.int32, device=device)
        else:
            style_img_len = style_img_len.to(device)
        if style_img_len.dim() == 0:
            style_img_len = style_img_len.unsqueeze(0)

        encoded = self.label_converter.encode(word)
        if not encoded:
            encoded = [0]
        fake_lbs = torch.LongTensor([encoded]).to(device)
        fake_lb_lens = torch.IntTensor([len(encoded)]).to(device)

        with torch.no_grad():
            if style_guided and 'E' in self.models:
                enc_styles = self.models.E(style_img, style_img_len, self.models.S)
                noise = torch.randn(
                    (1, self.opt.GenModel.style_dim - self.opt.EncModel.style_dim),
                    device=device,
                    dtype=torch.float32
                )
                enc_z = torch.cat([noise, enc_styles], dim=-1)
            else:
                enc_z = torch.randn(1, self.opt.GenModel.style_dim, device=device, dtype=torch.float32)

            fake_img = self.models.G(enc_z, fake_lbs, fake_lb_lens)
            width = int(fake_lb_lens[0].item()) * self.opt.char_width
            img = fake_img[0, 0, :, :width].cpu().numpy()

        img = (255 * (img + 1) / 2).clip(0, 255).astype(np.uint8)
        return img

    def save_paragraph(self, save_root=None, sentences=None, words_per_line=10):
    
        # Default sentence if none provided
        if sentences is None:
            sentences = ["The quick brown fox jumps over the lazy dog"]
    
        sentences = self._preprocess_sentences(sentences)
    
        self.set_mode('eval')
        device = self.device
    
        # Output root directory
        if save_root is None:
            save_root = os.path.join(self.log_root, 'sentence_images_by_wid')
        os.makedirs(save_root, exist_ok=True)
    
        dataset = self._get_dataset(
            section="valid",
            dset_name=self.opt.valid.dset_name,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=5,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collect_fn,
            drop_last=False
        )
    
        wid_sample_counts = {}
        total_generated = 0
        
        for batch in tqdm(dataloader, desc="Processing writer styles"):
            imgs, img_lens, lbs, lb_lens, wids = batch
            style_imgs, style_img_lens = imgs.to(device), img_lens.to(device)
            
            with torch.no_grad():
                if 'E' in self.models:
                    enc_styles = self.models.E(style_imgs, style_img_lens, self.models.S)
                    noises = torch.randn((style_imgs.size(0), self.opt.GenModel.style_dim
                                          - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                else:
                    enc_z = torch.randn(style_imgs.size(0), self.opt.GenModel.style_dim).to(device)
                
                for sentence in sentences:
                    lines = []
                    for i in range(0, len(sentence), words_per_line):
                        line = sentence[i:i + words_per_line]
                        lines.append(line)
                    
                    batch_words = []
                    for line in lines:
                        for word in line:
                            batch_words.extend([word] * style_imgs.size(0))
                    
                    fake_lbs, fake_lb_lens = self.label_converter.encode(batch_words)
                    fake_lbs, fake_lb_lens = torch.LongTensor(fake_lbs).to(device), torch.IntTensor(fake_lb_lens).to(device)
                    
                    for i in range(style_imgs.size(0)):
                        wid = wids[i].item()
                        
                        if wid not in wid_sample_counts:
                            wid_sample_counts[wid] = 0
                        
                        wid_dir = os.path.join(save_root, f'wid{wid}')
                        os.makedirs(wid_dir, exist_ok=True)
                        
                        paragraph_lines = []
                        
                        ref_img = style_imgs[i:i+1][:, :, :, :style_img_lens[i]]
                        paragraph_lines.append(ref_img)
                        
                        ref_padding = torch.ones(
                            1,
                            style_imgs.size(1),
                            style_imgs.size(2),
                            16,
                        ).to(device)
                        paragraph_lines.append(ref_padding)
                        
                        word_idx_offset = 0
                        for line_idx, line in enumerate(lines):
                            line_imgs_list = []
                            
                            for j, word in enumerate(line):
                                word_idx = i + word_idx_offset * style_imgs.size(0)
                                word_lbs = fake_lbs[word_idx:word_idx+1]
                                word_lb_lens = fake_lb_lens[word_idx:word_idx+1]
                                
                                fake_img = self.models.G(enc_z[i:i+1], word_lbs, word_lb_lens)
                                fake_img = fake_img[:, :, :, :word_lb_lens * self.opt.char_width]
                                line_imgs_list.append(fake_img)
                                
                                if j < len(line) - 1:
                                    word_padding = torch.ones(
                                        1,
                                        fake_img.size(1),
                                        fake_img.size(2),
                                        16,
                                    ).to(device)
                                    line_imgs_list.append(word_padding)
                                
                                word_idx_offset += 1
                            
                            line_img = torch.cat(line_imgs_list, dim=3)
                            paragraph_lines.append(line_img)
                            
                            if line_idx < len(lines) - 1:
                                line_padding = torch.ones(
                                    1,
                                    line_img.size(1),
                                    8,
                                    line_img.size(3),
                                ).to(device)
                                paragraph_lines.append(line_padding)
                        
                        max_width = max([line.size(3) for line in paragraph_lines])
                        
                        padded_lines = []
                        for line in paragraph_lines:
                            if line.size(3) < max_width:
                                right_padding = torch.ones(
                                    line.size(0),
                                    line.size(1),
                                    line.size(2),
                                    max_width - line.size(3)
                                ).to(device)
                                padded_line = torch.cat([line, right_padding], dim=3)
                            else:
                                padded_line = line
                            padded_lines.append(padded_line)
                        
                        paragraph_img = torch.cat(padded_lines, dim=2)
                        
                        
                        img = paragraph_img[0][0]  
                        img = 255 * ((img + 1) / 2)  
                        img = img.cpu().numpy()
                        
                        sentence_str = '_'.join(sentence)
                        if len(sentence_str) > 100:
                            sentence_str = sentence_str[:100] + "..."
                        path = os.path.join(wid_dir, f'{wid_sample_counts[wid]:04d}_{words_per_line}wpl_{sentence_str}.png')
                        cv2.imwrite(path, img)
                        
                        wid_sample_counts[wid] += 1
                        total_generated += 1
    
        self.print(f"Generated {total_generated} paragraph images across {len(wid_sample_counts)} writer IDs")
        self.print(f"Images saved to {save_root} with {words_per_line} words per line")
        
        # Print sample counts per writer ID
        wid_counts_str = ", ".join([f"wid{wid}: {count}" for wid, count in wid_sample_counts.items()])
        self.print(f"Sample counts per writer ID: {wid_counts_str}")
        
        return save_root
