import argparse
import random
import time
import os

import numpy as np
import torch
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import TextDataset, CollectionTextDataset
from models.model import VATr
from util.misc import EpochLossTracker, add_vatr_args, VATRPP_BASE_ALPHABET, VATRPP_BASE_SPECIAL, SPANISH_EXTRA
from util.util import loss_hinge_dis, loss_hinge_gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--finetune_checkpoint", type=str, default=None)
    parser = add_vatr_args(parser)

    args = parser.parse_args()

    rSeed(args.seed)
    if getattr(args, "english_words_path", None) and getattr(args, "words_path", None) == "files/spanish_words.txt":
        args.words_path = args.english_words_path

    if args.preset == "vatrpp":
        # Keep base character order stable for checkpoint compatibility, append Spanish chars as extra.
        args.alphabet = VATRPP_BASE_ALPHABET
        args.special_alphabet = VATRPP_BASE_SPECIAL + ''.join(c for c in SPANISH_EXTRA if c not in VATRPP_BASE_ALPHABET)
    allowed_chars = args.alphabet + args.special_alphabet
    restrict_ids = None
    if args.restrict_author_ids:
        restrict_ids = [int(x) for x in str(args.restrict_author_ids).split(",") if x.strip()]
    dataset = CollectionTextDataset(
        args.dataset,
        'files',
        TextDataset,
        file_suffix=args.file_suffix,
        num_examples=args.num_examples,
        collator_resolution=args.resolution,
        min_virtual_size=0,
        validation=False,
        debug=False,
        height=args.img_height,
        split="train",
        strict_split=args.strict_split,
        max_width=args.max_width,
        max_label_len=args.max_label_len,
        allowed_chars=allowed_chars,
        restrict_author_ids=restrict_ids,
    )
    datasetval = CollectionTextDataset(
        args.dataset,
        'files',
        TextDataset,
        file_suffix=args.file_suffix,
        num_examples=args.num_examples,
        collator_resolution=args.resolution,
        min_virtual_size=0,
        validation=False,
        debug=False,
        height=args.img_height,
        split="val",
        strict_split=args.strict_split,
        max_width=args.max_width,
        max_label_len=args.max_label_len,
        allowed_chars=allowed_chars,
        restrict_author_ids=restrict_ids,
    )

    args.num_writers = dataset.num_writers

    if len(dataset) == 0:
        raise RuntimeError(
            "Empty TRAIN split. With --strict_split enabled, this usually means your author IDs only fall into the "
            "VALIDATION/TEST buckets (e.g. only 1541/1544) or your pickle has no usable authors.\n"
            "Fix: ensure your dataset contains TRAIN author IDs per the strict mapping, or rerun with --no-strict_split."
        )
    if len(datasetval) == 0:
        print(
            "WARNING: Empty VAL split under strict mapping. Validation metrics/previews will be skipped. "
            "If this is not expected, check your author ID ranges or use --no-strict_split."
        )

    args.alphabet = ''.join(dict.fromkeys(args.alphabet))
    args.special_alphabet = ''.join(c for c in args.special_alphabet if c not in args.alphabet)

    args.exp_name = f"{args.dataset}-{args.num_writers}-{args.num_examples}-LR{args.g_lr}-bs{args.batch_size}-{args.tag}"

    args.wandb = False
    wandb_id = None

    MODEL_PATH = os.path.join(args.save_model_path, args.exp_name)
    os.makedirs(MODEL_PATH, exist_ok=True)

    def make_loaders(batch_size: int):
        train_loader_ = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=dataset.collate_fn,
        )
        val_loader_ = torch.utils.data.DataLoader(
            datasetval,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=datasetval.collate_fn,
        )
        return train_loader_, val_loader_

    # Auto-reduce batch size on OOM (target=32 on Colab T4).
    # Also cap batch size to number of writers to avoid drop_last producing 0 batches.
    bs = int(args.batch_size)
    if len(dataset) < bs:
        bs = max(1, int(len(dataset)))
        print(f"Dataset has only {len(dataset)} writers; capping batch size to {bs}")
    while True:
        args.batch_size = bs
        train_loader, val_loader = make_loaders(bs)
        model = VATr(args)
        if not torch.cuda.is_available():
            break
        try:
            data0 = next(iter(train_loader))
            model._set_input(data0)
            model.optimizer_G.zero_grad(set_to_none=True)
            model.forward()
            with model._autocast():
                probe = model.netD(**{'x': model.fake.detach()}).mean()
            model._backward(probe)
            model.optimizer_G.zero_grad(set_to_none=True)
            break
        except StopIteration:
            raise RuntimeError("TRAIN DataLoader produced no batches (empty dataset). Check your split and dataset file.")
        except torch.cuda.OutOfMemoryError:
            del model
            torch.cuda.empty_cache()
            bs = bs // 2
            if bs < 1:
                raise
            print(f"OOM during probe step; reducing batch size to {bs}")
            if len(dataset) < bs:
                bs = max(1, int(len(dataset)))
                print(f"Dataset has only {len(dataset)} writers; capping batch size to {bs}")

    # Recreate model with a clean init after probing.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rSeed(args.seed)
    model = VATr(args)
    start_epoch = 0

    schedulers = []
    if args.scheduler == "cosine":
        schedulers.append(CosineAnnealingLR(model.optimizer_G, T_max=max(1, args.epochs)))
        schedulers.append(CosineAnnealingLR(model.optimizer_D, T_max=max(1, args.epochs)))
        if getattr(model, "optimizer_OCR", None) is not None:
            schedulers.append(CosineAnnealingLR(model.optimizer_OCR, T_max=max(1, args.epochs)))
        if getattr(model, "optimizer_wl", None) is not None:
            schedulers.append(CosineAnnealingLR(model.optimizer_wl, T_max=max(1, args.epochs)))

    checkpoint_path = os.path.join(MODEL_PATH, 'model.pth')

    loss_tracker = EpochLossTracker()

    if args.resume and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print(checkpoint_path + ' : Model loaded Successfully')
    elif args.resume:
        raise FileNotFoundError(f'No model found at {checkpoint_path}')
    else:
        if args.finetune_checkpoint is not None and args.finetune_checkpoint.lower() != 'none':
            print('Loading finetune checkpoint...', args.finetune_checkpoint)
            assert os.path.exists(args.finetune_checkpoint)
            try:
                checkpoint = torch.load(args.finetune_checkpoint, map_location=args.device, weights_only=True)
            except TypeError:
                checkpoint = torch.load(args.finetune_checkpoint, map_location=args.device)
            checkpoint = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

            model_state = model.state_dict()
            filtered_state = {}
            skipped = []
            for key, value in checkpoint.items():
                if key in model_state and model_state[key].shape == value.shape:
                    filtered_state[key] = value
                else:
                    skipped.append(key)

            miss, unexp = model.load_state_dict(filtered_state, strict=False)
            print(f'Finetune load: matched={len(filtered_state)} skipped={len(skipped)} missing={len(miss)} unexpected={len(unexp)}')
            if skipped:
                print(f"Skipped keys sample: {skipped[:5]}")
        elif args.feat_model_path is not None and args.feat_model_path.lower() != 'none':
            print('Loading...', args.feat_model_path)
            assert os.path.exists(args.feat_model_path)
            try:
                checkpoint = torch.load(args.feat_model_path, map_location=args.device, weights_only=True)
            except TypeError:
                checkpoint = torch.load(args.feat_model_path, map_location=args.device)
            checkpoint['model']['conv1.weight'] = checkpoint['model']['conv1.weight'].mean(1).unsqueeze(1)
            del checkpoint['model']['fc.weight']
            del checkpoint['model']['fc.bias']
            miss, unexp = model.netG.Feat_Encoder.load_state_dict(checkpoint['model'], strict=False)
            if not os.path.isdir(MODEL_PATH):
                os.mkdir(MODEL_PATH)
        else:
            print(f'WARNING: No resume of Resnet-18, starting from scratch')

    torch.backends.cudnn.benchmark = True

    def save_preview(page_arr: np.ndarray, out_path: str):
        page_arr = np.asarray(page_arr)
        if np.issubdtype(page_arr.dtype, np.floating):
            mx = float(np.nanmax(page_arr)) if page_arr.size else 0.0
            mn = float(np.nanmin(page_arr)) if page_arr.size else 0.0
            # Common ranges:
            # - [-1, 1] for tanh outputs
            # - [0, 1] for normalized images
            if mn >= -1.01 and mx <= 1.01:
                if mn < 0.0:
                    page_arr = (page_arr + 1.0) * 0.5
                page_arr = page_arr * 255.0
        page_arr = np.clip(page_arr, 0, 255).astype(np.uint8)
        Image.fromarray(page_arr).save(out_path)

    @torch.no_grad()
    def eval_losses(loader, max_batches: int = 10):
        model.eval()
        losses = []
        for i, data in enumerate(loader):
            if i >= max_batches:
                break
            model._set_input(data)
            model.forward()
            with model._autocast():
                pred_real = model.netD(model.real.detach())
                pred_fake = model.netD(**{'x': model.fake.detach()})
                d_real, d_fake = loss_hinge_dis(
                    pred_fake, pred_real, model.len_text_fake.detach(), model.len_text.detach(), True
                )
                loss_d = float((d_real + d_fake).detach().float().mean().item())
                loss_g = float(loss_hinge_gen(
                    model.netD(**{'x': model.fake.detach()}), model.len_text_fake.detach(), True
                ).detach().float().mean().item())
            loss_ocr_real = 0.0 if args.no_ocr_loss else float(model.compute_real_ocr_loss().detach().float().item())
            loss_ocr_fake = 0.0 if args.no_ocr_loss else float(model.compute_fake_ocr_loss().detach().float().item())
            losses.append((loss_g, loss_d, loss_ocr_real, loss_ocr_fake))

        if not losses:
            return {"G": float("nan"), "D": float("nan"), "OCR_real": float("nan"), "OCR_fake": float("nan")}
        arr = np.array(losses, dtype=np.float32)
        return {
            "G": float(arr[:, 0].mean()),
            "D": float(arr[:, 1].mean()),
            "OCR_real": float(arr[:, 2].mean()),
            "OCR_fake": float(arr[:, 3].mean()),
        }

    print(f"Starting training")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        log_time = time.time()
        loss_tracker.reset()
        model.d_acc.update(0.0)
        if args.text_augment_strength > 0:
            model.set_text_aug_strength(args.text_augment_strength)

        for i, data in enumerate(train_loader):
            model.update_parameters(epoch)
            model._set_input(data)

            model.optimize_G_only()
            model.optimize_G_step()

            if not args.no_ocr_loss:
                model.optimize_D_OCR()
                model.optimize_D_OCR_step()

            if not args.no_writer_loss:
                model.optimize_G_WL()
                model.optimize_G_step()

                model.optimize_D_WL()
                model.optimize_D_WL_step()

            if time.time() - log_time > 10:
                print(
                    f'Epoch {epoch} {i / len(train_loader) * 100:.02f}% running, current time: {time.time() - start_time:.2f} s')
                log_time = time.time()

            batch_losses = model.get_current_losses()
            batch_losses['d_acc'] = model.d_acc.avg
            loss_tracker.add_batch(batch_losses)

        end_time = time.time()
        losses = loss_tracker.get_epoch_loss()
        val_metrics = None
        if len(datasetval) > 0:
            val_metrics = eval_losses(val_loader, max_batches=10)
            try:
                data_val = next(iter(val_loader))
                page_val = model._generate_page(data_val['simg'].to(args.device), data_val['swids'])
                save_preview(page_val, os.path.join(MODEL_PATH, f"preview_val_epoch{epoch:04d}.png"))
            except StopIteration:
                pass

        for s in schedulers:
            s.step()

        print({'EPOCH': epoch, 'TIME': end_time - start_time, 'LOSSES': losses, 'VAL': val_metrics})
        print(f"Text sample: {model.get_text_sample(10)}")

        checkpoint = {
            'model': model.state_dict(),
            'wandb_id': wandb_id,
            'epoch': epoch
        }
        if epoch % args.save_model == 0:
            torch.save(checkpoint, os.path.join(MODEL_PATH, 'model.pth'))

        if epoch % args.save_model_history == 0:
            torch.save(checkpoint, os.path.join(MODEL_PATH, f'{epoch:04d}_model.pth'))

    # Final test-only evaluation (no train/test leakage during training).
    datasettest = CollectionTextDataset(
        args.dataset,
        'files',
        TextDataset,
        file_suffix=args.file_suffix,
        num_examples=args.num_examples,
        collator_resolution=args.resolution,
        min_virtual_size=0,
        validation=False,
        debug=False,
        height=args.img_height,
        split="test",
        strict_split=args.strict_split,
        max_width=args.max_width,
        max_label_len=args.max_label_len,
        allowed_chars=allowed_chars,
    )
    if len(datasettest) == 0:
        print("WARNING: Empty TEST split under strict mapping; skipping final test evaluation.")
    else:
        test_loader = torch.utils.data.DataLoader(
            datasettest,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=datasettest.collate_fn,
        )
        test_metrics = eval_losses(test_loader, max_batches=20)
        print({'TEST': test_metrics})

        try:
            data_test = next(iter(test_loader))
            page_test = model._generate_page(data_test['simg'].to(args.device), data_test['swids'])
            save_preview(page_test, os.path.join(MODEL_PATH, "preview_test_final.png"))
        except StopIteration:
            pass


def rSeed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)


if __name__ == "__main__":
    print("Training Model")
    main()
