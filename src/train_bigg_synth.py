"""Adapt the bigG image tower with my own labelled generated images (X3 of the 16 November plan).

The student is the one of train_test_selftrain.py: PE-Core-bigG with LoRA on
every linear layer of the image tower, fixed text prototypes (80 templates),
scale 100. Training images: the 32px files of one or more manifests, target =
the prompt class with label smoothing. Seeds below --val-seeds are held out
and scored at the end; their logits are saved for x2_pilot_check.py.

With --test-fold the same batches also contain the test images of one fold
with the soft targets of a teacher file, as in the cross-fitted self-training.
The saved format is that of train_test_selftrain.py, so encode_test_selftrain.py
reads the student unchanged: without --test-fold it encodes the whole test set
(the student saw no test image), with --test-fold only the other fold.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_lowres_student import add_lora, load_model, lora_state
from train_synth_supervised import load_sets, report
from train_test_selftrain import SEED, Images, folds, load_images, load_teacher


@torch.inference_mode()
def score(model, prototypes, images, rows, preprocess, device):
    loader = DataLoader(Images(images, rows, preprocess, False), batch_size=32, shuffle=False, num_workers=4)
    logits = []
    for batch, _ in loader:
        with torch.autocast('cuda'):
            feature = model.encode_image(batch.to(device))
        logits.append((100*F.normalize(feature.float(), dim=1)@prototypes.T).cpu())
    return torch.cat(logits).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sets', nargs='+', required=True, help='Set names: external_data/synth_style_<name>.')
    parser.add_argument('--clean', choices=['none', 'top2'], default='top2')
    parser.add_argument('--val-seeds', type=int, default=24)
    parser.add_argument('--max-train-index', type=int)
    parser.add_argument('--test-fold', type=int, choices=[0, 1], help='Also train on this test fold with teacher targets.')
    parser.add_argument('--teacher', type=Path, help='Run folder with test_predictions.npz (soft targets of the test fold).')
    parser.add_argument('--split-seed', type=int, default=SEED)
    parser.add_argument('--base', type=Path, default=Path('runs/pe_bigg_imagenet80_fp16'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--label-smoothing', type=float, default=.1)
    parser.add_argument('--augment', choices=['weak', 'strong'], default='strong')
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--smoke-steps', type=int, help='Stop after this many steps and save nothing (code check).')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory.')
    assert (args.test_fold is None) == (args.teacher is None), 'Give --test-fold and --teacher together.'
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    base = json.loads((args.base/'metrics.json').read_text())
    images, labels, index, style, top2, sets, record_ids = load_sets(args.sets)
    held_out = index < args.val_seeds
    train = ~held_out
    if args.max_train_index is not None:
        train &= index < args.max_train_index
    if args.clean == 'top2':
        train &= top2
    rows, val_rows = np.flatnonzero(train), np.flatnonzero(held_out)
    assert len(rows) and len(val_rows) and not set(rows) & set(val_rows)
    smooth = args.label_smoothing
    targets = np.full((len(labels), 10), smooth/10, dtype=np.float32)
    targets[np.arange(len(labels)), labels] += 1-smooth
    test_rows = 0
    if args.test_fold is not None:
        test_images, test_ids, test_labels = load_images('test')
        assert test_labels is None
        chosen = np.flatnonzero(folds(len(test_ids), args.split_seed) == args.test_fold)
        rows = np.concatenate([rows, len(images)+chosen])
        targets = np.concatenate([targets, load_teacher(test_ids, args.teacher)])
        images = np.concatenate([images, test_images])
        test_rows = int(len(chosen))
    if args.smoke_steps:
        val_rows = np.random.default_rng(0).permutation(val_rows)[:48]
    targets = torch.from_numpy(targets)
    print('generated training rows', len(rows)-test_rows, 'test rows', test_rows, 'validation rows', len(val_rows), flush=True)
    model, preprocess, prototypes = load_model(base, device)
    blocks = add_lora(model, args.rank)
    params = [p for p in blocks.parameters() if p.requires_grad]
    loader = DataLoader(Images(images, rows, preprocess, 'strong' if args.augment == 'strong' else True),
                        batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    steps = args.epochs*len(loader)
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=steps, pct_start=.05)
    scaler = torch.amp.GradScaler('cuda')
    if not args.smoke_steps:
        args.output.mkdir(parents=True)
    history, started, step, window, stop = [], time.time(), 0, np.zeros(3), False
    for epoch in range(args.epochs):
        sums = np.zeros(3)
        for batch, row in loader:
            target = targets[row].to(device)
            with torch.autocast('cuda'):
                feature = model.encode_image(batch.to(device).requires_grad_(True))
            logits = 100*F.normalize(feature.float(), dim=1)@prototypes.T
            loss = F.kl_div(F.log_softmax(logits, 1), target, reduction='batchmean')
            assert torch.isfinite(loss), 'Non-finite loss.'
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            step += 1
            agree = float((logits.argmax(1) == target.argmax(1)).float().mean())
            sums += [float(loss), agree, 1]
            window += [float(loss), agree, 1]
            if step == 1 or step % 100 == 0 or args.smoke_steps:
                print(f'step {step}/{steps} kl {window[0]/window[2]:.4f} agree {window[1]/window[2]:.3f} '
                      f'{time.time()-started:.0f}s peak {torch.cuda.max_memory_allocated()/2**20:.0f}MB', flush=True)
                window[:] = 0
            if args.smoke_steps and step >= args.smoke_steps:
                stop = True
                break
        history.append(dict(epoch=epoch+1, kl=sums[0]/sums[2], agreement_with_target=sums[1]/sums[2],
                            seconds=time.time()-started))
        print(json.dumps(history[-1]), flush=True)
        if stop:
            break
    model.set_grad_checkpointing(False)
    validation = score(model, prototypes, images, val_rows, preprocess, device)
    held = report(validation, labels[val_rows], style[val_rows], top2[val_rows])
    print('held-out generated images', json.dumps({k: held[k] for k in ['rows', 'accuracy', 'kept_accuracy', 'kept_style_macro']}),
          flush=True)
    if args.smoke_steps:
        print('SMOKE_OK', flush=True)
        return
    torch.save(lora_state(blocks), args.output/'lora.pt')
    np.savez(args.output/'validation_logits.npz', ids=record_ids[val_rows], labels=labels[val_rows],
             style=style[val_rows], top2=top2[val_rows], logits=validation)
    (args.output/'metrics.json').write_text(json.dumps(dict(
        model=base['model'], base_run=str(args.base),
        data='test' if args.test_fold is not None else 'own generated images with prompt labels',
        fold=args.test_fold if args.test_fold is not None else -1, fold_seed=args.split_seed,
        teacher=str(args.teacher) if args.teacher else None, sets=sets, clean=args.clean, val_seeds=args.val_seeds,
        max_train_index=args.max_train_index, rows=int(len(rows)), generated_rows=int(len(rows)-test_rows),
        test_rows=test_rows, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, rank=args.rank,
        label_smoothing=smooth, seed=args.seed,
        loss='KL(target || student zero-shot probabilities), 80 templates, scale 100; target = smoothed prompt class'
             + (' or teacher probabilities on test rows' if test_rows else ''),
        augmentation=('random resized crop 0.6-1, flip, colour jitter 0.3 (p 0.8), greyscale (p 0.1)'
                      if args.augment == 'strong' else 'horizontal flip, shift up to 2 pixels with reflection'),
        held_out_generated=held, test_labels_used=False, warehouse_data_used=False, history=history,
        peak_memory_mb=torch.cuda.max_memory_allocated()/2**20), indent=2)+'\n')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
