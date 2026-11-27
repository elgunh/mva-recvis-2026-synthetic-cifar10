"""Supervised adaptation of DINOv2-g on my own generated images (X2 of the 16 November plan).

Images: the 32px PNG files of one or more manifests (prepare_synth_manifest.py).
Label: the class named in the prompt of each generated image. Seeds below
--val-seeds are held out and only scored. The model and the saved format are
those of train_dino_selftrain.py (LoRA on every linear layer of the blocks,
cosine head), so encode_dino_selftrain.py reads the student unchanged; fold -1
tells it that the student saw no test image and may encode the whole test set.
Without --test-fold no competition image is read. With --test-fold the unlabelled
test images of one fold are read with the soft targets of a teacher file. No test
label exists or is used.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from generate_style_images import STYLES
from train_dino_selftrain import REPO, REVISION, Head, add_lora, features, load_dino
from train_lowres_student import lora_state
from train_test_selftrain import SEED, Images, folds, load_images, load_teacher

STYLE_KEYS = [k for k, _ in STYLES]


def load_sets(names):
    images, labels, index, style, top2, sets, ids = [], [], [], [], [], [], []
    for name in names:
        root = Path(f'external_data/synth_style_{name}')
        raw = (root/'manifest.json').read_bytes()
        manifest = json.loads(raw)
        records = manifest['records']
        with np.load(Path(f'runs/synth_{name}_judge')/'judge.npz') as d:
            assert np.array_equal(d['ids'], np.array([r['id'] for r in records])), 'Judge and manifest differ.'
            kept = d['top2'].astype(bool)
        for record in records:
            with Image.open(root/record['path']) as image:
                images.append(np.asarray(image.convert('RGB'), dtype=np.uint8))
        labels += [r['label'] for r in records]
        ids += [r['id'] for r in records]
        index += [r['index'] for r in records]
        style += [STYLE_KEYS.index(r['domain']) for r in records]
        top2.append(kept)
        sets.append(dict(name=name, manifest_sha256=hashlib.sha256(raw).hexdigest(), rows=len(records),
                         kernel=manifest['kernel'], kernel_source=manifest.get('kernel_source', 'hires'),
                         judge_top2_rate=float(kept.mean())))
    images = np.stack(images)
    assert images.shape[1:] == (32, 32, 3)
    return images, np.array(labels), np.array(index), np.array(style), np.concatenate(top2), sets, np.array(ids)


@torch.inference_mode()
def encode(model, images, rows, transform, device):
    loader = DataLoader(Images(images, rows, transform, False), batch_size=128, shuffle=False, num_workers=4)
    return torch.cat([F.normalize(features(model, batch.to(device), device), dim=1).cpu() for batch, _ in loader])


def start_head(feature, y, smoothing):
    """Class means of frozen features of a training subset; the scale with the lowest training loss."""
    prototypes = F.normalize(torch.stack([feature[y == c].mean(0) for c in range(10)]), dim=1)
    cosine = feature@prototypes.T
    scale = min([5, 10, 15, 20, 30, 40, 60, 80, 100],
                key=lambda s: float(F.cross_entropy(s*cosine, y, label_smoothing=smoothing)))
    return prototypes.numpy(), scale, float((cosine.argmax(1) == y).float().mean())


def report(logits, y, style, kept):
    correct = logits.argmax(1) == y
    macro = lambda mask: float(np.mean([correct[mask & (style == s)].mean() for s in range(13) if (mask & (style == s)).any()]))
    everything = np.ones(len(y), bool)
    return dict(rows=int(len(y)), accuracy=float(correct.mean()), style_macro=macro(everything),
                kept_rows=int(kept.sum()), kept_accuracy=float(correct[kept].mean()), kept_style_macro=macro(kept),
                kept_by_style={k: float(correct[kept & (style == s)].mean()) for s, k in enumerate(STYLE_KEYS)
                               if (kept & (style == s)).any()},
                kept_by_class=[float(correct[kept & (y == c)].mean()) for c in range(10)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sets', nargs='+', required=True, help='Set names: external_data/synth_style_<name>.')
    parser.add_argument('--clean', choices=['none', 'top2'], required=True,
                        help='Training rows: all, or only those where the judge has the prompt class in its top 2.')
    parser.add_argument('--val-seeds', type=int, default=24, help='Images with a seed index below this are held out.')
    parser.add_argument('--max-train-index', type=int, help='Train only on seed indices below this.')
    parser.add_argument('--exclude-train-range', type=int, nargs=2, metavar=('START', 'STOP'),
                        help='Exclude the seed indices in [START, STOP) from training.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--label-smoothing', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-steps', type=int, help='Stop after this many steps and save nothing (code check).')
    # Joint mode: the batches also contain the test images of one fold with the soft targets of a teacher
    # file (cross-fitted self-training). The student then encodes only the other test fold.
    parser.add_argument('--test-fold', type=int, choices=[0, 1])
    parser.add_argument('--teacher', type=Path, help='Run folder with test_predictions.npz.')
    parser.add_argument('--split-seed', type=int, default=SEED)
    parser.add_argument('--image-size', type=int, default=224, help='Input side in pixels (a multiple of 14).')
    args = parser.parse_args()
    if args.exclude_train_range:
        start, stop = args.exclude_train_range
        if not 0 <= start < stop:
            parser.error('--exclude-train-range requires 0 <= START < STOP.')
    assert args.image_size % 14 == 0
    if args.output.exists():
        raise FileExistsError('Use a new output directory.')
    assert (args.test_fold is None) == (args.teacher is None), 'Give --test-fold and --teacher together.'
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    images, labels, index, style, top2, sets, record_ids = load_sets(args.sets)
    held_out = index < args.val_seeds
    train = ~held_out
    if args.max_train_index is not None:
        train &= index < args.max_train_index
    if args.exclude_train_range:
        train &= (index < start) | (index >= stop)
    if args.clean == 'top2':
        train &= top2
    rows, val_rows = np.flatnonzero(train), np.flatnonzero(held_out)
    assert len(rows) and len(val_rows) and not set(rows) & set(val_rows)
    if args.smoke_steps:  # code check: a few rows only
        val_rows = np.random.default_rng(0).permutation(val_rows)[:40]
    y = torch.from_numpy(labels)
    print('train rows', len(rows), 'validation rows', len(val_rows), 'class counts', np.bincount(labels[rows], minlength=10).tolist(),
          flush=True)
    model, transform = load_dino(device, args.image_size)
    subset = np.random.default_rng(args.seed).permutation(rows)[:60 if args.smoke_steps else 2600]
    prototypes, scale, start_accuracy = start_head(encode(model, images, subset, transform, device), y[subset],
                                                   args.label_smoothing)
    frozen_validation = (encode(model, images, val_rows, transform, device)@F.normalize(torch.as_tensor(prototypes), dim=1).T).numpy()
    start = report(frozen_validation, labels[val_rows], style[val_rows], top2[val_rows])
    print('head scale', scale, 'start accuracy on the subset', round(start_accuracy, 4),
          'frozen validation', round(start['kept_accuracy'], 4), flush=True)
    smoothing = args.label_smoothing
    targets = np.full((len(labels), 10), smoothing/10, dtype=np.float32)
    targets[np.arange(len(labels)), labels] += 1-smoothing
    test_rows = 0
    if args.test_fold is not None:
        test_images, test_ids, test_labels = load_images('test')
        assert test_labels is None
        chosen = np.flatnonzero(folds(len(test_ids), args.split_seed) == args.test_fold)
        rows = np.concatenate([rows, len(images)+chosen])
        targets = np.concatenate([targets, load_teacher(test_ids, args.teacher)])
        images = np.concatenate([images, test_images])
        test_rows = int(len(chosen))
        print('joint training: test rows', test_rows, 'fold', args.test_fold, 'teacher', args.teacher, flush=True)
    targets = torch.from_numpy(targets)
    blocks = add_lora(model, args.rank)
    model.set_grad_checkpointing(True)
    head = Head(prototypes, scale).to(device)
    adapters = [p for p in blocks.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW([dict(params=adapters, lr=args.lr), dict(params=head.parameters(), lr=args.head_lr)],
                                  weight_decay=0)
    loader = DataLoader(Images(images, rows, transform, True), batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=device.type == 'cuda')
    steps = args.epochs*len(loader)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=[args.lr, args.head_lr], total_steps=steps,
                                                   pct_start=.05)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    if not args.smoke_steps:
        args.output.mkdir(parents=True)
    history, started, step, window = [], time.time(), 0, np.zeros(3)
    for epoch in range(args.epochs):
        sums = np.zeros(3)
        for batch, row in loader:
            target = targets[row].to(device)
            logits = head(features(model, batch.to(device).requires_grad_(True), device))
            # Soft cross-entropy; on generated rows this is cross-entropy with label smoothing.
            loss = -(target*F.log_softmax(logits, 1)).sum(1).mean()
            target = target.argmax(1)
            assert torch.isfinite(loss), 'Non-finite loss.'
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(adapters+list(head.parameters()), 1.0)
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            step += 1
            hit = float((logits.argmax(1) == target).float().mean())
            sums += [float(loss), hit, 1]
            window += [float(loss), hit, 1]
            if step == 1 or step % 50 == 0 or args.smoke_steps:
                memory = torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else 0
                print(f'step {step}/{steps} loss {window[0]/window[2]:.4f} train accuracy {window[1]/window[2]:.3f} '
                      f'{time.time()-started:.0f}s peak {memory:.0f}MB', flush=True)
                window[:] = 0
            if args.smoke_steps and step >= args.smoke_steps:
                break
        model.set_grad_checkpointing(False)
        with torch.inference_mode():
            validation = head(encode(model, images, val_rows, transform, device).to(device)).float().cpu().numpy()
        model.set_grad_checkpointing(True)
        history.append(dict(epoch=epoch+1, loss=sums[0]/sums[2], train_accuracy=sums[1]/sums[2],
                            seconds=time.time()-started,
                            validation=report(validation, labels[val_rows], style[val_rows], top2[val_rows])))
        print(json.dumps({k: (v if k != 'validation' else {a: v[a] for a in ['accuracy', 'kept_accuracy', 'kept_style_macro']})
                          for k, v in history[-1].items()}), flush=True)
        if args.smoke_steps:
            print('SMOKE_OK', flush=True)
            return
    torch.save(dict(lora=lora_state(blocks), head=head.weight.detach().cpu(), scale=head.scale), args.output/'student.pt')
    # Held-out generated images, original view: used by x2_pilot_check.py.
    np.savez(args.output/'validation_logits.npz', ids=record_ids[val_rows], labels=labels[val_rows],
             style=style[val_rows], top2=top2[val_rows], logits=validation)
    (args.output/'metrics.json').write_text(json.dumps(dict(
        model=REPO, revision=REVISION, image_size=args.image_size,
        data='test' if args.test_fold is not None else 'own generated images with prompt labels',
        fold=args.test_fold if args.test_fold is not None else -1, fold_seed=args.split_seed,
        teacher=str(args.teacher) if args.teacher else None, test_rows=test_rows,
        generated_rows=int(len(rows)-test_rows), sets=sets,
        clean=args.clean, val_seeds=args.val_seeds, max_train_index=args.max_train_index,
        exclude_train_range=args.exclude_train_range, rows=int(len(rows)),
        validation_rows=int(len(val_rows)), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        head_lr=args.head_lr, rank=args.rank, label_smoothing=args.label_smoothing, seed=args.seed,
        head_scale=scale, frozen_head_validation=start, loss='cross-entropy on the prompt class, cosine head on the class token',
        augmentation='horizontal flip, shift up to 2 pixels with reflection', test_images_used=bool(test_rows),
        test_labels_used=False, warehouse_data_used=False, history=history,
        peak_memory_mb=torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None), indent=2)+'\n')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
