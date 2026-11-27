"""Adapt the bigG image tower to the 32px competition images (LoRA).

Two sources of supervision, chosen with --data:

test   Transductive self-training. The soft targets are the final
       probabilities of the reference pipeline (runs/dinov2_giant_candidate,
       public 0.97565). With --fold 0 or 1 the student is trained on one half
       of the test set (fixed split) and later predicts only the other half,
       so no image is classified by a model that saw its own pseudo-label.
       --fold -1 trains on all test images.
train  Supervised adaptation on the 50,000 CIFAR-10 training images (real
       32px images with their labels).

Student: PE-Core-bigG with LoRA on every linear layer of the image tower
blocks. Loss: KL(target || student zero-shot probabilities) with the 80
templates and scale 100 of the reference, so the text prototypes stay fixed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_lowres_student import add_lora, load_model, lora_state

TEACHER = Path('runs/dinov2_giant_candidate')
DATA = Path('data')
SEED = 20261114


def folds(n, seed=SEED):
    order = np.random.default_rng(seed).permutation(n)
    fold = np.zeros(n, dtype=np.int64)
    fold[order[n//2:]] = 1
    return fold


def load_images(split):
    with np.load(DATA/f'{split}.npz') as archive:
        images = archive['images'].astype(np.uint8)
        ids = archive['ids'].astype(np.int64) if 'ids' in archive.files else np.arange(len(images))
        labels = archive['labels'].astype(np.int64) if 'labels' in archive.files else None
    assert images.shape[1:] == (32, 32, 3)
    assert split != 'test' or labels is None, 'Competition test data must not contain labels.'
    return images, ids, labels


def load_teacher(ids, teacher=TEACHER):
    with np.load(Path(teacher)/'test_predictions.npz') as d:
        assert 'labels' not in d.files and np.array_equal(d['ids'], ids)
        p = d['probabilities'].astype(np.float32)
    return p/p.sum(1, keepdims=True)


def strong_augmentation():
    """Noisy-student style input noise on the 32px image: crop and rescale, flip, colour, grey."""
    import torchvision.transforms as T
    return T.Compose([T.RandomResizedCrop(32, scale=(.6, 1.), ratio=(.8, 1.25)), T.RandomHorizontalFlip(),
                      T.RandomApply([T.ColorJitter(.3, .3, .3, .05)], p=.8), T.RandomGrayscale(p=.1)])


class Images(Dataset):
    """32px arrays read as in the reference encoding; optional flip and shift of up to 2 pixels,
    or strong augmentation (augment='strong')."""

    def __init__(self, images, rows, preprocess, augment):
        self.images, self.rows, self.preprocess, self.augment = images, rows, preprocess, augment
        self.strong = strong_augmentation() if augment == 'strong' else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        image = self.images[row]
        if self.strong is not None:
            return self.preprocess(self.strong(Image.fromarray(image, mode='RGB'))), row
        if self.augment:
            if torch.rand(1).item() < .5:
                image = image[:, ::-1]
            dy, dx = torch.randint(0, 5, (2,)).tolist()
            image = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode='reflect')[dy:dy+32, dx:dx+32]
        return self.preprocess(Image.fromarray(np.ascontiguousarray(image), mode='RGB')), row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', choices=['test', 'train'], default='test')
    parser.add_argument('--fold', type=int, choices=[-1, 0, 1], default=-1, help='Test fold to train on; -1 is all.')
    parser.add_argument('--base', type=Path, default=Path('runs/pe_bigg_imagenet80_fp16'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--smoke-steps', type=int, help='Stop after this many steps and save nothing (code check).')
    parser.add_argument('--teacher', type=Path, default=TEACHER,
                        help='Run folder whose test_predictions.npz gives the soft targets (test data only).')
    parser.add_argument('--max-rows', type=int, help='Train on the first rows only and save (code check).')
    parser.add_argument('--split-seed', type=int, default=SEED, help='Seed of the two-fold test split.')
    parser.add_argument('--augment', choices=['weak', 'strong'], default='weak')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory.')
    torch.manual_seed(args.split_seed)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    base = json.loads((args.base/'metrics.json').read_text())
    images, ids, labels = load_images(args.data)
    if args.data == 'test':
        targets = load_teacher(ids, args.teacher)
        fold = folds(len(ids), args.split_seed)
        rows = np.arange(len(ids)) if args.fold < 0 else np.flatnonzero(fold == args.fold)
    else:
        targets = np.eye(10, dtype=np.float32)[labels]
        rows = np.arange(len(ids))
    if args.max_rows:
        rows = rows[:args.max_rows]
    targets = torch.from_numpy(targets)
    print(args.data, 'rows', len(rows), 'fold', args.fold, flush=True)
    model, preprocess, prototypes = load_model(base, device)
    blocks = add_lora(model, args.rank)
    params = [p for p in blocks.parameters() if p.requires_grad]
    loader = DataLoader(Images(images, rows, preprocess, 'strong' if args.augment == 'strong' else True),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True)
    steps = args.epochs*len(loader)
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=steps, pct_start=.05)
    scaler = torch.amp.GradScaler('cuda')
    if not args.smoke_steps:
        args.output.mkdir(parents=True)
    history, started, step = [], time.time(), 0
    window = np.zeros(3)
    for epoch in range(args.epochs):
        sums = np.zeros(3)
        for batch, index in loader:
            target = targets[index].to(device)
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
                print('SMOKE_OK', flush=True)
                return
        history.append(dict(epoch=epoch+1, kl=sums[0]/sums[2], agreement_with_target=sums[1]/sums[2],
                            seconds=time.time()-started))
        print(json.dumps(history[-1]), flush=True)
    torch.save(lora_state(blocks), args.output/'lora.pt')
    (args.output/'metrics.json').write_text(json.dumps(dict(
        model=base['model'], base_run=str(args.base), data=args.data, fold=args.fold, rows=int(len(rows)),
        fold_seed=args.split_seed, teacher=str(args.teacher) if args.data == 'test' else 'CIFAR-10 training labels',
        max_rows=args.max_rows,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, rank=args.rank,
        loss='KL(target || student zero-shot probabilities), 80 templates, scale 100',
        augmentation=('random resized crop 0.6-1, flip, colour jitter 0.3 (p 0.8), greyscale (p 0.1)'
                      if args.augment == 'strong' else 'horizontal flip, shift up to 2 pixels with reflection'),
        test_labels_used=False, history=history, peak_memory_mb=torch.cuda.max_memory_allocated()/2**20),
        indent=2)+'\n')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
