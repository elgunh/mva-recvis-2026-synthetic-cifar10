"""Cross-fitted self-training of the DINOv2-g branch of the graph.

The reference graph uses 25% DINOv2 ViT-g/14 reg4 class-token features
(224px, frozen). Here the same model gets LoRA adapters on every linear layer
of its blocks and a cosine classifier head (10 classes). Both are trained on
one half of the unlabeled test set with soft targets from a teacher file
(KL divergence); the head starts from teacher-weighted class means of the
frozen features. Only the adapted class-token feature is used later, in the
graph. The fold split and the augmentation are those of train_test_selftrain.py.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from train_lowres_student import LoRA, lora_state
from train_test_selftrain import SEED, TEACHER, Images, folds, load_images, load_teacher

REPO = 'timm/vit_giant_patch14_reg4_dinov2.lvd142m'
REVISION = '6994d74074a8f74e220de3f403dc49dd5d8ae704'
FROZEN = Path('runs/dinov2_giant_test/test_features.npz')


def load_dino(device, size=224):
    """size: side of the square input in pixels (the 32px image is resized to it); 224 in all runs before 18 November."""
    model = timm.create_model('hf-hub:'+REPO+'@'+REVISION, pretrained=True, num_classes=0, img_size=size)
    cfg = timm.data.resolve_model_data_config(model)
    cfg.update(input_size=(3, size, size), crop_pct=1.0)
    transform = timm.data.create_transform(**cfg, is_training=False)
    model = model.to(device).eval().requires_grad_(False)
    if device.type == 'cuda':
        model = model.half()  # frozen base in half precision; adapters and head stay in float32
    return model, transform


def add_lora(model, rank):
    def wrap(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRA(child, rank))
            else:
                wrap(child)
    wrap(model.blocks)
    model.blocks.to(next(model.parameters()).device)
    return model.blocks


class Head(nn.Module):
    def __init__(self, prototypes, scale):
        super().__init__()
        self.weight = nn.Parameter(torch.as_tensor(prototypes, dtype=torch.float32).clone())
        self.scale = float(scale)

    def forward(self, feature):
        return self.scale*F.normalize(feature, dim=1)@F.normalize(self.weight, dim=1).T


def start_head(ids, rows, targets):
    """Teacher-weighted class means of the frozen features; the scale that best fits the teacher."""
    with np.load(FROZEN) as d:
        assert np.array_equal(d['ids'], ids) and 'labels' not in d.files
        frozen = d['features'][rows].astype(np.float64)
    t = targets[rows].numpy().astype(np.float64)
    prototypes = t.T@frozen
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    cosine = frozen@prototypes.T
    best = None
    for scale in [5, 10, 15, 20, 30, 40, 60, 80, 100]:
        z = scale*cosine
        z -= z.max(1, keepdims=True)
        log_p = z-np.log(np.exp(z).sum(1, keepdims=True))
        kl = float(np.mean(np.sum(t*(np.log(np.maximum(t, 1e-12))-log_p), axis=1)))
        if best is None or kl < best[1]:
            best = (scale, kl, float((cosine.argmax(1) == t.argmax(1)).mean()))
    return prototypes, best


def features(model, images, device):
    with torch.autocast(device.type, enabled=device.type == 'cuda'):
        return model(images).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fold', type=int, choices=[0, 1], required=True, help='Test fold to train on.')
    parser.add_argument('--teacher', type=Path, default=TEACHER)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-steps', type=int, help='Stop after this many steps and save nothing (code check).')
    parser.add_argument('--max-rows', type=int, help='Train on the first rows only and save (code check).')
    parser.add_argument('--split-seed', type=int, default=SEED, help='Seed of the two-fold test split.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory.')
    torch.manual_seed(args.split_seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    images, ids, labels = load_images('test')
    assert labels is None
    targets = torch.from_numpy(load_teacher(ids, args.teacher))
    rows = np.flatnonzero(folds(len(ids), args.split_seed) == args.fold)
    if args.max_rows:
        rows = rows[:args.max_rows]
    prototypes, (scale, start_kl, start_agree) = start_head(ids, rows, targets)
    print('test rows', len(rows), 'fold', args.fold, 'head scale', scale, 'start kl', round(start_kl, 4),
          'start agreement', round(start_agree, 4), flush=True)
    model, transform = load_dino(device)
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
        for batch, index in loader:
            target = targets[index].to(device)
            logits = head(features(model, batch.to(device).requires_grad_(True), device))
            loss = F.kl_div(F.log_softmax(logits, 1), target, reduction='batchmean')
            assert torch.isfinite(loss), 'Non-finite loss.'
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(adapters+list(head.parameters()), 1.0)
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            step += 1
            agree = float((logits.argmax(1) == target.argmax(1)).float().mean())
            sums += [float(loss), agree, 1]
            window += [float(loss), agree, 1]
            if step == 1 or step % 50 == 0 or args.smoke_steps:
                memory = torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else 0
                print(f'step {step}/{steps} kl {window[0]/window[2]:.4f} agree {window[1]/window[2]:.3f} '
                      f'{time.time()-started:.0f}s peak {memory:.0f}MB', flush=True)
                window[:] = 0
            if args.smoke_steps and step >= args.smoke_steps:
                print('SMOKE_OK', flush=True)
                return
        history.append(dict(epoch=epoch+1, kl=sums[0]/sums[2], agreement_with_target=sums[1]/sums[2],
                            seconds=time.time()-started))
        print(json.dumps(history[-1]), flush=True)
    torch.save(dict(lora=lora_state(blocks), head=head.weight.detach().cpu(), scale=head.scale), args.output/'student.pt')
    (args.output/'metrics.json').write_text(json.dumps(dict(
        model=REPO, revision=REVISION, data='test', fold=args.fold, rows=int(len(rows)), fold_seed=args.split_seed,
        teacher=str(args.teacher), max_rows=args.max_rows, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, head_lr=args.head_lr, rank=args.rank, head_scale=scale, head_start_kl=start_kl,
        head_start_agreement=start_agree, loss='KL(teacher || cosine head on the class token)',
        augmentation='horizontal flip, shift up to 2 pixels with reflection', test_labels_used=False,
        history=history, peak_memory_mb=torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None),
        indent=2)+'\n')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
