"""Exploratory: adapt the bigG image tower to 32px inputs by feature distillation.

Student: PE-Core-bigG with LoRA on every linear layer of the image tower
blocks, fed the 32px proxy image with the usual preprocessing. Teacher: the
frozen encoder on the saved 224px file (cached features). Loss: cosine
distance to the teacher feature plus the KL divergence between the teacher
and student zero-shot class probabilities (80 templates, scale 100). The
prompt labels are not used. Development generators only; the confirmation
set and the competition images are not read.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from clip_baseline import text_prototypes


class LoRA(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.down = nn.Parameter(torch.randn(rank, base.in_features)*.01)
        self.up = nn.Parameter(torch.zeros(base.out_features, rank))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x)+F.linear(F.linear(x, self.down.to(x.dtype)), self.up.to(x.dtype))


def add_lora(model, rank):
    """Freeze the model, then wrap every linear layer of the image tower blocks."""
    model.eval().requires_grad_(False)
    blocks = model.visual.trunk.blocks

    def wrap(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRA(child, rank))
            else:
                wrap(child)
    wrap(blocks)
    blocks.to(next(model.parameters()).device)
    model.set_grad_checkpointing(True)
    return blocks


def lora_state(blocks):
    return {k: v.detach().cpu() for k, v in blocks.state_dict().items() if k.rsplit('.', 1)[-1] in ('down', 'up')}


def load_lora(blocks, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert set(state) == set(lora_state(blocks)), 'Adapter names do not match.'
    assert not blocks.load_state_dict(state, strict=False).unexpected_keys


def load_model(base, device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        base['model'], pretrained=None, precision=base.get('weight_precision', 'fp32'), device=device)
    model.eval()
    with torch.inference_mode():
        prototypes, prompts, _ = text_prototypes(model, open_clip.get_tokenizer(base['model']), 'imagenet', device)
    clean = lambda s: re.sub(r'0x[0-9a-fA-F]+', '<address>', s)
    assert prompts == base['prompts'] and clean(str(preprocess)) == clean(base['preprocess'])
    return model, preprocess, prototypes.float().clone()


class Small(Dataset):
    """The 32px proxy files, read exactly as in the reference encoding."""

    def __init__(self, items, preprocess):
        self.items, self.preprocess = items, preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        with Image.open(self.items[index]) as image:
            image = image.convert('RGB').resize((32, 32), Image.Resampling.BICUBIC)
            return self.preprocess(image), index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, nargs='+', required=True, help='Manifest roots.')
    parser.add_argument('--teacher', type=Path, nargs='+', required=True, help='224px feature runs, same order.')
    parser.add_argument('--base', type=Path, default=Path('runs/pe_bigg_imagenet80_fp16'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261114)
    parser.add_argument('--smoke-steps', type=int, help='Stop after this many steps and save nothing (code check).')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory.')
    assert len(args.train) == len(args.teacher)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    base = json.loads((args.base/'metrics.json').read_text())
    items, targets = [], []
    for root, run in zip(args.train, args.teacher):
        manifest = json.loads((root/'manifest.json').read_text())
        assert manifest['role'] == 'dev', 'Only development generators are used for training.'
        meta = json.loads((run/'metrics.json').read_text())
        assert meta['model'] == base['model'] and meta['input'].startswith('saved 224px PNG')
        with np.load(run/'hires_features.npz') as d:
            assert np.array_equal(d['ids'], np.array([r['id'] for r in manifest['records']]))
            targets.append(d['features'].astype(np.float32))
        items += [root/r['path'] for r in manifest['records']]
        print(root, len(manifest['records']), 'images', flush=True)
    teacher = torch.from_numpy(np.concatenate(targets))
    assert torch.allclose(teacher.norm(dim=1), torch.ones(len(teacher)), atol=1e-3)
    model, preprocess, prototypes = load_model(base, device)
    blocks = add_lora(model, args.rank)
    params = [p for p in blocks.parameters() if p.requires_grad]
    loader = DataLoader(Small(items, preprocess), batch_size=args.batch_size, shuffle=True, num_workers=4,
                        drop_last=True, pin_memory=True)
    steps = args.epochs*len(loader)
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=steps, pct_start=.05)
    scaler = torch.amp.GradScaler('cuda')
    if not args.smoke_steps:
        args.output.mkdir(parents=True)
    history, started, step = [], time.time(), 0
    for epoch in range(args.epochs):
        sums = np.zeros(4)
        for images, index in loader:
            images = images.to(device).requires_grad_(True)  # keeps gradients through checkpointed blocks
            target = teacher[index].to(device)
            with torch.autocast('cuda'):
                feature = model.encode_image(images)
            feature = F.normalize(feature.float(), dim=1)
            cosine = (1-(feature*target).sum(1)).mean()
            teacher_logits, student_logits = 100*target@prototypes.T, 100*feature@prototypes.T
            divergence = F.kl_div(F.log_softmax(student_logits, 1), F.softmax(teacher_logits, 1),
                                  reduction='batchmean')
            loss = cosine+divergence
            assert torch.isfinite(loss), 'Non-finite loss.'
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            step += 1
            agree = float((student_logits.argmax(1) == teacher_logits.argmax(1)).float().mean())
            sums += [float(cosine), float(divergence), agree, 1]
            if step == 1 or step % 50 == 0 or args.smoke_steps:
                print(f'step {step}/{steps} cosine {float(cosine):.4f} kl {float(divergence):.4f} agree {agree:.2f} '
                      f'{time.time()-started:.0f}s peak {torch.cuda.max_memory_allocated()/2**20:.0f}MB', flush=True)
            if args.smoke_steps and step >= args.smoke_steps:
                print('SMOKE_OK', flush=True)
                return
        history.append(dict(epoch=epoch+1, cosine=sums[0]/sums[3], kl=sums[1]/sums[3],
                            agreement_with_teacher=sums[2]/sums[3], seconds=time.time()-started))
        print(json.dumps(history[-1]), flush=True)
    torch.save(lora_state(blocks), args.output/'lora.pt')
    (args.output/'metrics.json').write_text(json.dumps(dict(
        model=base['model'], base_run=str(args.base), train=[str(p) for p in args.train],
        teacher=[str(p) for p in args.teacher], images=len(items), epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, rank=args.rank, seed=args.seed,
        adapters=len(params)//2, trainable_parameters=int(sum(p.numel() for p in params)),
        loss='cosine distance to the 224px feature + KL(teacher || student) on 80-template class probabilities',
        student_input='32px manifest file, model default preprocessing, original view, no augmentation',
        labels_used=False, history=history, peak_memory_mb=torch.cuda.max_memory_allocated()/2**20),
        indent=2)+'\n')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
