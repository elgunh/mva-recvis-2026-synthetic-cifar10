"""Cleaning judge for generated images.

A frozen DFN5B CLIP model that is not part of the classifier scores the
224px file with the 80 existing templates. The flags are written to a sidecar
file; the manifest is not changed. See SYNTH_STYLE_PLAN.md.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from clip_baseline import text_prototypes


class Hires(Dataset):
    def __init__(self, records, preprocess):
        self.records, self.preprocess = records, preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index]['hires_path']) as image:
            assert image.size == (224, 224)
            return self.preprocess(image.convert('RGB'))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=Path('runs/clip_vith14_dfn5b_imagenet80'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new judge output directory.')
    args.output.mkdir(parents=True)
    raw = (args.root/'manifest.json').read_bytes()
    records = json.loads(raw)['records']
    base = json.loads((args.base/'metrics.json').read_text())
    assert 'DFN5B' in base['model'], 'The judge must be the DFN model, outside the classifier.'
    precision = base.get('weight_precision', 'fp32')
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, preprocess = open_clip.create_model_and_transforms(
        base['model'], pretrained=None, precision=precision, device=device)
    model.eval()
    prototypes, prompts, source = text_prototypes(
        model, open_clip.get_tokenizer(base['model']), 'imagenet', device)
    assert prompts == base['prompts']
    loader = DataLoader(Hires(records, preprocess), batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=device.type == 'cuda')
    logits, started = [], time.time()
    for batch, images in enumerate(loader, 1):
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            encoded = model.encode_image(images.to(device))
        encoded = torch.nn.functional.normalize(encoded.float(), dim=1)
        logits.append((100*encoded@prototypes.T).cpu().numpy())
        if batch == 1 or batch % 50 == 0 or batch == len(loader):
            print(f'judge {min(batch*args.batch_size, len(records))}/{len(records)} '
                  f'{time.time()-started:.1f}s', flush=True)
    logits = np.concatenate(logits)
    assert np.isfinite(logits).all()
    labels = np.array([r['label'] for r in records])
    order = np.argsort(-logits, axis=1)
    top1 = order[:, 0] == labels
    top2 = top1 | (order[:, 1] == labels)
    styles = sorted({r['domain'] for r in records})
    domain = np.array([r['domain'] for r in records])
    np.savez(args.output/'judge.npz', ids=np.array([r['id'] for r in records]),
             logits=logits, top1=top1, top2=top2)
    metrics = dict(judge_model=base['model'], judge_input='224px PNG, model default preprocessing',
                   prompt_source=source, weight_precision=precision, rows=len(records),
                   manifest_sha256=hashlib.sha256(raw).hexdigest(),
                   keep_rate_top1=float(top1.mean()), keep_rate_top2=float(top2.mean()),
                   keep_rate_top2_by_style={s: float(top2[domain == s].mean()) for s in styles},
                   keep_rate_top1_by_style={s: float(top1[domain == s].mean()) for s in styles},
                   keep_rate_top2_by_class={str(c): float(top2[labels == c].mean()) for c in range(10)},
                   seconds=time.time()-started, primary_rule='prompt class in judge top 2')
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
