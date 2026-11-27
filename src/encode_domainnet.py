"""Encode the fixed external images using an existing frozen encoder configuration."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import open_clip
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from clip_baseline import text_prototypes


class ExternalImages(Dataset):
    def __init__(self, root, records, preprocess):
        self.root, self.records, self.preprocess = root, records, preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.root/self.records[index]['path']) as image:
            image = image.convert('RGB').resize((32, 32), Image.Resampling.BICUBIC)
            return self.preprocess(image)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--roles', nargs='+', help='Optional subset of manifest roles.')
    parser.add_argument('--original-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new encoder output directory.')
    args.output.mkdir(parents=True)
    raw_manifest = (args.root/'manifest.json').read_bytes()
    records = json.loads(raw_manifest)['records']
    if args.roles:
        records = [r for r in records if r['role'] in args.roles]
    if not records:
        raise ValueError('No images selected.')
    base = json.loads((args.base/'metrics.json').read_text())
    precision = base.get('weight_precision','fp32')
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, preprocess = open_clip.create_model_and_transforms(
        base['model'], pretrained=None, precision=precision, device=device)
    model.eval()
    prototypes, prompts, source = text_prototypes(
        model, open_clip.get_tokenizer(base['model']), 'imagenet', device)
    clean = lambda s: re.sub(r'0x[0-9a-fA-F]+', '<address>', s)
    assert prompts == base['prompts'], 'Text templates differ from competition baseline.'
    assert clean(str(preprocess)) == clean(base['preprocess']), 'Preprocessing mismatch.'
    loader = DataLoader(ExternalImages(args.root, records, preprocess),
                        batch_size=args.batch_size, shuffle=False, num_workers=4,
                        pin_memory=device.type == 'cuda')
    features, flipped, logits, flip_logits = [], [], [], []
    started = time.time()
    for batch, images in enumerate(loader, 1):
        images = images.to(device)
        views = [(images, features, logits)]
        if not args.original_only:
            views.append((images.flip(-1), flipped, flip_logits))
        for view, ff, ll in views:
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                encoded = model.encode_image(view)
            encoded = torch.nn.functional.normalize(encoded.float(), dim=1)
            ff.append(encoded.cpu().numpy())
            ll.append((100*encoded@prototypes.T).cpu().numpy())
        if batch == 1 or batch % 20 == 0 or batch == len(loader):
            print(f'{base["model"]}: {min(batch*args.batch_size,len(records))}/'
                  f'{len(records)}, {len(views)} view(s), {time.time()-started:.1f}s', flush=True)
    saved = dict(ids=np.array([r['id'] for r in records]),
                 features=np.concatenate(features), logits=np.concatenate(logits))
    if not args.original_only:
        saved.update(flip_features=np.concatenate(flipped),flip_logits=np.concatenate(flip_logits))
    assert all(np.isfinite(saved[k]).all() for k in saved if k != 'ids')
    np.savez(args.output/'external_features.npz', **saved)
    metrics = dict(model=base['model'], checkpoint=base['checkpoint'],
                   weight_precision=precision, loaded_weight_dtype=str(next(model.parameters()).dtype),
                   precision_source='saved configuration' if 'weight_precision' in base else
                                    'original baseline default fp32 (no precision flag in saved command)',
                   preprocess=str(preprocess),
                   prompt_source=source, prompts=prompts, base_run=str(args.base),
                   manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
                   rows=len(records), feature_dimensions=saved['features'].shape[1],
                   seconds=time.time()-started, both_views_encoded=not args.original_only,
                   selected_roles=args.roles,
                   input_resize='PIL RGB bicubic 32x32')
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print('Completed', args.output, flush=True)


if __name__ == '__main__':
    main()
