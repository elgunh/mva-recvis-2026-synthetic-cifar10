"""Encode competition images with an adapted DINOv2-g student (train_dino_selftrain.py).

Test split: the student encodes only the half it was not trained on; the
reflected view only on the rows that the reference view rule reflects.
Validation split: original view, with the accuracy of the cosine head (a check
that the adapted features still separate real photographs).
Saved: unit class-token features and head logits, in the layout of
encode_test_selftrain.py.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_dino_selftrain import Head, add_lora, features, load_dino
from train_test_selftrain import SEED, Images, folds, load_images

SELECTION = Path('runs/next3_selection_20261112.npz')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--split', choices=['test', 'validation'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--limit', type=int, help='Encode only the first rows (code check).')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new encoder output directory.')
    trained = json.loads((args.student/'metrics.json').read_text())
    images, ids, labels = load_images(args.split)
    if args.split == 'test':
        rows = np.flatnonzero(folds(len(ids), trained.get('fold_seed', SEED)) != trained['fold'])
        with np.load(SELECTION) as selection:
            assert np.array_equal(selection['test_ids'], ids)
            reflect = selection['test_mask'].astype(bool)
        views = ['original', 'reflected']
    else:
        rows, reflect, views = np.arange(len(ids)), np.zeros(len(ids), bool), ['original']
    if args.limit:
        rows = rows[:args.limit]
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, transform = load_dino(device, trained.get('image_size', 224))
    blocks = add_lora(model, trained['rank'])
    state = torch.load(args.student/'student.pt', map_location='cpu', weights_only=True)
    assert set(state['lora']) == {k for k in blocks.state_dict() if k.rsplit('.', 1)[-1] in ('down', 'up')}
    blocks.load_state_dict(state['lora'], strict=False)
    head = Head(state['head'], state['scale']).to(device).eval()
    loader = DataLoader(Images(images, rows, transform, False), batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=device.type == 'cuda')
    saved = {f'{view}_{kind}': [] for view in views for kind in ['features', 'logits']}
    order, reflected_order, started = [], [], time.time()
    for batch, (tensors, index) in enumerate(loader, 1):
        tensors, index = tensors.to(device), index.numpy()
        for view in views:
            x = tensors
            if view == 'reflected':
                keep = reflect[index]
                if not keep.any():
                    continue
                x = tensors[torch.from_numpy(keep).to(device)].flip(-1)
                reflected_order.append(index[keep])
            feature = F.normalize(features(model, x, device), dim=1)
            saved[f'{view}_features'].append(feature.cpu().numpy())
            saved[f'{view}_logits'].append(head(feature).cpu().numpy())
        order.append(index)
        if batch == 1 or batch % 100 == 0 or batch == len(loader):
            print(f'{args.split}: {min(batch*args.batch_size, len(rows))}/{len(rows)} {time.time()-started:.1f}s',
                  flush=True)
    arrays = {k: np.concatenate(v) for k, v in saved.items() if v}
    if 'reflected' in views:
        arrays['reflected_rows'] = np.concatenate(reflected_order) if reflected_order else np.zeros(0, np.int64)
        assert np.array_equal(arrays['reflected_rows'], rows[reflect[rows]])
    assert np.array_equal(np.concatenate(order), rows) and all(np.isfinite(v).all() for v in arrays.values())
    metrics = dict(model=trained['model'], student=str(args.student), student_data=trained.get('data', 'test'),
                   student_fold=trained['fold'], split=args.split, rows=int(len(rows)), views=views,
                   reflected_rows_mode='selected', reflected_rows=int(reflect[rows].sum()),
                   held_out_fold_only=args.split == 'test', seconds=time.time()-started)
    if args.split == 'validation':
        predicted = arrays['original_logits'].argmax(1)
        metrics.update(correct=int((predicted == labels[rows]).sum()), accuracy=float((predicted == labels[rows]).mean()))
        arrays['labels'] = labels[rows]
        print('validation head accuracy', metrics['accuracy'], flush=True)
    else:
        assert labels is None
    args.output.mkdir(parents=True)
    np.savez(args.output/f'{args.split}_features.npz', ids=ids[rows], rows=rows, **arrays)
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print('Completed', args.output, flush=True)


if __name__ == '__main__':
    main()
