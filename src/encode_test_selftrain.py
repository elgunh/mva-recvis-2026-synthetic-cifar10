"""Encode competition images with an adapted bigG student (train_test_selftrain.py).

Test split: a student trained on one test fold encodes only the other fold;
a student trained on all test images or on CIFAR-10 train encodes every test
image. Validation split: every image, original view, with the accuracy.
Features and logits are saved for the original and the reflected view.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_lowres_student import add_lora, load_lora, load_model
from train_test_selftrain import SEED, Images, folds, load_images

SELECTION = Path('runs/next3_selection_20261112.npz')  # rows that the reference view rule reflects


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--split', choices=['test', 'validation'], required=True)
    parser.add_argument('--views', choices=['both', 'original', 'reflected'], default='both')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int, help='Encode only the first rows (code check).')
    parser.add_argument('--part', type=int, nargs=2, metavar=('INDEX', 'COUNT'),
                        help='Encode every COUNT-th row starting at INDEX; the parts together cover all rows.')
    parser.add_argument('--reflected-rows', choices=['all', 'selected'], default='all',
                        help='selected: encode the reflected view only for the rows the reference reflects.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new encoder output directory.')
    trained = json.loads((args.student/'metrics.json').read_text())
    base = json.loads((Path(trained['base_run'])/'metrics.json').read_text())
    images, ids, labels = load_images(args.split)
    data, fold, split_seed = trained.get('data', 'other'), trained.get('fold', -1), trained.get('fold_seed', SEED)
    held_out = args.split == 'test' and data == 'test' and fold >= 0
    split = folds(len(ids), split_seed)
    rows = np.flatnonzero(split != fold) if held_out else np.arange(len(ids))
    assert not held_out or not np.intersect1d(rows, np.flatnonzero(split == fold)).size
    if args.limit:
        rows = rows[:args.limit]
    if args.part:  # one of several encodes that together cover the rows (two GPUs)
        rows = rows[args.part[0]::args.part[1]]
    views = ['original', 'reflected'] if args.views == 'both' else [args.views]
    if args.split == 'validation':
        views = ['original']
    reflect = np.ones(len(ids), bool)
    if args.reflected_rows == 'selected':
        assert args.split == 'test'
        with np.load(SELECTION) as selection:
            assert np.array_equal(selection['test_ids'], ids)
            reflect = selection['test_mask'].astype(bool)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    model, preprocess, prototypes = load_model(base, device)
    blocks = add_lora(model, trained['rank'])
    load_lora(blocks, args.student/'lora.pt')
    model.set_grad_checkpointing(False)
    args.output.mkdir(parents=True)
    loader = DataLoader(Images(images, rows, preprocess, False), batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
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
            with torch.autocast('cuda'):
                encoded = model.encode_image(x)
            encoded = torch.nn.functional.normalize(encoded.float(), dim=1)
            saved[f'{view}_features'].append(encoded.cpu().numpy())
            saved[f'{view}_logits'].append((100*encoded@prototypes.T).cpu().numpy())
        order.append(index)
        if batch == 1 or batch % 100 == 0 or batch == len(loader):
            print(f'{args.split}: {min(batch*args.batch_size, len(rows))}/{len(rows)} {time.time()-started:.1f}s',
                  flush=True)
    arrays = {k: np.concatenate(v) for k, v in saved.items() if v}
    if 'reflected' in views:
        arrays['reflected_rows'] = np.concatenate(reflected_order) if reflected_order else np.zeros(0, np.int64)
        assert np.array_equal(arrays['reflected_rows'], rows[reflect[rows]])
    assert np.array_equal(np.concatenate(order), rows) and all(np.isfinite(v).all() for v in arrays.values())
    metrics = dict(model=base['model'], student=str(args.student), student_data=data,
                   student_fold=fold, student_split_seed=split_seed, split=args.split, rows=int(len(rows)), views=views,
                   reflected_rows_mode=args.reflected_rows, reflected_rows=int(reflect[rows].sum()),
                   held_out_fold_only=bool(held_out), seconds=time.time()-started)
    if args.split == 'validation':
        arrays['labels'] = labels[rows]
        metrics['accuracy'] = float((arrays['original_logits'].argmax(1) == labels[rows]).mean())
        metrics['correct'] = int((arrays['original_logits'].argmax(1) == labels[rows]).sum())
        print('validation accuracy', metrics['accuracy'], flush=True)
    else:
        assert labels is None
    np.savez(args.output/f'{args.split}_features.npz', ids=ids[rows], rows=rows, **arrays)
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print('Completed', args.output, flush=True)


if __name__ == '__main__':
    main()
