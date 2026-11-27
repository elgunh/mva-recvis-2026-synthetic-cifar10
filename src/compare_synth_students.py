"""Pair two saved supervised DINOv2 students on a disjoint generated seed range."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_dino_selftrain import Head, add_lora, load_dino
from train_synth_supervised import encode, load_sets
from x2_pilot_check import counts


@torch.inference_mode()
def score(folder, images, labels, indices, rows, device):
    metrics = json.loads((folder / 'metrics.json').read_text())
    if metrics['fold'] != -1:
        raise ValueError(f'{folder} is not a supervised-only student.')
    possible_train = indices >= metrics['val_seeds']
    if metrics['max_train_index'] is not None:
        possible_train &= indices < metrics['max_train_index']
    if metrics.get('exclude_train_range'):
        start, stop = metrics['exclude_train_range']
        possible_train &= (indices < start) | (indices >= stop)
    if possible_train[rows].any():
        raise ValueError(f'Holdout overlaps the possible training seeds of {folder}.')

    model, transform = load_dino(device, metrics.get('image_size', 224))
    blocks = add_lora(model, metrics['rank'])
    state = torch.load(folder / 'student.pt', map_location='cpu', weights_only=True)
    adapter_keys = {k for k in blocks.state_dict() if k.rsplit('.', 1)[-1] in ('down', 'up')}
    if set(state['lora']) != adapter_keys:
        raise ValueError(f'Unexpected adapter layout in {folder}.')
    blocks.load_state_dict(state['lora'], strict=False)
    head = Head(state['head'], state['scale']).to(device).eval()
    logits = head(encode(model, images, rows, transform, device).to(device)).cpu().numpy()
    if not np.isfinite(logits).all():
        raise ValueError(f'Nonfinite holdout logits in {folder}.')
    return logits.argmax(1) == labels[rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', required=True, dest='set_name')
    parser.add_argument('--seed-range', required=True, type=int, nargs=2, metavar=('START', 'STOP'))
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start, stop = args.seed_range
    if not 0 <= start < stop:
        parser.error('--seed-range requires 0 <= START < STOP.')
    if args.output.exists():
        raise FileExistsError(args.output)

    torch.set_num_threads(4)
    device = torch.device('cuda')
    images, labels, indices, _, top2, sets, ids = load_sets([args.set_name])
    rows = np.flatnonzero((indices >= start) & (indices < stop))
    if len(rows) != 130 * (stop - start) or len(set(ids[rows])) != len(rows):
        raise ValueError('The fixed holdout must contain every unique generated image.')
    kept = top2[rows]
    if not kept.any():
        raise ValueError('No holdout images passed the top-two judge.')

    old = score(args.incumbent, images, labels, indices, rows, device)
    new = score(args.candidate, images, labels, indices, rows, device)
    result = dict(dataset=sets[0], seed_range=[start, stop],
                  incumbent=str(args.incumbent), candidate=str(args.candidate))
    for name, mask in [('all', np.ones(len(rows), dtype=bool)), ('top2', kept)]:
        result[name] = dict(rows=int(mask.sum()), incumbent_errors=int((~old[mask]).sum()),
                            candidate_errors=int((~new[mask]).sum()), **counts(new[mask], old[mask]))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('FRESH_HOLDOUT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
