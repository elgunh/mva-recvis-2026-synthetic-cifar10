"""Choose, without labels, the images that get a horizontally reflected view (Stage 1).

Rule fixed on 12 November 2026: the two zero-shot encoders disagree in their argmax, or the largest
0.75/0.25 mixed probability is below 0.9. The mask used in the competition
(runs/next3_selection_20261112.npz, 572 validation and 6,241 test rows) was written with an inline
command. This script applies the same rule to the saved zero-shot predictions; --check compares the
result with a saved mask.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from joint_graph_diffusion import mix_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs=2, required=True,
                        help='Zero-shot run folders of PE-Core bigG and SigLIP 2 (clip_baseline.py).')
    parser.add_argument('--weights', type=float, nargs=2, default=[.75, .25])
    parser.add_argument('--threshold', type=float, default=.9)
    parser.add_argument('--splits', nargs='+', default=['validation', 'test'])
    parser.add_argument('--output', type=Path, help='New .npz file for the mask.')
    parser.add_argument('--check', type=Path, help='Saved mask to compare with.')
    args = parser.parse_args()
    arrays, summary = {}, {}
    for split in args.splits:
        ids, logits = None, []
        for run in args.runs:
            with np.load(run / f'{split}_predictions.npz') as saved:
                if ids is None:
                    ids = saved['ids']
                assert np.array_equal(ids, saved['ids']), 'The two runs list different images.'
                logits.append(saved['logits'])
        mixed = mix_probabilities(logits, args.weights)
        mask = (logits[0].argmax(1) != logits[1].argmax(1)) | (mixed.max(1) < args.threshold)
        arrays[f'{split}_ids'], arrays[f'{split}_mask'] = ids, mask
        summary[split] = dict(rows=len(ids), selected=int(mask.sum()))
    if args.check:
        with np.load(args.check) as saved:
            for split in args.splits:
                assert np.array_equal(arrays[f'{split}_ids'], saved[f'{split}_ids'])
                summary[split]['equal_to_saved_mask'] = bool(
                    np.array_equal(arrays[f'{split}_mask'], saved[f'{split}_mask']))
    if args.output:
        if args.output.exists():
            raise FileExistsError('Use a new file name.')
        np.savez(args.output, **arrays)
    print(json.dumps(dict(policy='raw encoders disagree OR mixed maximum probability below the threshold',
                          weights=args.weights, threshold=args.threshold,
                          true_labels_used_for_selection=False, splits=summary), indent=2))


if __name__ == '__main__':
    main()
