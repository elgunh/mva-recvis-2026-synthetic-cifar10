"""S1 rule on a fresh generated holdout for any combination of supervised students (23 November).

It reuses a holdout prepared by check_s1_fresh_20261120.py (frozen encodes, judge,
and the adapted-base parts bigg.npz, siglip.npz, dino.npz). Stage `encode` saves the
probabilities of further DINOv2-g students on the holdout rows (GPU). Stage `score`
rebuilds the adapted base inside the test graph exactly as the earlier script (with
the same controls) and scores the S1 rule for each requested combination of students
(normalised geometric mean), paired against a named incumbent. Labels are the prompt
classes of my own generated images.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import check_s1_fresh_20261120 as C
import selftrain_export as X
import synth_style_experiment as E
from external_cache import smooth
from graph_diffusion import neighbours
from train_dino_selftrain import Head, add_lora, features, load_dino
from x2_pilot_check import counts


def training_ids(folder):
    """Ids of the generated images that the student may have trained on."""
    meta = C.read_json(folder / 'metrics.json')
    assert meta['fold'] == -1, 'Only students trained on generated images alone.'
    ids = set()
    for item in meta['sets']:
        for row in C.read_json(Path(f"external_data/synth_style_{item['name']}") / 'manifest.json')['records']:
            index = row['index']
            used = index >= meta['val_seeds']
            if meta.get('max_train_index') is not None:
                used = used and index < meta['max_train_index']
            if meta.get('exclude_train_range'):
                start, stop = meta['exclude_train_range']
                used = used and not start <= index < stop
            if used:
                ids.add(row['id'])
    return ids


@torch.inference_mode()
def stage_encode(args, records, ids, kept, selected):
    images = C.read_pixels(args.root, records)
    device = torch.device('cuda')
    for item in args.student:
        name, folder = item.split('=', 1)
        folder = Path(folder)
        output = args.work / f'x4_student_{name}.npz'
        if output.exists():
            print('exists', output, flush=True)
            continue
        overlap = training_ids(folder) & set(ids.tolist())
        assert not overlap, f'{folder} trained on {len(overlap)} holdout images.'
        meta = C.read_json(folder / 'metrics.json')
        model, transform = load_dino(device, meta.get('image_size', 224))
        blocks = add_lora(model, meta['rank'])
        state = torch.load(folder / 'student.pt', map_location='cpu', weights_only=True)
        assert set(state['lora']) == {k for k in blocks.state_dict() if k.rsplit('.', 1)[-1] in ('down', 'up')}
        blocks.load_state_dict(state['lora'], strict=False)
        head = Head(state['head'], state['scale']).to(device).eval()

        def forward(batch):
            embedding = F.normalize(features(model, batch, device), dim=1)
            return embedding, head(embedding)

        arrays = C.encode(images, np.arange(len(images)), selected, transform, forward, device, 64, {})
        probabilities = X.view_rule(arrays, selected)[0]
        assert np.isfinite(probabilities).all()
        truth = np.array([row['label'] for row in records])
        print('STUDENT_ALONE', name, 'errors', int((probabilities.argmax(1) != truth).sum()), 'of', len(truth), flush=True)
        np.savez(output, ids=ids, probabilities=probabilities, student=str(folder))
        del model, blocks, head
        torch.cuda.empty_cache()


def target_base(args, ids, selected):
    """The adapted base (0.98048 pipeline) on the holdout rows, scored inside the test graph."""
    cache = args.work / 'x4_target_base.npz'
    if cache.exists():
        with np.load(cache) as data:
            assert np.array_equal(data['ids'], ids)
            return data['probabilities']
    with np.load(E.SELECTION) as selection:
        test_ids, test_selected = selection['test_ids'], selection['test_mask']
    with np.load(C.BASE / 'test_predictions.npz') as prediction:
        assert 'labels' not in prediction.files and np.array_equal(test_ids, prediction['ids'])
        saved_base = prediction['probabilities'].astype(np.float64)
    test, fresh = {}, {}
    for encoder in C.KEYS:
        parts = [X.view_rule(X.merge([Path(r['folder']) for r in pair], test_ids, test_selected, C.KEYS[encoder])[0],
                             test_selected) for pair in C.groups(encoder)]
        test[encoder] = C.combine(parts)
        with np.load(args.work / f'{encoder}.npz') as data:
            assert np.array_equal(data['ids'], ids) and np.array_equal(data['selected'], selected)
            fresh[encoder] = data['probabilities'], data['features']
    test_prior, test_feature = C.adapted_inputs(test)
    fresh_prior, fresh_feature = C.adapted_inputs(fresh)
    rebuilt = smooth(test_prior, *neighbours(test_feature, 20, .07, 'cpu', 512), 'cpu')
    assert not (rebuilt.argmax(1) != saved_base.argmax(1)).sum() and np.abs(rebuilt-saved_base).max() < 1e-4, \
        'Base control failed.'
    n = len(test_ids)
    joined = smooth(np.concatenate([test_prior, fresh_prior]),
                    *neighbours(np.concatenate([test_feature, fresh_feature]), 20, .07, 'cpu', 512), 'cpu')
    np.savez(cache, ids=ids, probabilities=joined[n:])
    return joined[n:]


def stage_score(args, records, ids, kept, selected):
    output = args.work / f'x4_result_{args.tag}.json'
    if output.exists():
        raise FileExistsError(output)
    base = target_base(args, ids, selected)
    labels = np.array([row['label'] for row in records])
    students = {}
    with np.load(args.work / 'dino.npz') as data:
        for item in args.saved or []:  # name=incumbent or name=candidate (students scored by the earlier script)
            name, key = item.split('=', 1)
            students[name] = np.log(np.maximum(data[key].astype(np.float64), 1e-12))
    for path in sorted(args.work.glob('x4_student_*.npz')):
        with np.load(path) as data:
            assert np.array_equal(data['ids'], ids)
            students[path.stem[len('x4_student_'):]] = np.log(np.maximum(data['probabilities'].astype(np.float64), 1e-12))

    def right(combo):
        mean = np.mean([students[name] for name in combo.split('+')], axis=0)
        return C.s1(base, E.softmax(mean)) == labels

    reference = right(args.incumbent)
    seeds = sorted({row['index'] for row in records})
    result = dict(seed_range=[seeds[0], seeds[-1]+1],
                  rows=int(len(ids)), base=str(C.BASE), incumbent=args.incumbent, students=sorted(students),
                  base_low_confidence_rows=int((base.max(1) < .9).sum()), combinations={})
    for combo in args.combos:
        ok = right(combo)
        entry = {}
        for name, mask in [('all', np.ones(len(ids), bool)), ('top2', kept)]:
            entry[name] = dict(rows=int(mask.sum()), errors=int((~ok[mask]).sum()),
                               incumbent_errors=int((~reference[mask]).sum()), **counts(ok[mask], reference[mask]))
        result['combinations'][combo] = entry
        print('X4', combo, json.dumps({k: {a: v[a] for a in ['errors', 'incumbent_errors', 'fixes', 'new_errors']}
                                       for k, v in entry.items()}), flush=True)
    output.write_text(json.dumps(result, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['encode', 'score'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--student', nargs='*', default=[], help='encode: name=student_folder')
    parser.add_argument('--saved', nargs='*', help='score: name=incumbent|candidate from dino.npz of the earlier script')
    parser.add_argument('--combos', nargs='*', default=[], help='score: for example a b a+b')
    parser.add_argument('--incumbent', help='score: combination used as the paired reference')
    parser.add_argument('--tag', default='1')
    args = parser.parse_args()
    torch.set_num_threads(8 if args.stage == 'score' else 4)
    fresh = C.load_fresh(args.root, args.work)
    (stage_score if args.stage == 'score' else stage_encode)(args, *fresh)


if __name__ == '__main__':
    main()
