"""Build submission candidates from an adapted bigG student (train_test_selftrain.py).

The student's test features replace the frozen bigG features in the reference
pipeline. Three candidates are written, each in its own directory:

prior  student bigG probabilities in the initial mixture (0.75 bigG, 0.25
       SigLIP2, same view rule), reference graph.
graph  the same initial probabilities, and the student features also replace
       the bigG part of the graph feature.
half   mean of the reference initial probabilities and the `prior` ones,
       reference graph.

--encodes lists the encode directories of one student setting. Together they
must cover every test row once for the original and the reflected view (two
cross-fit halves, or one full encode, or one directory per view). The
reference CSV is not modified.
"""

import argparse
import collections
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import synth_style_experiment as E
import synth_style_v2 as V
from check_dino_geometry import geometry
from external_cache import smooth
from graph_diffusion import neighbours, save_predictions
from joint_graph_diffusion import fuse_features

KINDS = ['original_features', 'original_logits', 'reflected_features', 'reflected_logits']


def merge(folders, ids, selected, model_key):
    """Original view on every test row; reflected view at least on the rows the reference reflects."""
    arrays, seen, sources = {}, {k: np.zeros(len(ids), bool) for k in KINDS}, []
    for folder in folders:
        meta = json.loads((folder/'metrics.json').read_text())
        assert meta['split'] == 'test' and model_key in meta['model'], (folder, meta['model'])
        sources.append(dict(folder=str(folder), model=meta['model'], student=meta['student'],
                            student_data=meta['student_data'], student_fold=meta['student_fold'],
                            held_out_fold_only=meta['held_out_fold_only'], rows=meta['rows'], views=meta['views'],
                            reflected_rows_mode=meta.get('reflected_rows_mode', 'all')))
        with np.load(folder/'test_features.npz') as d:
            assert 'labels' not in d.files
            rows = d['rows']
            assert np.array_equal(d['ids'], ids[rows])
            for kind in KINDS:
                if kind not in d.files:
                    continue
                target = d['reflected_rows'] if kind.startswith('reflected') and 'reflected_rows' in d.files else rows
                values = d[kind]
                assert len(values) == len(target)
                if kind not in arrays:
                    arrays[kind] = np.zeros((len(ids), values.shape[1]), dtype=np.float32)
                assert not seen[kind][target].any(), 'A test row is covered twice.'
                arrays[kind][target] = values
                seen[kind][target] = True
    assert seen['original_features'].all() and seen['original_logits'].all(), 'Original view incomplete.'
    assert seen['reflected_features'][selected].all() and seen['reflected_logits'][selected].all(), \
        'Reflected view missing on rows that the reference reflects.'
    return arrays, sources


def view_rule(student, selected):
    """Probabilities and unit features under the reference view rule (mean of both views on selected rows)."""
    p = E.softmax(np.asarray(student['original_logits'], dtype=np.float64))
    p[selected] = .5*p[selected]+.5*E.softmax(np.asarray(student['reflected_logits'][selected], dtype=np.float64))
    feature = student['original_features'].astype(np.float64)
    mean = feature[selected]+student['reflected_features'][selected]
    feature[selected] = mean/np.linalg.norm(mean, axis=1, keepdims=True)
    return p, feature


def describe(predicted, reference, guessed):
    changed = predicted != reference
    counts, old = np.bincount(predicted, minlength=10), np.bincount(reference, minlength=10)
    return dict(
        changed_test_predictions=int(changed.sum()),
        top_moves_reference_to_candidate=[[list(k), int(v)] for k, v in collections.Counter(
            (int(a), int(b)) for a, b in zip(reference[changed], predicted[changed])).most_common(8)],
        changed_by_guessed_style=dict(zip(E.STYLE_KEYS, [int((changed & (guessed == s)).sum()) for s in range(13)])),
        class_counts=counts.tolist(), reference_class_counts=old.tolist(),
        largest_class_count_shift=int(np.abs(counts-old).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help='Short name of the student setting, for the output folders.')
    # Each flag may be given several times: one group of encodes (covering the test set) per split.
    # Groups are averaged (probabilities, and unit features renormalised). Absent: frozen encoder.
    parser.add_argument('--encodes', type=Path, nargs='+', action='append', default=[],
                        help='bigG student encodes.')
    parser.add_argument('--siglip-encodes', type=Path, nargs='+', action='append', default=[],
                        help='SigLIP2 student encodes.')
    parser.add_argument('--dino-encodes', type=Path, nargs='+', action='append', default=[],
                        help='DINOv2 student encodes for the graph.')
    parser.add_argument('--validation', type=Path, nargs='*', default=[], help='Validation encodes (reported only).')
    parser.add_argument('--tag', required=True, help='Date tag for the output folders, for example 20261115.')
    args = parser.parse_args()
    assert args.encodes or args.siglip_encodes or args.dino_encodes, 'Give at least one student.'
    outputs = {v: Path(f'runs/adapted_{args.name}_{v}_{args.tag}') for v in ['prior', 'graph', 'half']}
    for output in outputs.values():
        if output.exists():
            raise FileExistsError('Preserve existing outputs.')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    started = time.time()
    texts = E.load_texts()
    test = E.load_competition('test', texts)
    reference_graph, _, reference, control_error = E.reference_check(test, 'test', device)
    ids, selected = test['ids'], test['selected']
    # reference parts, to check that this script rebuilds the reference initial probabilities
    parts = []
    for (original, flipped), text in zip(test['views'], texts):
        p = E.class_probs(original, text, 'm0')
        p[selected] = .5*p[selected]+.5*E.class_probs(flipped, text, 'm0')
        parts.append(p)
    rebuilt = .75*parts[0]+.25*parts[1]
    assert np.abs(rebuilt/rebuilt.sum(1, keepdims=True)-test['P']['M0']).max() < 1e-4
    averaged = V.averaged_views(test)
    assert np.abs(fuse_features(averaged, np.array([.75, .25]))-test['base']).max() < 1e-4
    sources = []

    def combine(groups, key, probabilities, unit):
        if not groups:
            return probabilities, unit
        ps, fs = [], []
        for group in groups:
            student, used = merge(group, ids, selected, key)
            p, f = view_rule(student, selected)
            ps.append(p)
            fs.append(f)
            sources.extend(used)
        if len(groups) == 1:
            return ps[0], fs[0]
        f = np.mean(fs, axis=0)
        return np.mean(ps, axis=0), f/np.linalg.norm(f, axis=1, keepdims=True)

    big, feature = combine(args.encodes, 'PE-Core-bigG', parts[0], averaged[0])
    small, small_feature = combine(args.siglip_encodes, 'SigLIP2', parts[1], averaged[1])
    _, dino = combine(args.dino_encodes, 'dinov2', None, test['dino'])
    prior = .75*big+.25*small
    prior /= prior.sum(1, keepdims=True)
    half = .5*np.asarray(test['P']['M0'], dtype=np.float64)+.5*prior
    student_graph = neighbours(geometry(fuse_features([feature, small_feature], np.array([.75, .25])), dino, .25),
                               20, .07, device, 512)
    results = dict(prior=smooth(prior, *reference_graph, device), graph=smooth(prior, *student_graph, device),
                   half=smooth(half, *reference_graph, device))
    guessed = E.class_probs(test['views'][0][0], texts[0], 'style').argmax(1)
    student_alone = big.argmax(1)
    frozen_alone = parts[0].argmax(1)
    validation = []
    for folder in args.validation:
        meta = json.loads((folder/'metrics.json').read_text())
        validation.append(dict(folder=str(folder), student=meta['student'], correct=meta['correct'], rows=meta['rows']))
    common = dict(
        status='ready_not_submitted', experimental=True, student_setting=args.name, encodes=sources,
        bigg_student_groups=len(args.encodes), siglip2_student_groups=len(args.siglip_encodes),
        dinov2_student_groups=len(args.dino_encodes),
        student_validation_original_view=validation, frozen_bigg_validation_accuracy=.9935,
        student_alone_agrees_with_reference=float((student_alone == reference).mean()),
        frozen_bigg_alone_agrees_with_reference=float((frozen_alone == reference).mean()),
        student_alone_differs_from_frozen_bigg=int((student_alone != frozen_alone).sum()),
        prior_changed_before_graph=int((prior.argmax(1) != np.asarray(test['P']['M0']).argmax(1)).sum()),
        reference=str(E.REFERENCE), reference_public_accuracy=.97565, public_accuracy=None,
        control_max_probability_error=control_error, test_rows=len(reference), true_test_labels_accessed=False,
        warehouse_data_used=False, device=device,
        source_sha256={n: E.digest(Path(n).read_bytes()) for n in [
            'selftrain_export.py', 'train_test_selftrain.py', 'encode_test_selftrain.py', 'train_lowres_student.py']})
    print('student', json.dumps({k: common[k] for k in [
        'student_alone_agrees_with_reference', 'frozen_bigg_alone_agrees_with_reference',
        'student_alone_differs_from_frozen_bigg', 'prior_changed_before_graph',
        'student_validation_original_view']}), flush=True)
    sample = pd.read_csv('data/sample_submission.csv')
    assert np.array_equal(ids, sample['id']) and len(np.unique(ids)) == len(ids)
    for variant, probabilities in results.items():
        output, predicted = outputs[variant], probabilities.argmax(1)
        output.mkdir(parents=True)
        save_predictions(output, 'test', dict(ids=ids), probabilities)
        csv = output/'submission.csv'
        pd.DataFrame(dict(id=ids, label=predicted)).to_csv(csv, index=False)
        metrics = dict(common, variant=variant, **describe(predicted, reference, guessed),
                       submission_sha256=E.digest(csv.read_bytes()), seconds=time.time()-started)
        E.write(output/'metrics.json', metrics)
        print(variant, json.dumps({k: metrics[k] for k in [
            'changed_test_predictions', 'top_moves_reference_to_candidate', 'changed_by_guessed_style',
            'largest_class_count_shift', 'submission_sha256']}), flush=True)
    agreement = {f'{a}_vs_{b}': int((results[a].argmax(1) != results[b].argmax(1)).sum())
                 for a, b in [('prior', 'graph'), ('prior', 'half')]}
    print('differences between candidates', json.dumps(agreement), flush=True)


if __name__ == '__main__':
    main()
