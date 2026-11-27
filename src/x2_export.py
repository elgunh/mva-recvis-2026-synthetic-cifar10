"""File S1 of the 16 November plan: a model trained on my own labelled generated
images is used only where the current best file is unsure.

Base: the saved probabilities of the best file. On test rows where its maximum
probability is below --threshold, the new probabilities are the normalised
geometric mean of the base probabilities and those of the supervised student
(cosine head, reference view rule). All other rows keep the base label.
--check-student is a second student (other cleaning of its training set). It
does not change the file; it gives an agreement check:
among the rows where at least one of the two students changes the base label,
both must give the same new label in at least 70% of the cases.
"""

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import selftrain_export as X
import synth_style_experiment as E
from graph_diffusion import save_predictions

BEST = Path('runs/adapted_bagbigg_sigdino_graph_20261116')
PUBLIC_ROWS = 15323  # about 30% of the 51,077 test rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True)
    parser.add_argument('--tag', required=True)
    # May be given several times: one group of test encodes per student. The students are combined by a
    # normalised geometric mean before the mix with the base.
    parser.add_argument('--student', type=Path, nargs='+', action='append', required=True,
                        help='Test encodes of one student (together they cover every test row once).')
    parser.add_argument('--check-student', type=Path, nargs='+', help='Test encodes of the second student.')
    parser.add_argument('--base', type=Path, default=BEST)
    parser.add_argument('--base-public', type=float, default=.97937)
    parser.add_argument('--threshold', type=float, default=.9)
    parser.add_argument('--weight', type=float, default=.5, help='Weight of the student in the log-space mean.')
    parser.add_argument('--agreement-min', type=float, default=.7)
    args = parser.parse_args()
    output = Path(f'runs/x2_{args.name}_{args.tag}')
    if output.exists():
        raise FileExistsError('Preserve existing outputs.')
    texts = E.load_texts()
    test = E.load_competition('test', texts)
    ids, selected = test['ids'], test['selected']
    with np.load(args.base/'test_predictions.npz') as d:
        assert 'labels' not in d.files and np.array_equal(d['ids'], ids)
        base = d['probabilities'].astype(np.float64)
        base_label = d['predictions']
    assert np.array_equal(base.argmax(1), base_label)
    reference = pd.read_csv(E.REFERENCE/'submission.csv')['label'].to_numpy()
    low = base.max(1) < args.threshold

    def probabilities(group):
        model = json.loads((group[0]/'metrics.json').read_text())['model']
        key = next(k for k in ['dinov2', 'PE-Core-bigG', 'SigLIP2'] if k in model)
        arrays, used = X.merge(group, ids, selected, key)
        return X.view_rule(arrays, selected)[0], used

    def mixed(groups):
        if isinstance(groups[0], Path):  # a single group (the check student)
            groups = [groups]
        parts, used = [], []
        for group in groups:
            p, sources = probabilities(group)
            parts.append(np.log(np.maximum(p, 1e-12)))
            used += sources
        p = E.softmax(np.mean(parts, axis=0))
        q = base.copy()
        q[low] = E.softmax((1-args.weight)*np.log(np.maximum(base[low], 1e-12))
                           + args.weight*np.log(np.maximum(p[low], 1e-12)))
        return q, p, used

    q, student, sources = mixed(args.student)
    label = q.argmax(1)
    assert np.array_equal(label[~low], base_label[~low])
    moved = label != base_label
    sure = base.max(1) > .99
    metrics = dict(
        status='ready_not_submitted', experimental=True, name=args.name, base=str(args.base),
        base_public_accuracy=args.base_public, threshold=args.threshold, student_weight=args.weight,
        encodes=sources, student_groups=len(args.student), students_combined='normalised geometric mean',
        test_rows=int(len(ids)), low_confidence_rows=int(low.sum()),
        changed_from_base=int(moved.sum()),
        public_noise_points=100*math.sqrt(.3*max(int(moved.sum()), 1))/PUBLIC_ROWS,
        top_moves_base_to_candidate=[[list(k), int(v)] for k, v in collections.Counter(
            (int(a), int(b)) for a, b in zip(base_label[moved], label[moved])).most_common(8)],
        student_alone_agrees_with_base=float((student.argmax(1) == base_label).mean()),
        student_alone_agrees_with_base_where_base_above_0_99=float((student.argmax(1)[sure] == base_label[sure]).mean()),
        student_alone_agrees_with_base_on_low_confidence_rows=float((student.argmax(1)[low] == base_label[low]).mean()),
        changed_test_predictions=int((label != reference).sum()),
        class_counts=np.bincount(label, minlength=10).tolist(),
        base_class_counts=np.bincount(base_label, minlength=10).tolist(),
        reference=str(E.REFERENCE), reference_public_accuracy=.97565, public_accuracy=None,
        true_test_labels_accessed=False, warehouse_data_used=False,
        source_sha256={n: E.digest(Path(n).read_bytes()) for n in [
            'x2_export.py', 'train_synth_supervised.py', 'encode_dino_selftrain.py', 'selftrain_export.py']})
    passed = True
    if args.check_student:
        other = mixed(args.check_student)[0].argmax(1)
        union = moved | (other != base_label)
        agreement = float((label[union] == other[union]).mean()) if union.any() else 1.0
        passed = agreement >= args.agreement_min
        metrics.update(check_student=[str(p) for p in args.check_student], rows_changed_by_either=int(union.sum()),
                       rows_changed_by_both_to_the_same_label=int((moved & (label == other)).sum()),
                       agreement_on_changed_rows=agreement, agreement_min=args.agreement_min,
                       agreement_check_passed=bool(passed))
    output.mkdir(parents=True)
    save_predictions(output, 'test', dict(ids=ids), q)
    sample = pd.read_csv('data/sample_submission.csv')
    assert np.array_equal(ids, sample['id']) and len(np.unique(ids)) == len(ids)
    csv = output/'submission.csv'
    pd.DataFrame(dict(id=ids, label=label)).to_csv(csv, index=False)
    metrics['submission_sha256'] = E.digest(csv.read_bytes())
    E.write(output/'metrics.json', metrics)
    print('X2_EXPORT', json.dumps({k: metrics[k] for k in metrics if k not in ('encodes', 'source_sha256')}), flush=True)
    if not passed:
        print('X2_CHECK_FAILED: the two students disagree too often; the file is not used.', flush=True)


if __name__ == '__main__':
    main()
