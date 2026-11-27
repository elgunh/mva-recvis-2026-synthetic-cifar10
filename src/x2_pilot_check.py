"""Labelled check of the S1 rule on held-out generated images (seeds below 24).

The frozen reference pipeline scores the held-out images inside the test graph
(as in x1_union_order.py). The S1 rule of x2_export.py is then applied with the
supervised student: on rows where the reference is below the threshold, the
probabilities become the normalised geometric mean of reference and student.
Labels are the prompt classes of my own generated images. The student never
saw these seeds.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import synth_signature as S
import synth_style_experiment as E
from external_cache import smooth
from graph_diffusion import neighbours


def counts(new, old):
    fixes, errors = int((new & ~old).sum()), int((~new & old).sum())
    return dict(accuracy=float(new.mean()), fixes=fixes, new_errors=errors,
                z=(fixes-errors)/math.sqrt(max(fixes+errors, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', required=True, help='Encoded pilot set, for example sd21_768.')
    parser.add_argument('--students', type=Path, nargs='+', required=True, help='Student folders with validation_logits.npz.')
    parser.add_argument('--threshold', type=float, default=.9)
    parser.add_argument('--weight', type=float, default=.5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Results are never overwritten.')
    texts = E.load_texts()
    test = E.load_competition('test', texts)
    data = E.load_external(S.config(args.pilot), texts)
    n = len(test['ids'])
    features = np.concatenate([test['graph'], data['graph']])
    initial = np.concatenate([test['P']['M0'], data['P']['M0']])
    reference = smooth(initial, *neighbours(features, 20, .07, 'cpu', 512), 'cpu')[n:]
    position = {key: i for i, key in enumerate(data['ids'].tolist())}
    report = dict(pilot=args.pilot, threshold=args.threshold, student_weight=args.weight, students={})
    for folder in args.students:
        with np.load(folder/'validation_logits.npz') as d:
            rows = np.array([position[key] for key in d['ids'].tolist()])
            assert np.array_equal(d['labels'], data['y'][rows])
            student = E.softmax(d['logits'].astype(np.float64))
            kept = d['top2'].astype(bool)
        base, y, style = reference[rows], data['y'][rows], data['style'][rows]
        low = base.max(1) < args.threshold
        geometric = E.softmax((1-args.weight)*np.log(np.maximum(base, 1e-12))+args.weight*np.log(np.maximum(student, 1e-12)))
        s1 = base.copy()
        s1[low] = geometric[low]
        ok = lambda p: (p.argmax(1) == y)[kept]
        old = ok(base)
        result = dict(
            rows=int(len(rows)), kept_rows=int(kept.sum()), low_confidence_kept=int(low[kept].sum()),
            reference_accuracy=float(old.mean()), reference_style_macro=E.style_macro(old, style[kept]),
            student_alone=counts(ok(student), old), s1_rule=counts(ok(s1), old),
            mix_on_all_rows=counts(ok(geometric), old),
            s1_style_macro=E.style_macro(ok(s1), style[kept]),
            student_style_macro=E.style_macro(ok(student), style[kept]),
            reference_by_style=dict(zip(E.STYLE_KEYS, [round(v, 4) for v in E.per_style(old.astype(float), style[kept])])),
            student_by_style=dict(zip(E.STYLE_KEYS, [round(v, 4) for v in E.per_style(ok(student).astype(float), style[kept])])),
            s1_by_style=dict(zip(E.STYLE_KEYS, [round(v, 4) for v in E.per_style(ok(s1).astype(float), style[kept])])),
            student_accuracy_on_low_confidence=float((student.argmax(1) == y)[kept & low].mean()),
            reference_accuracy_on_low_confidence=float((base.argmax(1) == y)[kept & low].mean()))
        report['students'][str(folder)] = result
        print('X2_PILOT', folder, json.dumps({k: result[k] for k in [
            'kept_rows', 'low_confidence_kept', 'reference_accuracy', 'student_alone', 's1_rule', 'mix_on_all_rows',
            'reference_accuracy_on_low_confidence', 'student_accuracy_on_low_confidence']}), flush=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
