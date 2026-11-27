"""X1 of the 16 November plan: does an own-generated set behave like the test set?

The frozen reference pipeline (public 97.565%) scores a labelled set that I
generated myself. Three checks, with thresholds written in EXPERIMENTS.md before
any Stable Diffusion 2.x image was scored:
  (a) mixing rate with the test images at least 0.880;
  (b) reference accuracy within 0.7 points of 97.565, and at least 4 of the 5
      contrasts with a known public difference have the right sign;
  (c) automobile/truck share among low-confidence images between 25% and 55%,
      and low-confidence rates ordered cage > origami > lego.
Labels come from my own prompts.
"""

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch

import synth_style_experiment as E
from external_cache import smooth
from generate_style_images import CLASS_NAMES
from graph_diffusion import neighbours

OUT = Path('runs/x1_signature_20261116')
PUBLIC_REFERENCE = .97565
THRESHOLDS = dict(mixing_min=.880, mixing_rows=3120, accuracy_tolerance=.007, ladder_min=4,
                  pair_share=[.25, .55], style_order=['cage', 'origami', 'lego'])
IDEAL_AT_3120 = .9415  # measured on 16 November: a test subsample against the rest of the test set


def config(name):
    return dict(root=Path(f'external_data/synth_style_{name}'),
                runs=[Path(f'runs/synth_{name}_bigg'), Path(f'runs/synth_{name}_siglip2')],
                dino=Path(f'runs/synth_{name}_dinov2_g'), judge=Path(f'runs/synth_{name}_judge'))


def signature(prob, style_prob):
    """Label-free error signature: where the pipeline is unsure and between which classes."""
    low = prob.max(1) < .9
    top2 = np.sort(np.argsort(-prob, axis=1)[:, :2], axis=1)
    pairs = collections.Counter(map(tuple, top2[low].tolist()))
    style, sure = style_prob.argmax(1), style_prob.max(1) > .5
    rates = {}
    for s, key in enumerate(E.STYLE_KEYS):
        group = (style == s) & sure
        rates[key] = float(low[group].mean()) if group.any() else None
    return dict(rows=int(len(prob)), low_confidence=int(low.sum()), low_confidence_rate=float(low.mean()),
                automobile_truck_share=float(pairs[(1, 9)]/max(int(low.sum()), 1)),
                top_pairs=[[f'{CLASS_NAMES[i]}/{CLASS_NAMES[j]}', int(c)] for (i, j), c in pairs.most_common(6)],
                style_guess_confident_fraction=float(sure.mean()),
                low_confidence_rate_by_guessed_style=rates)


def ordered(rates, keys):
    values = [rates[k] for k in keys]
    return bool(all(v is not None for v in values) and all(a > b for a, b in zip(values, values[1:])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, help='generator name: external_data/synth_style_<name>')
    parser.add_argument('--max-index', type=int, help='score only images with index below this (default: all)')
    parser.add_argument('--mixing-index', type=int, default=24, help='rows used for the mixing rate')
    parser.add_argument('--tag', default='')
    parser.add_argument('--no-union', action='store_true', help='skip the supplementary joint graph with the test set')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/f'{args.name}{args.tag}.json'
    if target.exists():
        raise FileExistsError('Use a new tag; results are never overwritten.')
    texts = E.load_texts()
    test = E.load_competition('test', texts)
    _, test_prob, _, control_error = E.reference_check(test, 'test', device)  # stops if the reference is not reproduced
    test_signature = signature(test_prob, E.class_probs(test['views'][0][0], texts[0], 'style'))

    data = E.load_external(config(args.name), texts)
    index = np.array([r['index'] for r in data['records']])
    inside = index < (args.max_index if args.max_index is not None else index.max()+1)
    keep = data['keep']['top2'] & inside
    style, y = data['style'], data['y']

    # (a) Same procedure as the anchors of 16 November: unfiltered rows against the test set without
    # the first n rows of a fixed permutation; those n rows give the ideal value for the same size.
    mix_rows = index < args.mixing_index
    n = int(mix_rows.sum())
    order = np.random.default_rng(0).permutation(len(test['ids']))
    rest = order[n:]
    mixing = E.mixing_rate(data['graph'][mix_rows], test['graph'][rest], device)
    ideal = float(E.mixing_rate(test['graph'][order[:n]], test['graph'][rest], device).mean())
    matched = n == THRESHOLDS['mixing_rows']
    mixing_mean = float(mixing.mean())
    mixing_pass = (mixing_mean >= THRESHOLDS['mixing_min'] if matched
                   else mixing_mean/ideal >= THRESHOLDS['mixing_min']/IDEAL_AT_3120)

    # (b) Known public contrasts and the reference accuracy, on rows kept by the judge (top 2).
    accuracy, contrasts = E.ladder(data, keep, device)
    signs = {k: bool(v > 0) for k, v in contrasts.items()}
    prob = smooth(data['P']['M0'][keep], *neighbours(data['graph'][keep], 20, .07, device, 512), device)
    correct = prob.argmax(1) == y[keep]
    assert abs(E.style_macro(correct, style[keep])-accuracy['reference']) < 1e-9

    # (c) Error signature with the same rule as on the test set (guessed style, not the prompt style).
    style_prob = E.class_probs(data['arrays'][0]['features'][keep], texts[0], 'style')
    pilot_signature = signature(prob, style_prob)
    low = prob.max(1) < .9
    pilot_signature.update(
        style_recognition_accuracy=float((style_prob.argmax(1) == style[keep]).mean()),
        low_confidence_rate_by_prompt_style=dict(zip(E.STYLE_KEYS, E.per_style(low.astype(float), style[keep]))))

    union = None
    if not args.no_union:  # supplementary, not part of the decision: the set scored inside the test graph
        features = np.concatenate([test['graph'], data['graph'][keep]])
        initial = np.concatenate([test['P']['M0'], data['P']['M0'][keep]])
        joint = smooth(initial, *neighbours(features, 20, .07, device, 512), device)[len(test['ids']):]
        union = dict(style_macro_accuracy=E.style_macro(joint.argmax(1) == y[keep], style[keep]),
                     accuracy=float((joint.argmax(1) == y[keep]).mean()),
                     automobile_truck_share=signature(joint, style_prob)['automobile_truck_share'],
                     low_confidence_rate=float((joint.max(1) < .9).mean()))

    lower, upper = THRESHOLDS['pair_share']
    checks = dict(
        a_mixing=bool(mixing_pass),
        b_accuracy=bool(abs(accuracy['reference']-PUBLIC_REFERENCE) <= THRESHOLDS['accuracy_tolerance']),
        b_ladder=bool(sum(signs.values()) >= THRESHOLDS['ladder_min']),
        c_pair_share=bool(lower <= pilot_signature['automobile_truck_share'] <= upper),
        c_style_order=ordered(pilot_signature['low_confidence_rate_by_guessed_style'], THRESHOLDS['style_order']))
    manifest = data['manifest']
    report = dict(
        name=args.name, max_index=args.max_index, rows=int(inside.sum()), kept_top2=int(keep.sum()),
        kept_top2_by_style=dict(zip(E.STYLE_KEYS, E.per_style(data['keep']['top2'][inside].astype(float), style[inside]))),
        kernel=manifest['kernel'], kernel_source=manifest.get('kernel_source', 'hires'),
        low_level=manifest['low_level'], generator_settings=manifest['generator_settings'],
        mixing=dict(rows=n, mean=mixing_mean, ideal_same_size=ideal, relative_to_ideal=mixing_mean/ideal,
                    chance=len(rest)/(len(rest)+n-1), threshold_matched_to_3120_rows=matched,
                    by_style=dict(zip(E.STYLE_KEYS, E.per_style(mixing, style[mix_rows])))),
        reference_style_macro_accuracy=accuracy['reference'], reference_accuracy=float(correct.mean()),
        accuracy_by_style=dict(zip(E.STYLE_KEYS, E.per_style(correct.astype(float), style[keep]))),
        accuracy_by_class={CLASS_NAMES[c]: float(correct[y[keep] == c].mean()) for c in range(10)},
        ladder_accuracy=accuracy, ladder_contrasts=contrasts, public_ladder_points=E.LADDER,
        ladder_signs_match=signs, ladder_matches=int(sum(signs.values())),
        signature=pilot_signature, test_signature=test_signature, union_with_test=union,
        thresholds=THRESHOLDS, public_reference=PUBLIC_REFERENCE, checks=checks, passed=bool(all(checks.values())),
        test_reference_control_max_probability_error=control_error,
        competition_test_labels_accessed=False, warehouse_data_used=False, device=device,
        seconds=round(time.time()-started, 1))
    target.write_text(json.dumps(report, indent=2)+'\n')
    brief = dict(name=args.name, rows=report['rows'], kept=report['kept_top2'], kernel=report['kernel'],
                 kernel_source=report['kernel_source'], mixing=round(mixing_mean, 4), ideal=round(ideal, 4),
                 reference_style_macro=round(accuracy['reference'], 4), accuracy=round(float(correct.mean()), 4),
                 contrasts={k: round(v, 4) for k, v in contrasts.items()}, ladder_matches=report['ladder_matches'],
                 pair_share=round(pilot_signature['automobile_truck_share'], 3),
                 test_pair_share=round(test_signature['automobile_truck_share'], 3),
                 top_pairs=pilot_signature['top_pairs'][:4], test_top_pairs=test_signature['top_pairs'][:4],
                 low_rate={k: pilot_signature['low_confidence_rate_by_guessed_style'][k] for k in THRESHOLDS['style_order']},
                 test_low_rate={k: test_signature['low_confidence_rate_by_guessed_style'][k] for k in THRESHOLDS['style_order']},
                 union=union, checks=checks, passed=report['passed'])
    print('X1_SIGNATURE', json.dumps(brief), flush=True)


if __name__ == '__main__':
    main()
