"""Second synthetic style experiment. Rules are fixed in SYNTH_STYLE_PLAN_V2.md.

Stages, in this order: select, confirm, export. Data, judge views, metric,
graph, Gate A, Gate B and guards are those of the first plan; only the
candidate heads (M4, M5) are new.
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
from external_cache import smooth
from graph_diffusion import neighbours, save_predictions

OUT = Path('runs/synth_style_v2_20261114')
CANDIDATE = Path('runs/synth_style_v2_candidate_20261114')
WDS = [1e-4, 1e-3, 1e-2, 1e-1]
ORDER = ['M5', 'M4']  # simpler first
SOURCES = E.SOURCES+['SYNTH_STYLE_PLAN_V2.md', 'synth_style_v2.py']
HEAD_INFO = ['wd', 'loss', 'closure_calls', 'train_accuracy', 'weight_norm']


def source_hashes():
    return {name: E.digest(Path(name).read_bytes()) for name in SOURCES}


def averaged_views(data):
    """Image feature per encoder under the fixed view rule (mean of two views on selected rows)."""
    result = []
    for original, flipped in data['views']:
        x = np.asarray(original, dtype=np.float64).copy()
        mean = x[data['selected']]+np.asarray(flipped, dtype=np.float64)
        x[data['selected']] = mean/np.linalg.norm(mean, axis=1, keepdims=True)
        result.append(x)
    return result


def inputs(data, texts, name):
    if name == 'M4':
        return np.concatenate([100*x@text['joint'].reshape(130, -1).T
                               for x, text in zip(averaged_views(data), texts)], axis=1)
    assert name == 'M5'
    style = E.prior(data['views'], data['selected'], texts, 'style')
    anchor = np.log(np.maximum(np.asarray(data['P']['M0'], dtype=np.float64), 1e-4))
    return np.concatenate([style, (style[:, :, None]*anchor[:, None, :]).reshape(len(style), -1)], axis=1)


def add_inputs(data, texts, names=ORDER):
    data['Z'] = {name: inputs(data, texts, name) for name in names}
    return data


def fit(z, p0, y, wd, device):
    mean, sd = z.mean(0), np.maximum(z.std(0), 1e-6)
    head = E.fit_head((z-mean)/sd, p0, y, wd, device)
    head.update(mean=mean, sd=sd)
    return head


def apply(z, p0, head):
    return E.apply_head((z-head['mean'])/head['sd'], p0, head)


def direction(train, test, view, configs, device):
    ktr, kte = train['keep'][view], test['keep'][view]
    graph = neighbours(test['graph'][kte], 20, .07, device, 512)
    y, heads = test['y'][kte], {}
    results = dict(M0=E.score(test['P']['M0'][kte], graph, y, device))
    for name, wd in configs:
        head = fit(train['Z'][name][ktr], train['P']['M0'][ktr], train['y'][ktr], wd, device)
        prior = apply(test['Z'][name][kte], test['P']['M0'][kte], head)
        results[(name, wd)] = E.score(prior, graph, y, device)
        heads[str((name, wd))] = {k: head[k] for k in HEAD_INFO}
    return dict(results=results, style=test['style'][kte], y=y, heads=heads)


def row(runs, config):
    return dict(mean=E.summarise(runs, config),
                by_generator=[E.style_macro(r['results'][config]['after'], r['style']) for r in runs],
                before_graph=[E.style_macro(r['results'][config]['before'], r['style']) for r in runs])


def stage_select(device, texts):
    first = json.loads((E.OUT/'frozen_selection.json').read_text())
    assert not first['gate_a_passed'] and not (E.OUT/'confirmation.json').exists()
    assert not OUT.exists(), 'The second selection is run once.'
    sets = {name: add_inputs(E.load_external(E.SETS[name], texts), texts) for name in E.DEV}
    pairs = [(sets[E.DEV[0]], sets[E.DEV[1]]), (sets[E.DEV[1]], sets[E.DEV[0]])]
    configs = [(name, wd) for name in ORDER for wd in WDS]
    runs = [direction(a, b, 'top2', configs, device) for a, b in pairs]
    table = {str(c): row(runs, c) for c in ['M0']+configs}
    chosen = max(configs, key=lambda c: (table[str(c)]['mean'], -ORDER.index(c[0]), c[1]))
    after, before = E.gain(runs, chosen), E.gain(runs, chosen, 'before')
    groups = [(r['results'][chosen]['after'], r['results']['M0']['after'], r['style'], r['y']) for r in runs]
    low, high = E.bootstrap(groups, [.025, .975])
    views, view_counts = {}, {}
    for view in ['top1', 'raw']:
        other = [direction(a, b, view, [chosen], device) for a, b in pairs]
        views[view] = float(np.mean(E.gain(other, chosen)))
        view_counts[view] = dict(
            fixes=[int((r['results'][chosen]['after'] & ~r['results']['M0']['after']).sum()) for r in other],
            new_errors=[int((~r['results'][chosen]['after'] & r['results']['M0']['after']).sum()) for r in other])
    style_gain = np.mean([[x-y for x, y in zip(E.per_style(r['results'][chosen]['after'], r['style']),
                                               E.per_style(r['results']['M0']['after'], r['style']))]
                          for r in runs], axis=0)
    hard = float(np.mean([style_gain[E.STYLE_KEYS.index(k)] for k in ['cage', 'lego', 'origami']]))
    changed = float(np.mean([E.style_macro(r['results'][chosen]['pred'] != r['results']['M0']['pred'], r['style'])
                             for r in runs]))
    transitions = sum((E.pair_counts(r['results']['M0']['pred'], r['results'][chosen]['pred']) for r in runs),
                      collections.Counter())
    fixes = [int((c & ~m).sum()) for c, m, _, _ in groups]
    new_errors = [int((~c & m).sum()) for c, m, _, _ in groups]
    per_class = [[float(np.mean(r['results'][k]['after'][r['y'] == c])) for c in range(10)]
                 for r in runs for k in ['M0', chosen]]
    gates = dict(mean_gain_at_least_0_30=bool(np.mean(after) >= .003), gain_on_each_generator=bool(min(after) > 0),
                 bootstrap_lower_above_0=bool(low > 0), top1_view_gain=bool(views['top1'] > 0),
                 uncleaned_view_gain=bool(views['raw'] > 0), before_graph_gain=bool(np.mean(before) > 0),
                 styles_not_worse_at_least_9=bool(int((style_gain >= 0).sum()) >= 9), hard_styles_gain=bool(hard > 0))
    result = dict(selected=chosen[0], selected_wd=chosen[1], selected_config=str(chosen), table=table,
                  eval_generators=[E.DEV[1], E.DEV[0]], gain_after_graph=after, gain_before_graph=before,
                  bootstrap_95=[low, high], sensitivity_view_gains=views, sensitivity_view_counts=view_counts,
                  style_gain=dict(zip(E.STYLE_KEYS, [float(v) for v in style_gain])), hard_styles_gain=hard,
                  fixes=fixes, new_errors=new_errors, rows=[int(len(r['y'])) for r in runs],
                  per_class_after_graph_m0_then_selected=per_class, changed_fraction_style_macro=changed,
                  top_changed_pairs=[[list(k), int(v)] for k, v in transitions.most_common(5)],
                  heads={k: v for r in runs for k, v in r['heads'].items()},
                  first_plan_m2_mean=first['table'][first['selected_config']]['mean'],
                  gates=gates, gate_a_passed=bool(all(gates.values())), confirmation_set_read=False,
                  source_sha256=source_hashes(),
                  first_plan_selection_sha256=E.digest((E.OUT/'frozen_selection.json').read_bytes()))
    OUT.mkdir(parents=True)
    E.write(OUT/'frozen_selection.json', result)
    print(json.dumps({k: result[k] for k in ['selected_config', 'table', 'gain_after_graph', 'bootstrap_95',
                                              'sensitivity_view_gains', 'sensitivity_view_counts', 'style_gain',
                                              'fixes', 'new_errors', 'top_changed_pairs', 'gates',
                                              'gate_a_passed']}, indent=2), flush=True)


def load_head(path):
    with np.load(path) as d:
        return dict(w=d['w'], b=d['b'], mean=d['mean'], sd=d['sd'], name=str(d['name']), wd=float(d['wd']))


def stage_confirm(device, texts):
    frozen_bytes = (OUT/'frozen_selection.json').read_bytes()
    frozen = json.loads(frozen_bytes)
    assert frozen['gate_a_passed'], 'Gate A failed: stop.'
    assert not (OUT/'confirmation.json').exists()
    name, wd = frozen['selected'], frozen['selected_wd']
    sets = [add_inputs(E.load_external(E.SETS[n], texts), texts, [name]) for n in E.DEV]
    kept = [s['keep']['top2'] for s in sets]
    head = fit(np.concatenate([s['Z'][name][k] for s, k in zip(sets, kept)]),
               np.concatenate([s['P']['M0'][k] for s, k in zip(sets, kept)]),
               np.concatenate([s['y'][k] for s, k in zip(sets, kept)]), wd, device)
    np.savez(OUT/'head.npz', w=head['w'], b=head['b'], mean=head['mean'], sd=head['sd'], name=name, wd=wd)
    del sets
    target = add_inputs(E.load_external(E.SETS[E.CONFIRM], texts), texts, [name])
    report = {}
    for view in ['top2', 'top1', 'raw']:
        keep = target['keep'][view]
        graph = neighbours(target['graph'][keep], 20, .07, device, 512)
        y, style = target['y'][keep], target['style'][keep]
        reference = E.score(target['P']['M0'][keep], graph, y, device)
        candidate = E.score(apply(target['Z'][name][keep], target['P']['M0'][keep], head), graph, y, device)
        difference = E.style_macro(candidate['after'], style)-E.style_macro(reference['after'], style)
        report[view] = dict(
            rows=int(keep.sum()), reference_style_macro=E.style_macro(reference['after'], style),
            candidate_style_macro=E.style_macro(candidate['after'], style),
            reference_accuracy=float(reference['after'].mean()), candidate_accuracy=float(candidate['after'].mean()),
            gain=difference, one_sided_bootstrap_5=E.bootstrap([(candidate['after'], reference['after'], style, y)], [.05])[0],
            gain_before_graph=E.style_macro(candidate['before'], style)-E.style_macro(reference['before'], style),
            fixes=int((candidate['after'] & ~reference['after']).sum()),
            new_errors=int((~candidate['after'] & reference['after']).sum()),
            changed=int((candidate['pred'] != reference['pred']).sum()),
            style_gain=dict(zip(E.STYLE_KEYS, [a-b for a, b in zip(E.per_style(candidate['after'], style),
                                                                 E.per_style(reference['after'], style))])),
            reference_by_style=dict(zip(E.STYLE_KEYS, E.per_style(reference['after'], style))),
            per_class_reference=[float(reference['after'][y == c].mean()) for c in range(10)],
            per_class_candidate=[float(candidate['after'][y == c].mean()) for c in range(10)])
    primary = report['top2']
    result = dict(config=frozen['selected_config'], generator=E.CONFIRM, primary_view='top2', **primary,
                  other_views_reported_not_gated={k: report[k] for k in ['top1', 'raw']},
                  head={k: head[k] for k in HEAD_INFO}, selection_sha256=E.digest(frozen_bytes),
                  gate_b_passed=bool(primary['gain'] > 0 and primary['one_sided_bootstrap_5'] > 0),
                  source_sha256=source_hashes())
    E.write(OUT/'confirmation.json', result)
    print(json.dumps(result, indent=2), flush=True)


def candidate_prior(data, head, texts):
    return apply(inputs(data, texts, head['name']), data['P']['M0'], head)


def stage_export(device, texts, force_review):
    frozen = json.loads((OUT/'frozen_selection.json').read_text())
    confirmation_bytes = (OUT/'confirmation.json').read_bytes()
    assert json.loads(confirmation_bytes)['gate_b_passed'], 'Gate B failed: stop.'
    if CANDIDATE.exists():
        raise FileExistsError('Preserve existing outputs.')
    head = load_head(OUT/'head.npz')
    assert head['name'] == frozen['selected'] and head['wd'] == frozen['selected_wd']
    guards = {}
    val = E.load_competition('validation', texts)
    graph, _, reference, _ = E.reference_check(val, 'validation', device)
    val_p = smooth(candidate_prior(val, head, texts), *graph, device)
    correct = int((val_p.argmax(1) == val['y']).sum())
    guards['source_validation'] = dict(
        correct=correct, reference_correct=int((reference == val['y']).sum()),
        fixes=int(((val_p.argmax(1) == val['y']) & (reference != val['y'])).sum()),
        new_errors=int(((val_p.argmax(1) != val['y']) & (reference == val['y'])).sum()),
        passed=bool(correct >= 9955))
    dn = E.load_external(E.DOMAINNET, texts, judge=False)
    dn_prior = candidate_prior(dn, head, texts)
    net, detail = 0, {}
    for domain, role in [('clipart', 'calibration'), ('painting', 'calibration'), ('sketch', 'fresh_check')]:
        mask = np.array([r['role'] == role and r['domain'] == domain for r in dn['records']])
        g = neighbours(dn['graph'][mask], 20, .07, device, 512)
        y = dn['y'][mask]
        old = smooth(dn['P']['M0'][mask], *g, device).argmax(1) == y
        new = smooth(dn_prior[mask], *g, device).argmax(1) == y
        detail[domain] = dict(rows=int(mask.sum()), reference=int(old.sum()), candidate=int(new.sum()))
        net += int(new.sum())-int(old.sum())
    guards['domainnet'] = dict(net=net, by_domain=detail, passed=bool(net >= -5))
    test = E.load_competition('test', texts)
    graph, _, reference, control_error = E.reference_check(test, 'test', device)
    probabilities = smooth(candidate_prior(test, head, texts), *graph, device)
    predicted = probabilities.argmax(1)
    changed = int((predicted != reference).sum())
    fraction, proxy = changed/len(predicted), frozen['changed_fraction_style_macro']
    pairs = E.pair_counts(reference, predicted).most_common(5)
    moves = collections.Counter((int(a), int(b)) for a, b in zip(reference, predicted) if a != b).most_common(8)
    proxy_pairs = [tuple(p) for p, _ in frozen['top_changed_pairs'][:3]]
    guessed = E.class_probs(test['views'][0][0], texts[0], 'style').argmax(1)
    guards['test_label_free'] = dict(
        changed=changed, changed_fraction=fraction, proxy_changed_fraction=proxy,
        top_changed_pairs=[[list(k), int(v)] for k, v in pairs], proxy_top3_pairs=[list(p) for p in proxy_pairs],
        top_moves_reference_to_candidate=[[list(k), int(v)] for k, v in moves],
        changed_by_guessed_style=dict(zip(E.STYLE_KEYS, [int(((predicted != reference) & (guessed == s)).sum())
                                                         for s in range(13)])),
        class_counts=np.bincount(predicted, minlength=10).tolist(),
        reference_class_counts=np.bincount(reference, minlength=10).tolist(),
        passed=bool(changed <= 2500 and .5*proxy <= fraction <= 2*proxy and pairs and pairs[0][0] in proxy_pairs))
    passed = all(g['passed'] for g in guards.values())
    print(json.dumps(guards, indent=2), flush=True)
    if not passed and not (force_review and guards['source_validation']['passed'] and guards['domainnet']['passed']):
        E.write(OUT/'export_blocked.json', dict(guards=guards, exported=False))
        print('Guards failed: nothing exported.', flush=True)
        return
    CANDIDATE.mkdir(parents=True)
    sample = pd.read_csv('data/sample_submission.csv')
    assert np.array_equal(test['ids'], sample['id']) and len(np.unique(test['ids'])) == len(test['ids'])
    save_predictions(CANDIDATE, 'test', dict(ids=test['ids']), probabilities)
    save_predictions(CANDIDATE, 'validation', dict(ids=val['ids'], labels=val['y']), val_p)
    csv = CANDIDATE/'submission.csv'
    pd.DataFrame(dict(id=test['ids'], label=predicted)).to_csv(csv, index=False)
    np.savez(CANDIDATE/'head.npz', **head)
    metrics = dict(status='ready_not_submitted', method=frozen['selected_config'], plan='SYNTH_STYLE_PLAN_V2.md',
                   guards=guards, all_guards_passed=bool(passed),
                   exported_for_manual_review_despite_test_guard=bool(not passed),
                   reference=str(E.REFERENCE), reference_public_accuracy=.97565, public_accuracy=None,
                   control_max_probability_error=control_error, test_rows=len(predicted),
                   changed_test_predictions=changed, submission_sha256=E.digest(csv.read_bytes()),
                   confirmation_sha256=E.digest(confirmation_bytes), true_test_labels_accessed=False,
                   warehouse_data_used=False, source_sha256=source_hashes())
    E.write(CANDIDATE/'metrics.json', metrics)
    print('Prepared candidate:', json.dumps({k: metrics[k] for k in ['status', 'method', 'changed_test_predictions',
                                                                        'submission_sha256']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['select', 'confirm', 'export'])
    parser.add_argument('--manual-review', action='store_true',
                        help='Export for review when only a label-free test guard failed (manual decision).')
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    texts = E.load_texts()
    started = time.time()
    if args.stage == 'select':
        stage_select(device, texts)
    elif args.stage == 'confirm':
        stage_confirm(device, texts)
    else:
        stage_export(device, texts, args.manual_review)
    print(f'Stage {args.stage} finished in {time.time()-started:.0f}s on {device}', flush=True)


if __name__ == '__main__':
    main()
