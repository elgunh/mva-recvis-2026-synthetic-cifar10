"""Synthetic style proxy experiment. Rules are fixed in SYNTH_STYLE_PLAN.md.

Stages, in this order: control, instrument, select, confirm, export.
Each stage refuses to run when the previous one did not pass.
"""

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from check_dino_geometry import geometry
from external_cache import external_inputs, smooth
from generate_style_images import STYLES
from graph_diffusion import diffuse, load_split, neighbours, save_predictions
from joint_graph_diffusion import fuse_features, mix_probabilities

OUT = Path('runs/synth_style_20261114')
TEXT = [Path('runs/style_text_bigg_20261114'), Path('runs/style_text_siglip2_20261114')]
RAW = [Path('runs/pe_bigg_imagenet80_fp16'), Path('runs/siglip2_gopt_imagenet80_fp16')]
FLIP = [Path('runs/pe_bigg_selective_flip_v2'), Path('runs/siglip2_selective_flip_v2')]
REFERENCE = Path('runs/dinov2_giant_candidate')
SELECTION = Path('runs/next3_selection_20261112.npz')
DINO = dict(validation=Path('runs/dinov2_giant_pilot'), test=Path('runs/dinov2_giant_test'))
DEV, CONFIRM = ['sd15', 'kandinsky22'], 'sdxl_turbo'
SETS = {name: dict(root=Path(f'external_data/synth_style_{name}'),
                   runs=[Path(f'runs/synth_{name}_bigg'), Path(f'runs/synth_{name}_siglip2')],
                   dino=Path(f'runs/synth_{name}_dinov2_g'), judge=Path(f'runs/synth_{name}_judge'))
        for name in DEV+[CONFIRM]}
DOMAINNET = dict(root=Path('external_data/dfn_check'),
                 runs=[Path('runs/dfn_complementarity/encoder_0'), Path('runs/dfn_complementarity/encoder_1')],
                 dino=Path('runs/dinov2_giant_pilot'))
STYLE_KEYS = [k for k, _ in STYLES]
WDS = [1e-6, 1e-5, 1e-4, 1e-3]
ORDER = ['M1a', 'M1', 'M2', 'M3']  # simpler first
LADDER = dict(graph_vs_raw_bigg=.352, bigg_graph_vs_siglip2_graph=1.651,
              mixed_vs_bigg_prior=.189, joint_vs_bigg_graph=.294, reference_vs_selflip_joint=.202)
SOURCES = ['SYNTH_STYLE_PLAN.md', 'synth_style_experiment.py', 'generate_style_images.py',
           'prepare_synth_manifest.py', 'judge_synth.py', 'style_text_prototypes.py']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def softmax(z):
    z = z-z.max(1, keepdims=True)
    e = np.exp(z)
    return e/e.sum(1, keepdims=True)


def load_texts():
    texts = []
    for run, key in zip(TEXT, ['PE-Core-bigG', 'SigLIP2']):
        assert key in json.loads((run/'metrics.json').read_text())['model']
        with np.load(run/'text_prototypes.npz') as d:
            assert list(d['style_keys']) == STYLE_KEYS
            joint = d['style_joint'].astype(np.float64)
            mean13 = joint.mean(0)
            texts.append(dict(m0=d['imagenet80'].astype(np.float64), joint=joint,
                              m1a=mean13/np.linalg.norm(mean13, axis=1, keepdims=True)))
    return texts


def class_probs(x, text, kind):
    x = np.asarray(x, dtype=np.float64)
    if kind in ('m1', 'style'):
        p = softmax(100*x@text['joint'].reshape(130, -1).T).reshape(len(x), 13, 10)
        return p.sum(1) if kind == 'm1' else p.sum(2)
    return softmax(100*x@text[kind].T)


def prior(views, selected, texts, kind):
    """Mixed .75/.25 prior with the fixed view rule. views: (original, flip of selected rows)."""
    parts = []
    for (original, flipped), text in zip(views, texts):
        p = class_probs(original, text, kind)
        p[selected] = .5*p[selected]+.5*class_probs(flipped, text, kind)
        parts.append(p)
    mixed = .75*parts[0]+.25*parts[1]
    return mixed/mixed.sum(1, keepdims=True)


def load_external(cfg, texts, judge=True):
    raw = (cfg['root']/'manifest.json').read_bytes()
    manifest = json.loads(raw)
    records = manifest['records']
    ids = np.array([r['id'] for r in records])
    arrays = []
    for run, key in zip(cfg['runs'], ['PE-Core-bigG', 'SigLIP2']):
        meta = json.loads((run/'metrics.json').read_text())
        assert key in meta['model'], 'Encoder order must be bigG, SigLIP2.'
        assert meta['manifest_sha256'] == digest(raw) and meta['both_views_encoded']
        with np.load(run/'external_features.npz') as d:
            assert np.array_equal(d['ids'], ids)
            arrays.append({k: d[k] for k in ['features', 'flip_features', 'logits', 'flip_logits']})
    base, initial, _, selected = external_inputs(cfg['runs'], manifest)
    with np.load(cfg['dino']/'external_features.npz') as d:
        assert np.array_equal(d['ids'], ids) and np.array_equal(d['view_mask'], selected)
        dino = d['features']
    views = [(a['features'], a['flip_features'][selected]) for a in arrays]
    assert np.abs(prior(views, selected, texts, 'm0')-initial).max() < 1e-4
    data = dict(ids=ids, y=np.array([r['label'] for r in records]), records=records,
                arrays=arrays, base=base, dino=dino, graph=geometry(base, dino, .25),
                selected=selected, views=views,
                P=dict(M0=np.asarray(initial, dtype=np.float64),
                       M1=prior(views, selected, texts, 'm1'), M1a=prior(views, selected, texts, 'm1a')))
    if judge:
        data['style'] = np.array([STYLE_KEYS.index(r['domain']) for r in records])
        with np.load(cfg['judge']/'judge.npz') as d:
            assert np.array_equal(d['ids'], ids)
            data['keep'] = dict(top2=d['top2'], top1=d['top1'], raw=np.ones(len(ids), bool))
        data['manifest'] = manifest
    return data


def load_competition(split, texts):
    with np.load(SELECTION) as sel:
        mask, expected = sel[f'{split}_mask'], sel[f'{split}_ids']
    views = []
    for raw, flip in zip(RAW, FLIP):
        with np.load(raw/f'{split}_features.npz') as d:
            assert np.array_equal(d['ids'], expected)
            original = d['features']
        with np.load(flip/f'{split}_features.npz') as d:
            averaged = d['features']
        assert np.allclose(averaged[~mask], original[~mask], atol=1e-6)
        a, o = averaged[mask].astype(np.float64), original[mask].astype(np.float64)
        flipped = 2*(a*o).sum(1, keepdims=True)*a-o
        views.append((original, flipped/np.linalg.norm(flipped, axis=1, keepdims=True)))
    inputs = [load_split(r, r, split) for r in FLIP]
    assert np.array_equal(inputs[0]['ids'], expected) and np.array_equal(inputs[1]['ids'], expected)
    initial = mix_probabilities([a['logits'] for a in inputs], [.75, .25])
    base = fuse_features([a['features'] for a in inputs], np.array([.75, .25]))
    with np.load(DINO[split]/f'{split}_features.npz') as d:
        assert np.array_equal(d['ids'], expected) and 'labels' not in d.files
        dino = d['features']
    recovered = float(np.abs(prior(views, mask, texts, 'm0')-initial).max())
    assert recovered < 1e-4, recovered
    data = dict(ids=expected, base=base, dino=dino, graph=geometry(base, dino, .25),
                selected=mask, views=views, flip_recovery_max_error=recovered,
                P=dict(M0=np.asarray(initial, dtype=np.float64),
                       M1=prior(views, mask, texts, 'm1'), M1a=prior(views, mask, texts, 'm1a')))
    if split == 'validation':
        data['y'] = inputs[0]['labels']
    return data


def fit_head(x, p0, y, wd, device):
    """Prior-anchored residual head; convex, starts at the prior, L-BFGS in float64."""
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    anchor = torch.log(torch.as_tensor(p0, dtype=torch.float64, device=device).clamp_min(1e-4))
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    w = torch.zeros(10, x.shape[1], dtype=torch.float64, device=device, requires_grad=True)
    b = torch.zeros(10, dtype=torch.float64, device=device, requires_grad=True)
    optimiser = torch.optim.LBFGS([w, b], lr=1, max_iter=1000, tolerance_grad=1e-8,
                                  tolerance_change=1e-12, history_size=20, line_search_fn='strong_wolfe')
    state = dict(calls=0)

    def closure():
        optimiser.zero_grad()
        loss = torch.nn.functional.cross_entropy(anchor+x@w.T+b, y)+wd*(w**2).sum()
        loss.backward()
        state.update(calls=state['calls']+1, loss=float(loss))
        return loss
    optimiser.step(closure)
    with torch.no_grad():
        train_accuracy = float(((anchor+x@w.T+b).argmax(1) == y).double().mean())
    return dict(w=w.detach().cpu().numpy(), b=b.detach().cpu().numpy(), wd=wd,
                loss=state['loss'], closure_calls=state['calls'], train_accuracy=train_accuracy,
                weight_norm=float(w.detach().norm()))


def apply_head(x, p0, head):
    logits = np.log(np.maximum(np.asarray(p0, dtype=np.float64), 1e-4))
    return softmax(logits+np.asarray(x, dtype=np.float64)@head['w'].T+head['b'])


def style_macro(values, style):
    return float(np.mean([values[style == s].mean() for s in range(13) if (style == s).any()]))


def per_style(values, style):
    return [float(values[style == s].mean()) if (style == s).any() else None for s in range(13)]


def bootstrap(groups, quantiles, draws=2000, seed=20261114):
    """Quantiles of the pooled style-macro gain; resampling within style-class cells."""
    rng = np.random.default_rng(seed)
    total = np.zeros(draws)
    for cand, ref, style, y in groups:
        gain, used = np.zeros(draws), 0
        for s in range(13):
            in_style = style == s
            if not in_style.any():
                continue
            used += 1
            numerator = np.zeros(draws)
            for c in range(10):
                cell = in_style & (y == c)
                size = int(cell.sum())
                if not size:
                    continue
                a, b = cand[cell], ref[cell]
                up, down = int((a & ~b).sum()), int((~a & b).sum())
                sample = rng.multinomial(size, [up/size, down/size, 1-(up+down)/size], size=draws)
                numerator += sample[:, 0]-sample[:, 1]
            gain += numerator/in_style.sum()
        total += gain/used
    return [float(q) for q in np.quantile(total/len(groups), quantiles)]


def score(p0, graph, y, device):
    before = p0.argmax(1)
    after = smooth(p0, *graph, device).argmax(1)
    return dict(before=before == y, after=after == y, pred=after)


def direction(train, test, view, configs, device):
    """Fit on one generator, score on the other, on the rows kept by `view`."""
    ktr, kte = train['keep'][view], test['keep'][view]
    graph = neighbours(test['graph'][kte], 20, .07, device, 512)
    y, results, heads = test['y'][kte], {}, {}
    for config in configs:
        name, wd = config if isinstance(config, tuple) else (config, None)
        if name in ('M0', 'M1', 'M1a'):
            p0 = test['P'][name][kte]
        else:
            anchor = 'M0' if name == 'M2' else 'M1'
            head = fit_head(train['graph'][ktr], train['P'][anchor][ktr], train['y'][ktr], wd, device)
            p0 = apply_head(test['graph'][kte], test['P'][anchor][kte], head)
            heads[str(config)] = {k: v for k, v in head.items() if k not in ('w', 'b')}
        results[config] = score(p0, graph, y, device)
    return dict(results=results, style=test['style'][kte], y=y, heads=heads)


def summarise(runs, config):
    """Mean style-macro accuracy after the graph over the two directions."""
    return float(np.mean([style_macro(r['results'][config]['after'], r['style']) for r in runs]))


def gain(runs, config, key='after'):
    return [style_macro(r['results'][config][key], r['style'])-style_macro(r['results']['M0'][key], r['style'])
            for r in runs]


def pair_counts(old, new):
    changed = old != new
    counter = collections.Counter(tuple(sorted((int(a), int(b)))) for a, b in zip(old[changed], new[changed]))
    return counter


def source_hashes():
    return {name: digest(Path(name).read_bytes()) for name in SOURCES}


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2)+'\n')


def reference_check(data, split, device):
    graph = neighbours(data['graph'], 20, .07, device, 512)
    p = smooth(data['P']['M0'], *graph, device)
    with np.load(REFERENCE/f'{split}_predictions.npz') as ref:
        assert np.array_equal(ref['ids'], data['ids'])
        assert split == 'validation' or 'labels' not in ref.files
        same = bool(np.array_equal(p.argmax(1), ref['predictions']))
        error = float(np.abs(p-ref['probabilities']).max())
        reference = ref['predictions']
    assert same, f'{split}: the reference control was not reproduced.'
    return graph, p, reference, error


def stage_control(device, texts):
    OUT.mkdir(parents=True, exist_ok=False)
    report = {}
    for split in ['validation', 'test']:
        data = load_competition(split, texts)
        _, p, reference, error = reference_check(data, split, device)
        report[split] = dict(rows=len(reference), labels_identical=True, max_probability_error=error,
                             flip_recovery_max_error=data['flip_recovery_max_error'],
                             class_counts=np.bincount(reference, minlength=10).tolist())
        if split == 'validation':
            report[split]['accuracy'] = float((p.argmax(1) == data['y']).mean())
    report.update(reference=str(REFERENCE), chunk=512, source_sha256=source_hashes(),
                  reference_csv_sha256=digest((REFERENCE/'submission.csv').read_bytes()))
    write(OUT/'control.json', report)
    print(json.dumps(report, indent=2), flush=True)


@torch.inference_mode()
def mixing_rate(proxy, test, device, k=20):
    t = torch.nn.functional.normalize(torch.as_tensor(test, dtype=torch.float32, device=device), dim=1)
    p = torch.nn.functional.normalize(torch.as_tensor(proxy, dtype=torch.float32, device=device), dim=1)
    rates = []
    for offset in range(0, len(p), 512):
        q = p[offset:offset+512]
        inside = q@p.T
        rows = torch.arange(len(q), device=device)
        inside[rows, offset+rows] = -torch.inf
        index = torch.cat([q@t.T, inside], 1).topk(k, dim=1).indices
        rates.append((index < len(t)).float().mean(1).cpu())
    return torch.cat(rates).numpy()


def ladder(data, keep, device):
    a, style, y = data['arrays'], data['style'][keep], data['y'][keep]
    logits = [x['logits'][keep] for x in a]
    macro = lambda pred: style_macro(pred == y, style)
    single = []
    for i in range(2):
        graph = neighbours(a[i]['features'][keep], 20, .07, device, 512)
        single.append(macro(diffuse(logits[i], *graph, .5, device)[0].argmax(1)))
    joint = neighbours(fuse_features([x['features'][keep] for x in a], np.array([.75, .25])), 20, .07, device, 512)
    mixed = macro(smooth(mix_probabilities(logits, [.75, .25]), *joint, device).argmax(1))
    bigg_prior = macro(diffuse(logits[0], *joint, .5, device)[0].argmax(1))
    selflip = macro(smooth(data['P']['M0'][keep], *neighbours(data['base'][keep], 20, .07, device, 512), device).argmax(1))
    final = macro(smooth(data['P']['M0'][keep], *neighbours(data['graph'][keep], 20, .07, device, 512), device).argmax(1))
    raw = macro(logits[0].argmax(1))
    accuracy = dict(raw_bigg=raw, bigg_graph=single[0], siglip2_graph=single[1], joint_mixed=mixed,
                    joint_bigg_prior=bigg_prior, selflip_joint=selflip, reference=final)
    contrasts = dict(graph_vs_raw_bigg=single[0]-raw, bigg_graph_vs_siglip2_graph=single[0]-single[1],
                     mixed_vs_bigg_prior=mixed-bigg_prior, joint_vs_bigg_graph=mixed-single[0],
                     reference_vs_selflip_joint=final-selflip)
    return accuracy, contrasts


def stage_instrument(device, texts):
    assert (OUT/'control.json').exists() and not (OUT/'instrument.json').exists()
    test = load_competition('test', texts)
    report, pooled = dict(generators={}), collections.defaultdict(list)
    for name in DEV:
        data = load_external(SETS[name], texts)
        keep = data['keep']['top2']
        style = data['style']
        guessed = class_probs(data['arrays'][0]['features'], texts[0], 'style').argmax(1)
        mixing = mixing_rate(data['graph'][keep], test['graph'], device)
        accuracy, contrasts = ladder(data, keep, device)
        confidence = smooth(data['P']['M0'][keep], *neighbours(data['graph'][keep], 20, .07, device, 512), device).max(1)
        signs = {k: bool(v > 0) for k, v in contrasts.items()}
        for k, v in contrasts.items():
            pooled[k].append(v)
        report['generators'][name] = dict(
            rows=len(keep), kept_top2=int(keep.sum()), kept_top1=int(data['keep']['top1'].sum()),
            kept_top2_by_style=dict(zip(STYLE_KEYS, per_style(keep.astype(float), style))),
            style_recognition_accuracy=float((guessed == style).mean()),
            style_recognition_by_style=dict(zip(STYLE_KEYS, per_style((guessed == style).astype(float), style))),
            low_level=data['manifest']['low_level'], kernel=data['manifest']['kernel'],
            mixing_rate_mean=float(mixing.mean()),
            mixing_rate_by_style=dict(zip(STYLE_KEYS, per_style(mixing, style[keep]))),
            m0_accuracy=accuracy, ladder_contrasts=contrasts, ladder_signs_match=signs,
            ladder_matches=int(sum(signs.values())),
            uncertain_rate_by_style=dict(zip(STYLE_KEYS, per_style((confidence < .9).astype(float), style[keep]))))
        print(name, json.dumps({k: report['generators'][name][k] for k in
                                ['kept_top2', 'style_recognition_accuracy', 'mixing_rate_mean',
                                 'm0_accuracy', 'ladder_contrasts']}), flush=True)
    pooled_contrasts = {k: float(np.mean(v)) for k, v in pooled.items()}
    pooled_matches = int(sum(v > 0 for v in pooled_contrasts.values()))
    g = report['generators']
    checks = dict(
        mixing=bool(max(g[n]['mixing_rate_mean'] for n in DEV) >= .10),
        m0_range=bool(all(.93 <= g[n]['m0_accuracy']['reference'] <= .998 for n in DEV)),
        ladder=bool(pooled_matches >= 4 and all(g[n]['ladder_matches'] >= 3 for n in DEV)))
    report.update(pooled_ladder_contrasts=pooled_contrasts, pooled_ladder_matches=pooled_matches,
                  public_ladder_points=LADDER, chance_mixing_rate=len(test['ids'])/(len(test['ids'])+8320),
                  checks=checks, passed=bool(all(checks.values())), candidate_metrics_computed=False,
                  source_sha256=source_hashes())
    write(OUT/'instrument.json', report)
    print('Instrument checks:', checks, 'pooled ladder:', pooled_contrasts, flush=True)


def stage_select(device, texts):
    assert json.loads((OUT/'instrument.json').read_text())['passed'], 'Instrument checks failed: stop.'
    assert not (OUT/'frozen_selection.json').exists()
    sets = {name: load_external(SETS[name], texts) for name in DEV}
    pairs = [(sets[DEV[0]], sets[DEV[1]]), (sets[DEV[1]], sets[DEV[0]])]
    base = ['M0', 'M1a', 'M1']+[('M2', wd) for wd in WDS]
    runs = [direction(a, b, 'top2', base, device) for a, b in pairs]
    table = {str(c): dict(mean=summarise(runs, c), by_generator=[style_macro(r['results'][c]['after'], r['style']) for r in runs],
                          before_graph=[style_macro(r['results'][c]['before'], r['style']) for r in runs]) for c in base}
    best_wd = max(WDS, key=lambda wd: (table[str(('M2', wd))]['mean'], wd))
    candidates = {'M1a': 'M1a', 'M1': 'M1', 'M2': ('M2', best_wd)}
    reference = table['M0']['mean']
    if table['M1']['mean'] > reference and table[str(('M2', best_wd))]['mean'] > reference:
        extra = [direction(a, b, 'top2', ['M0', ('M3', best_wd)], device) for a, b in pairs]
        for run, more in zip(runs, extra):
            run['results'][('M3', best_wd)] = more['results'][('M3', best_wd)]
            run['heads'].update(more['heads'])
        c = ('M3', best_wd)
        table[str(c)] = dict(mean=summarise(runs, c), by_generator=[style_macro(r['results'][c]['after'], r['style']) for r in runs],
                             before_graph=[style_macro(r['results'][c]['before'], r['style']) for r in runs])
        candidates['M3'] = c
    chosen_name = max(candidates, key=lambda n: (table[str(candidates[n])]['mean'], -ORDER.index(n)))
    chosen = candidates[chosen_name]
    after, before = gain(runs, chosen), gain(runs, chosen, 'before')
    groups = [(r['results'][chosen]['after'], r['results']['M0']['after'], r['style'], r['y']) for r in runs]
    low, high = bootstrap(groups, [.025, .975])
    views = {}
    for view in ['top1', 'raw']:
        other = [direction(a, b, view, ['M0', chosen], device) for a, b in pairs]
        views[view] = float(np.mean(gain(other, chosen)))
    style_gain = np.mean([[x-y for x, y in zip(per_style(r['results'][chosen]['after'], r['style']),
                                               per_style(r['results']['M0']['after'], r['style']))] for r in runs], axis=0)
    hard = float(np.mean([style_gain[STYLE_KEYS.index(k)] for k in ['cage', 'lego', 'origami']]))
    changed = float(np.mean([style_macro(r['results'][chosen]['pred'] != r['results']['M0']['pred'], r['style']) for r in runs]))
    transitions = sum((pair_counts(r['results']['M0']['pred'], r['results'][chosen]['pred']) for r in runs), collections.Counter())
    fixes = int(sum((c & ~m).sum() for c, m, _, _ in groups))
    new_errors = int(sum((~c & m).sum() for c, m, _, _ in groups))
    per_class = [[float(np.mean(r['results'][k]['after'][r['y'] == c])) for c in range(10)] for r in runs for k in ['M0', chosen]]
    gates = dict(mean_gain_at_least_0_30=bool(np.mean(after) >= .003), gain_on_each_generator=bool(min(after) > 0),
                 bootstrap_lower_above_0=bool(low > 0), top1_view_gain=bool(views['top1'] > 0),
                 uncleaned_view_gain=bool(views['raw'] > 0), before_graph_gain=bool(np.mean(before) > 0),
                 styles_not_worse_at_least_9=bool(int((style_gain >= 0).sum()) >= 9), hard_styles_gain=bool(hard > 0))
    result = dict(selected=chosen_name, selected_config=str(chosen), selected_wd=best_wd if chosen_name in ('M2', 'M3') else None,
                  table=table, eval_generators=[DEV[1], DEV[0]], gain_after_graph=after, gain_before_graph=before,
                  bootstrap_95=[low, high], sensitivity_view_gains=views,
                  style_gain=dict(zip(STYLE_KEYS, [float(v) for v in style_gain])), hard_styles_gain=hard,
                  fixes=fixes, new_errors=new_errors, rows=[int(len(r['y'])) for r in runs],
                  per_class_after_graph_m0_then_selected=per_class,
                  changed_fraction_style_macro=changed,
                  top_changed_pairs=[[list(k), int(v)] for k, v in transitions.most_common(5)],
                  heads={k: v for r in runs for k, v in r['heads'].items()},
                  gates=gates, gate_a_passed=bool(all(gates.values())),
                  confirmation_set_read=False, source_sha256=source_hashes(),
                  instrument_sha256=digest((OUT/'instrument.json').read_bytes()))
    write(OUT/'frozen_selection.json', result)
    print(json.dumps({k: result[k] for k in ['selected_config', 'table', 'gain_after_graph', 'bootstrap_95',
                                              'sensitivity_view_gains', 'style_gain', 'fixes', 'new_errors',
                                              'gates', 'gate_a_passed']}, indent=2), flush=True)


def selected_config(frozen):
    name = frozen['selected']
    return name if name in ('M1', 'M1a') else (name, frozen['selected_wd'])


def stage_confirm(device, texts):
    frozen_bytes = (OUT/'frozen_selection.json').read_bytes()
    frozen = json.loads(frozen_bytes)
    assert frozen['gate_a_passed'], 'Gate A failed: stop.'
    assert not (OUT/'confirmation.json').exists()
    config = selected_config(frozen)
    sets = {name: load_external(SETS[name], texts) for name in DEV}
    target = load_external(SETS[CONFIRM], texts)
    keep = target['keep']['top2']
    graph = neighbours(target['graph'][keep], 20, .07, device, 512)
    y, style = target['y'][keep], target['style'][keep]
    head_info = None
    if isinstance(config, tuple):
        anchor = 'M0' if config[0] == 'M2' else 'M1'
        rows = [(s['graph'][s['keep']['top2']], s['P'][anchor][s['keep']['top2']], s['y'][s['keep']['top2']]) for s in sets.values()]
        head = fit_head(np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows]),
                        np.concatenate([r[2] for r in rows]), config[1], device)
        np.savez(OUT/'head.npz', w=head['w'], b=head['b'], wd=config[1], anchor=anchor)
        head_info = {k: v for k, v in head.items() if k not in ('w', 'b')}
        candidate_prior = apply_head(target['graph'][keep], target['P'][anchor][keep], head)
    else:
        candidate_prior = target['P'][config][keep]
    reference = score(target['P']['M0'][keep], graph, y, device)
    candidate = score(candidate_prior, graph, y, device)
    difference = style_macro(candidate['after'], style)-style_macro(reference['after'], style)
    lower = bootstrap([(candidate['after'], reference['after'], style, y)], [.05])[0]
    result = dict(config=str(config), generator=CONFIRM, rows=int(keep.sum()),
                  reference_style_macro=style_macro(reference['after'], style),
                  candidate_style_macro=style_macro(candidate['after'], style),
                  reference_accuracy=float(reference['after'].mean()), candidate_accuracy=float(candidate['after'].mean()),
                  gain=difference, one_sided_bootstrap_5=lower,
                  gain_before_graph=style_macro(candidate['before'], style)-style_macro(reference['before'], style),
                  fixes=int((candidate['after'] & ~reference['after']).sum()),
                  new_errors=int((~candidate['after'] & reference['after']).sum()),
                  style_gain=dict(zip(STYLE_KEYS, [a-b for a, b in zip(per_style(candidate['after'], style), per_style(reference['after'], style))])),
                  head=head_info, selection_sha256=digest(frozen_bytes),
                  gate_b_passed=bool(difference > 0 and lower > 0), source_sha256=source_hashes())
    write(OUT/'confirmation.json', result)
    print(json.dumps(result, indent=2), flush=True)


def candidate_prior(data, config, head):
    if isinstance(config, tuple):
        return apply_head(data['graph'], data['P'][str(head['anchor'])], head)
    return data['P'][config]


def stage_export(device, texts, force_review):
    frozen = json.loads((OUT/'frozen_selection.json').read_text())
    confirmation_bytes = (OUT/'confirmation.json').read_bytes()
    assert json.loads(confirmation_bytes)['gate_b_passed'], 'Gate B failed: stop.'
    output = Path('runs/synth_style_candidate_20261114')
    if output.exists():
        raise FileExistsError('Preserve existing outputs.')
    config = selected_config(frozen)
    head = None
    if isinstance(config, tuple):
        with np.load(OUT/'head.npz') as d:
            head = dict(w=d['w'], b=d['b'], anchor=str(d['anchor']))
    guards = {}
    val = load_competition('validation', texts)
    graph, _, reference, _ = reference_check(val, 'validation', device)
    val_p = smooth(candidate_prior(val, config, head), *graph, device)
    correct = int((val_p.argmax(1) == val['y']).sum())
    guards['source_validation'] = dict(correct=correct, reference_correct=int((reference == val['y']).sum()),
                                       fixes=int(((val_p.argmax(1) == val['y']) & (reference != val['y'])).sum()),
                                       new_errors=int(((val_p.argmax(1) != val['y']) & (reference == val['y'])).sum()),
                                       passed=bool(correct >= 9955))
    dn = load_external(DOMAINNET, texts, judge=False)
    net, detail = 0, {}
    for domain, role in [('clipart', 'calibration'), ('painting', 'calibration'), ('sketch', 'fresh_check')]:
        mask = np.array([r['role'] == role and r['domain'] == domain for r in dn['records']])
        sub = dict(graph=dn['graph'][mask], P={k: v[mask] for k, v in dn['P'].items()})
        g = neighbours(sub['graph'], 20, .07, device, 512)
        y = dn['y'][mask]
        old = smooth(sub['P']['M0'], *g, device).argmax(1) == y
        new = smooth(candidate_prior(sub, config, head), *g, device).argmax(1) == y
        detail[domain] = dict(rows=int(mask.sum()), reference=int(old.sum()), candidate=int(new.sum()))
        net += int(new.sum())-int(old.sum())
    guards['domainnet'] = dict(net=net, by_domain=detail, passed=bool(net >= -5))
    test = load_competition('test', texts)
    graph, _, reference, control_error = reference_check(test, 'test', device)
    probabilities = smooth(candidate_prior(test, config, head), *graph, device)
    predicted = probabilities.argmax(1)
    changed = int((predicted != reference).sum())
    fraction, proxy = changed/len(predicted), frozen['changed_fraction_style_macro']
    pairs = pair_counts(reference, predicted).most_common(5)
    proxy_pairs = [tuple(p) for p, _ in frozen['top_changed_pairs'][:3]]
    guards['test_label_free'] = dict(
        changed=changed, changed_fraction=fraction, proxy_changed_fraction=proxy,
        top_changed_pairs=[[list(k), int(v)] for k, v in pairs], proxy_top3_pairs=[list(p) for p in proxy_pairs],
        class_counts=np.bincount(predicted, minlength=10).tolist(),
        reference_class_counts=np.bincount(reference, minlength=10).tolist(),
        passed=bool(changed <= 2500 and .5*proxy <= fraction <= 2*proxy and pairs and pairs[0][0] in proxy_pairs))
    passed = all(g['passed'] for g in guards.values())
    print(json.dumps(guards, indent=2), flush=True)
    if not passed and not (force_review and guards['source_validation']['passed'] and guards['domainnet']['passed']):
        write(OUT/'export_blocked.json', dict(guards=guards, exported=False))
        print('Guards failed: nothing exported.', flush=True)
        return
    output.mkdir(parents=True)
    sample = pd.read_csv('data/sample_submission.csv')
    assert np.array_equal(test['ids'], sample['id']) and len(np.unique(test['ids'])) == len(test['ids'])
    save_predictions(output, 'test', dict(ids=test['ids']), probabilities)
    save_predictions(output, 'validation', dict(ids=val['ids'], labels=val['y']), val_p)
    csv = output/'submission.csv'
    pd.DataFrame(dict(id=test['ids'], label=predicted)).to_csv(csv, index=False)
    if head is not None:
        np.savez(output/'head.npz', **head)
    metrics = dict(status='ready_not_submitted', method=str(config), guards=guards, all_guards_passed=bool(passed),
                   exported_for_manual_review_despite_test_guard=bool(not passed),
                   reference=str(REFERENCE), reference_public_accuracy=.97565, public_accuracy=None,
                   control_max_probability_error=control_error, test_rows=len(predicted),
                   changed_test_predictions=changed, submission_sha256=digest(csv.read_bytes()),
                   confirmation_sha256=digest(confirmation_bytes), true_test_labels_accessed=False,
                   warehouse_data_used=False, source_sha256=source_hashes())
    write(output/'metrics.json', metrics)
    print('Prepared candidate:', json.dumps({k: metrics[k] for k in ['status', 'method', 'changed_test_predictions',
                                                                        'submission_sha256']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['control', 'instrument', 'select', 'confirm', 'export'])
    parser.add_argument('--manual-review', action='store_true',
                        help='Export for review when only a label-free test guard failed (manual decision).')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    texts = load_texts()
    started = time.time()
    if args.stage == 'control':
        stage_control(device, texts)
    elif args.stage == 'instrument':
        stage_instrument(device, texts)
    elif args.stage == 'select':
        stage_select(device, texts)
    elif args.stage == 'confirm':
        stage_confirm(device, texts)
    else:
        stage_export(device, texts, args.manual_review)
    print(f'Stage {args.stage} finished in {time.time()-started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
