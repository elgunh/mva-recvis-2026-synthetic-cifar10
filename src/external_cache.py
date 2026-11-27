"""Calibrate a class-balanced external feature cache, then check an untouched style."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graph_diffusion import neighbours, diffuse, load_split, save_predictions
from joint_graph_diffusion import fuse_features, mix_probabilities


def unit(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or np.any(norms < 1e-8):
        raise ValueError('Nonfinite or zero features.')
    return features/norms


@torch.inference_mode()
def cache_distribution(features, keys, labels, domains, device, beta=20, chunk=256):
    """Mean kernel per class/domain; never receives query labels."""
    x, keys = unit(features), unit(keys)
    labels, domains = np.asarray(labels), np.asarray(domains)
    classes, groups = np.unique(labels), np.unique(domains)
    values = np.zeros((len(keys), 10), dtype=np.float32)
    for group in groups:
        for category in classes:
            mask = (labels == category) & (domains == group)
            if not mask.any():
                raise ValueError('Every support domain must cover every mapped class.')
            values[mask, category] = 1/(len(groups)*mask.sum())
    keys = torch.as_tensor(keys, device=device)
    values = torch.as_tensor(values, device=device)
    result = []
    for offset in range(0, len(x), chunk):
        similarity = torch.as_tensor(x[offset:offset+chunk], device=device)@keys.T
        scores = torch.exp(beta*(similarity-1))@values
        scores /= scores.sum(1, keepdim=True).clamp_min(1e-30)
        result.append(scores.cpu().numpy())
    result = np.concatenate(result)
    if not np.isfinite(result).all() or not np.allclose(result.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid cache distribution.')
    return result


def blend(initial, cache, classes, strength, uncertain_only=False):
    """Redistribute only covered-class mass; preserve absent-class probabilities."""
    if not 0 <= strength <= 1:
        raise ValueError('Invalid cache strength.')
    output = initial.copy()
    mass = initial[:, classes].sum(1, keepdims=True)
    output[:, classes] = ((1-strength)*initial[:, classes]
                          + strength*mass*cache[:, classes])
    if uncertain_only:
        confident = initial.max(1) >= .9
        output[confident] = initial[confident]
    if not np.allclose(output.sum(1), 1, atol=1e-6) or np.any(output < 0):
        raise ValueError('Invalid cache mixture.')
    return output


def accuracy(probabilities, labels):
    predicted = probabilities.argmax(1)
    classes = np.unique(labels)
    per_class = {str(c): float(np.mean(predicted[labels == c] == c)) for c in classes}
    return dict(accuracy=float(np.mean(predicted == labels)),
                macro_accuracy=float(np.mean(list(per_class.values()))),
                per_class=per_class, rows=len(labels))


def external_inputs(runs, manifest):
    records = manifest['records']
    expected_ids = np.array([r['id'] for r in records])
    arrays = []
    for run in runs:
        with np.load(run/'external_features.npz') as data:
            assert np.array_equal(expected_ids, data['ids'])
            arrays.append({k: data[k] for k in data.files if k != 'ids'})
    initial = mix_probabilities([a['logits'] for a in arrays], [.75, .25])
    selected = ((arrays[0]['logits'].argmax(1) != arrays[1]['logits'].argmax(1))
                | (initial.max(1) < .9))
    query_features, query_logits, support_features = [], [], []
    for data in arrays:
        averaged = unit(data['features']+data['flip_features'])
        features = data['features'].copy()
        features[selected] = averaged[selected]
        probabilities = mix_probabilities([data['logits']], [1.])
        probabilities[selected] = mix_probabilities(
            [data['logits'][selected], data['flip_logits'][selected]], [.5, .5])
        query_features.append(features)
        query_logits.append(np.log(np.maximum(probabilities, 1e-30)))
        support_features.append(averaged)
    return (fuse_features(query_features, [.75,.25]),
            mix_probabilities(query_logits, [.75,.25]),
            fuse_features(support_features, [.75,.25]), selected)


def smooth(initial, index, weights, device):
    return diffuse(np.log(np.maximum(initial,1e-30)), index, weights, .5, device)[0]


def checks():
    rng = np.random.default_rng(123)
    x, keys = unit(rng.normal(size=(7,5))), unit(rng.normal(size=(12,5)))
    labels = np.array([0,0,1,1,2,2]*2)
    domains = np.array(['a']*6+['b']*6)
    actual = cache_distribution(x,keys,labels,domains,'cpu')
    dense = np.zeros((7,10))
    kernel = np.exp(20*(x@keys.T-1))
    for i in range(7):
        for c in range(3):
            dense[i,c] = np.mean([kernel[i,(labels==c)&(domains==d)].mean()
                                   for d in ['a','b']])
    dense /= dense.sum(1,keepdims=True)
    assert np.allclose(actual,dense,atol=2e-6)
    initial = rng.dirichlet(np.ones(10),size=7)
    assert np.array_equal(blend(initial,actual,np.arange(3),0),initial)
    updated = blend(initial,actual,np.arange(3),.25)
    assert np.array_equal(updated[:,3:],initial[:,3:])
    assert np.allclose(updated[:,:3].sum(1),initial[:,:3].sum(1))
    duplicated = cache_distribution(x,np.tile(keys,(2,1)),np.tile(labels,2),
                                    np.tile(domains,2),'cpu')
    assert np.allclose(actual,duplicated,atol=2e-6)
    strong = initial.copy(); strong[0] = np.eye(10)[0]
    gated = blend(strong,actual,np.arange(3),.25,uncertain_only=True)
    assert np.array_equal(gated[0],strong[0])
    return dict(dense_reference_max_error=float(np.max(np.abs(actual-dense))),
                zero_strength_exact=True, absent_classes_exactly_preserved=True,
                covered_probability_mass_preserved=True,
                uniform_support_duplication_invariant=True,
                confidence_gate_preserves_confident_rows=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--external-runs', nargs=2, type=Path)
    parser.add_argument('--competition-runs', nargs=2, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--uncertain-only', action='store_true',
                        help='Apply cache only where initial mixed confidence is below .9.')
    args = parser.parse_args()
    numerical = checks()
    if args.self_test:
        print(json.dumps(numerical,indent=2)); return
    if not all([args.root,args.external_runs,args.competition_runs,args.reference,args.data,args.output]):
        parser.error('All input paths are required.')
    if args.output.exists():
        raise FileExistsError('Use a new output directory.')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest_bytes = (args.root/'manifest.json').read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    for run, competition_run in zip(args.external_runs,args.competition_runs):
        external_metadata = json.loads((run/'metrics.json').read_text())
        competition_metadata = json.loads((competition_run/'metrics.json').read_text())
        assert external_metadata['manifest_sha256'] == manifest_hash
        assert external_metadata['model'] == competition_metadata['model'], 'Encoder order differs.'
        assert external_metadata['prompts'] == competition_metadata['prompts'], 'Text templates differ.'
    records = manifest['records']
    features, initial, support_features, selected = external_inputs(args.external_runs,manifest)
    roles = np.array([r['role'] for r in records])
    domains = np.array([r['domain'] for r in records])
    support = roles == 'support'
    # Only support labels enter the cache kernel. Query labels are used for scoring.
    support_labels = np.array([r['label'] for r in records if r['role']=='support'])
    classes = np.unique(support_labels)
    keys = support_features[support]
    cache = cache_distribution(features,keys,support_labels,domains[support],device)
    strengths = [0.,.10,.25]
    calibration = {str(w): {} for w in strengths}
    for domain in ['clipart','painting']:
        mask = (roles=='calibration') & (domains==domain)
        index, edge_weights = neighbours(features[mask],20,.07,device,512)
        labels = np.array([r['label'] for r,m in zip(records,mask) if m])
        for strength in strengths:
            p = smooth(blend(initial[mask],cache[mask],classes,strength,args.uncertain_only),index,edge_weights,device)
            calibration[str(strength)][domain] = accuracy(p,labels)
            np.savez_compressed(args.output/f'calibration_{domain}_{strength}.npz',
                                probabilities=p,labels=labels,
                                ids=np.array([r['id'] for r,m in zip(records,mask) if m]))
    comp = [load_split(r,r,'validation') for r in args.competition_runs]
    assert np.array_equal(comp[0]['ids'],comp[1]['ids'])
    assert np.array_equal(comp[0]['labels'],comp[1]['labels'])
    source_features = fuse_features([a['features'] for a in comp],[.75,.25])
    source_initial = mix_probabilities([a['logits'] for a in comp],[.75,.25])
    source_cache = cache_distribution(source_features,keys,support_labels,domains[support],device)
    index, edge_weights = neighbours(source_features,20,.07,device,512)
    with np.load(args.reference/'validation_predictions.npz') as ref:
        assert np.array_equal(comp[0]['ids'],ref['ids'])
        assert np.array_equal(comp[0]['labels'],ref['labels'])
        ref_p, ref_predictions = ref['probabilities'], ref['predictions']
    source, source_probabilities = {}, {}
    baseline_source = float(np.mean(ref_predictions==comp[0]['labels']))
    for strength in strengths:
        p = smooth(blend(source_initial,source_cache,classes,strength,args.uncertain_only),index,edge_weights,device)
        if strength == 0:
            assert np.allclose(p,ref_p,atol=2e-6), 'Zero cache must reproduce best run.'
            assert np.array_equal(p.argmax(1),ref_predictions)
        source[str(strength)] = accuracy(p,comp[0]['labels'])
        source_probabilities[str(strength)] = p
        np.savez_compressed(args.output/f'source_validation_{strength}.npz',
                            ids=comp[0]['ids'],labels=comp[0]['labels'],probabilities=p)
    macro = {str(w): float(np.mean([v['macro_accuracy']
                    for v in calibration[str(w)].values()])) for w in strengths}
    eligible = [w for w in strengths if source[str(w)]['accuracy'] >= baseline_source-.0005-1e-12]
    chosen = max(eligible,key=lambda w:(macro[str(w)],-w))
    # Save and freeze the choice before reading holdout labels or scoring sketch.
    decision = dict(strength=chosen, calibration_macro=macro, source_validation=source,
                    calibration=calibration, numerical_checks=numerical,
                    eligible_strengths=eligible,
                    manifest_sha256=manifest_hash, beta=20, feature_weights=[.75,.25],
                    cache_gate='initial mixed maximum probability below .9' if args.uncertain_only else 'all rows',
                    graph=dict(k=20,temperature=.07,alpha=.5),
                    support_rows=int(support.sum()), selected_external_views=int(selected.sum()),
                    heldout_labels_used_for_selection=False,
                    competition_test_labels_accessed=False,
                    external_runs=[str(r) for r in args.external_runs],
                    competition_runs=[str(r) for r in args.competition_runs],
                    reference_run=str(args.reference),
                    source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                   for name in ['external_cache.py','encode_domainnet.py',
                                                'prepare_domainnet_cache.py','EXTERNAL_CACHE_PLAN.md']},
                    original_method='class/domain-balanced, missing-class-preserving cache inspired by Tip-Adapter')
    (args.output/'frozen_selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    print('Frozen choice:', chosen, 'calibration macro:', macro, flush=True)
    holdout = roles=='holdout'
    index, edge_weights = neighbours(features[holdout],20,.07,device,512)
    holdout_probabilities = {str(w): smooth(blend(initial[holdout],cache[holdout],classes,w,args.uncertain_only),
                             index,edge_weights,device) for w in sorted(set([0.,chosen]))}
    holdout_labels = np.array([r['label'] for r,m in zip(records,holdout) if m])
    heldout_results = {}
    for key,p in holdout_probabilities.items():
        heldout_results[key] = accuracy(p,holdout_labels)
        np.savez_compressed(args.output/f'holdout_{key}.npz', probabilities=p,labels=holdout_labels,
                            ids=np.array([r['id'] for r,m in zip(records,holdout) if m]))
    passed = chosen > 0 and macro[str(chosen)] > macro['0.0']
    passed = bool(passed and heldout_results[str(chosen)]['macro_accuracy'] >
                  heldout_results['0.0']['macro_accuracy'])
    metrics = dict(decision,holdout=heldout_results, export_gate_passed=passed,
                   validation_accuracy=source[str(chosen)]['accuracy'],
                   best_verified_public_accuracy=0.97363, new_public_accuracy=None)
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('Held-out results:',heldout_results,'export gate:',passed,flush=True)
    if not passed:
        print('No competition candidate exported; keep current best.',flush=True); return
    save_predictions(args.output,'validation',comp[0],source_probabilities[str(chosen)])
    del comp, source_features, source_cache, index, edge_weights
    test = [load_split(r,r,'test') for r in args.competition_runs]
    sample = pd.read_csv(args.data/'sample_submission.csv')
    assert np.array_equal(test[0]['ids'],test[1]['ids'])
    assert np.array_equal(test[0]['ids'],sample['id']) and len(np.unique(sample['id']))==len(sample)
    test_features = fuse_features([a['features'] for a in test],[.75,.25])
    test_initial = mix_probabilities([a['logits'] for a in test],[.75,.25])
    test_cache = cache_distribution(test_features,keys,support_labels,domains[support],device)
    index, edge_weights = neighbours(test_features,20,.07,device,512)
    probabilities = smooth(blend(test_initial,test_cache,classes,chosen,args.uncertain_only),index,edge_weights,device)
    predictions = save_predictions(args.output,'test',test[0],probabilities)
    csv = args.output/'submission.csv'
    pd.DataFrame(dict(id=test[0]['ids'],label=predictions)).to_csv(csv,index=False)
    with np.load(args.reference/'test_predictions.npz') as ref:
        assert 'labels' not in ref.files and np.array_equal(ref['ids'],sample['id'])
        metrics['changed_test_predictions'] = int(np.sum(predictions!=ref['predictions']))
    metrics.update(submission_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
                   test_rows=len(predictions))
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('Exported',csv,flush=True)


if __name__ == '__main__':
    main()
