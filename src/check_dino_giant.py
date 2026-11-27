"""Compare fixed DINOv2 giant CLS geometry with the submitted ViT-L reference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from check_dino_geometry import evaluate,changes,save
from external_cache import external_inputs,accuracy
from graph_diffusion import load_split
from joint_graph_diffusion import fuse_features,mix_probabilities


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output directory.')
    args.output.mkdir(parents=True);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    reference=Path('runs/dinov2_geometry_candidate')
    old_comparison=Path('runs/dinov2_geometry_comparison')
    cls_run=Path('runs/dinov2_geometry_pilot')
    results={};control_errors={}
    def compare(name,base,initial,cls,patch,ids,labels,expected):
        control=evaluate(base,cls,initial,.25,device)
        control_errors[name]=float(np.max(np.abs(control-expected)))
        assert np.allclose(control,expected,atol=1e-6,rtol=0)
        assert np.array_equal(control.argmax(1),expected.argmax(1))
        candidate=evaluate(base,patch,initial,.25,device)
        results[name]=dict(baseline=accuracy(control,labels),candidate=accuracy(candidate,labels),
                           changes=changes(control,candidate,labels))
        save(args.output,name+'_baseline',control,ids,labels)
        save(args.output,name+'_candidate',candidate,ids,labels)
        print(name,results[name],flush=True)
    inputs=[load_split(Path('runs')/r,Path('runs')/r,'validation')
            for r in ['pe_bigg_selective_flip_v2','siglip2_selective_flip_v2']]
    ids=inputs[0]['ids'];labels=inputs[0]['labels']
    assert np.array_equal(ids,inputs[1]['ids']) and np.array_equal(labels,inputs[1]['labels'])
    base=fuse_features([a['features'] for a in inputs],np.array([.75,.25]))
    initial=mix_probabilities([a['logits'] for a in inputs],[.75,.25])
    with np.load(args.candidate_run/'validation_features.npz') as p, np.load(cls_run/'validation_features.npz') as c, np.load(reference/'validation_predictions.npz') as r:
        for z in [p,c,r]:assert np.array_equal(z['ids'],ids)
        assert np.array_equal(r['labels'],labels)
        compare('source',base,initial,c['features'],p['features'],ids,labels,r['probabilities'])
    root=Path('external_data/dfn_check');raw=(root/'manifest.json').read_bytes();manifest=json.loads(raw)
    candidate_metrics=json.loads((args.candidate_run/'metrics.json').read_text())
    assert candidate_metrics['input_size']==[3,224,224]
    assert candidate_metrics['model']=='timm/vit_giant_patch14_reg4_dinov2.lvd142m'
    assert candidate_metrics['external_manifest_sha256']==hashlib.sha256(raw).hexdigest()
    ids=np.array([r['id'] for r in manifest['records']]);labels=np.array([r['label'] for r in manifest['records']])
    base,initial,_,_=external_inputs([Path('runs/dfn_complementarity/encoder_0'),Path('runs/dfn_complementarity/encoder_1')],manifest)
    with np.load(args.candidate_run/'external_features.npz') as p,np.load(cls_run/'external_features.npz') as c:
        assert np.array_equal(ids,p['ids']) and np.array_equal(ids,c['ids'])
        for domain in ['clipart','painting','sketch']:
            mask=np.array([r['domain']==domain for r in manifest['records']])
            with np.load(old_comparison/(domain+'_0.25.npz')) as expected:
                assert np.array_equal(expected['ids'],ids[mask]) and np.array_equal(expected['labels'],labels[mask])
                compare(domain,base[mask],initial[mask],c['features'][mask],p['features'][mask],ids[mask],labels[mask],expected['probabilities'])
    macro={method:float(np.mean([results[d][method]['macro_accuracy'] for d in ['clipart','painting']]))
           for method in ['baseline','candidate']}
    source_ok=results['source']['candidate']['accuracy']>=results['source']['baseline']['accuracy']-.0005-1e-12
    calibration_ok=macro['candidate']>=macro['baseline']
    sketch_ok=results['sketch']['candidate']['macro_accuracy']>=results['sketch']['baseline']['macro_accuracy']
    strict_external=(macro['candidate']>macro['baseline'] or results['sketch']['candidate']['macro_accuracy']>results['sketch']['baseline']['macro_accuracy'])
    passed=bool(source_ok and calibration_ok and sketch_ok and strict_external)
    metrics=dict(pilot_passed=passed,source_guard_passed=source_ok,calibration_noninferior=calibration_ok,strict_external_improvement=strict_external,
        sketch_noninferior=sketch_ok,results=results,calibration_macro=macro,
        cls_control_max_probability_errors=control_errors,reference_public_accuracy=.97520,
        new_public_accuracy=None,all_evaluation_sets_reused=True,one_candidate_no_parameter_search=True,
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()
                       for n in ['encode_dino_giant.py','check_dino_giant.py','DINO_GIANT_PLAN.md']})
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (args.output/'frozen_decision.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('DINOv2 giant pilot passed:',passed,'calibration macro:',macro,flush=True)


if __name__=='__main__':main()
