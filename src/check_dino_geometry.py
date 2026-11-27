"""Compare DINO graph geometries while preserving two-model initial probabilities."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from graph_diffusion import neighbours,load_split
from joint_graph_diffusion import fuse_features,mix_probabilities
from external_cache import external_inputs,accuracy,smooth


def geometry(base,dino,weight):
    if weight==0:return base
    if weight==1:return dino
    return fuse_features([base,dino],np.array([1-weight,weight]))


def evaluate(base,dino,initial,weight,device):
    index,edge=neighbours(geometry(base,dino,weight),20,.07,device,512)
    return smooth(initial,index,edge,device)


def save(output,name,p,ids,labels):
    np.savez_compressed(output/(name+'.npz'),probabilities=p,ids=ids,labels=labels)


def changes(base,p,labels):
    old,new=base.argmax(1),p.argmax(1)
    return dict(corrected=int(((old!=labels)&(new==labels)).sum()),
                new_errors=int(((old==labels)&(new!=labels)).sum()),
                changed=int((old!=new).sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dino-run',type=Path,required=True)
    parser.add_argument('--external-root',type=Path,required=True)
    parser.add_argument('--external-runs',type=Path,nargs=2,required=True)
    parser.add_argument('--competition-runs',type=Path,nargs=2,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output directory.')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    weights=[0.,.25,1.]
    inputs=[load_split(r,r,'validation') for r in args.competition_runs]
    assert np.array_equal(inputs[0]['ids'],inputs[1]['ids'])
    assert np.array_equal(inputs[0]['labels'],inputs[1]['labels'])
    initial=mix_probabilities([a['logits'] for a in inputs],[.75,.25])
    base=fuse_features([a['features'] for a in inputs],np.array([.75,.25]))
    with np.load(args.dino_run/'validation_features.npz') as d:
        assert np.array_equal(d['ids'],inputs[0]['ids']) and 'labels' not in d.files
        dino=d['features']
    with np.load(args.reference/'validation_predictions.npz') as ref:
        assert np.array_equal(ref['ids'],inputs[0]['ids'])
        reference=ref['probabilities'];labels=inputs[0]['labels']
        assert np.array_equal(ref['labels'],labels)
    source={}
    for w in weights:
        p=evaluate(base,dino,initial,w,device)
        if w==0:assert np.array_equal(p,reference),'Source control must reproduce exactly.'
        source[str(w)]=dict(accuracy(p,labels),**changes(reference,p,labels))
        save(args.output,'source_'+str(w),p,inputs[0]['ids'],labels)
        print('Source geometry',w,source[str(w)],flush=True)
    raw=(args.external_root/'manifest.json').read_bytes();manifest=json.loads(raw)
    expected_hash=hashlib.sha256(raw).hexdigest()
    assert json.loads((args.dino_run/'metrics.json').read_text())['external_manifest_sha256']==expected_hash
    records=manifest['records'];ids=np.array([r['id'] for r in records])
    features,initial,_,_=external_inputs(args.external_runs,manifest)
    with np.load(args.dino_run/'external_features.npz') as d:
        assert np.array_equal(d['ids'],ids) and 'labels' not in d.files
        dino=d['features']
    calibration={str(w):{} for w in weights};sketch={}
    for domain,role in [('clipart','calibration'),('painting','calibration'),('sketch','fresh_check')]:
        mask=np.array([r['role']==role and r['domain']==domain for r in records])
        y=np.array([r['label'] for r,m in zip(records,mask) if m])
        reference=None
        for w in weights:
            p=evaluate(features[mask],dino[mask],initial[mask],w,device)
            if w==0:reference=p
            measured=dict(accuracy(p,y),**changes(reference,p,y))
            if role=='calibration':calibration[str(w)][domain]=measured
            else:sketch[str(w)]=measured
            save(args.output,domain+'_'+str(w),p,ids[mask],y)
            print(domain,'geometry',w,measured,flush=True)
    macro={str(w):float(np.mean([v['macro_accuracy'] for v in calibration[str(w)].values()]))
           for w in weights}
    eligible=[w for w in weights if source[str(w)]['accuracy']>=source['0.0']['accuracy']-.0005-1e-12]
    chosen=max(eligible,key=lambda w:(macro[str(w)],-w))
    passed=bool(chosen>0 and macro[str(chosen)]>macro['0.0'])
    metrics=dict(selected_dino_geometry_weight=chosen,pilot_gate_passed=passed,
                 weights_compared=weights,eligible_weights=eligible,source=source,
                 calibration=calibration,calibration_macro=macro,reused_sketch_diagnostic=sketch,
                 selection='reused clipart/painting macro with five-example source guard; ties prefer smaller DINO weight',
                 sketch_used_for_selection=False,independent_check_not_scored=True,
                 reference=str(args.reference),dino_run=str(args.dino_run),
                 initial_probability_weights=[.75,.25],graph=dict(k=20,temperature=.07,alpha=.5,chunk=512),
                 best_verified_public_accuracy=.97363,new_public_accuracy=None,
                 source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()
                     for n in ['check_dino_geometry.py','encode_dino_geometry.py','DINO_GEOMETRY_PLAN.md']})
    (args.output/'frozen_selection.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('Selected geometry weight',chosen,'pilot gate',passed,'calibration macro',macro,flush=True)


if __name__=='__main__':main()
