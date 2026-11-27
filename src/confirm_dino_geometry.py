"""Confirm a frozen DINO geometry choice on CIFAR-10.1 without selecting parameters."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from check_dino_geometry import evaluate,changes,save
from external_cache import external_inputs,accuracy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--external-runs',type=Path,nargs=2,required=True)
    parser.add_argument('--dino-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output directory.')
    args.output.mkdir(parents=True);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    frozen_bytes=(args.selection/'frozen_selection.json').read_bytes();selection=json.loads(frozen_bytes)
    weight=selection['selected_dino_geometry_weight']
    assert selection['pilot_gate_passed'] and weight>0
    raw=(args.root/'manifest.json').read_bytes();manifest=json.loads(raw)
    ids=np.array([r['id'] for r in manifest['records']])
    features,initial,_,_=external_inputs(args.external_runs,manifest)
    assert json.loads((args.dino_run/'metrics.json').read_text())['external_manifest_sha256']==hashlib.sha256(raw).hexdigest()
    with np.load(args.dino_run/'external_features.npz') as a:
        assert 'labels' not in a.files and np.array_equal(a['ids'],ids)
        dino=a['features']
    base=evaluate(features,dino,initial,0,device)
    candidate=evaluate(features,dino,initial,weight,device)
    # Only now use confirmation labels for scoring; no parameter changes follow.
    labels=np.array([r['label'] for r in manifest['records']])
    results=dict(baseline=accuracy(base,labels),candidate=accuracy(candidate,labels),
                 changes=changes(base,candidate,labels))
    for name,p in [('baseline',base),('candidate',candidate)]:save(args.output,name,p,ids,labels)
    passed=results['candidate']['accuracy']>results['baseline']['accuracy']
    metrics=dict(weight=weight,confirmation_passed=passed,results=results,
                 confirmation_labels_used_for_selection=False,
                 selection_sha256=hashlib.sha256(frozen_bytes).hexdigest(),
                 manifest_sha256=hashlib.sha256(raw).hexdigest(),
                 source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()
                    for n in ['confirm_dino_geometry.py','check_dino_geometry.py','encode_dino_geometry.py','DINO_GEOMETRY_PLAN.md']},
                 reference_public_accuracy=.97363,new_public_accuracy=None)
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('Confirmation:',results,'passed:',passed,flush=True)


if __name__=='__main__':main()
