"""Confirm the frozen ViT-g candidate against ViT-L on CIFAR-10.2."""

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
    parser.add_argument('--baseline-dino-run',type=Path,required=True)
    parser.add_argument('--dino-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous confirmation.')
    frozen=(args.selection/'frozen_decision.json').read_bytes();selection=json.loads(frozen)
    assert selection['pilot_passed']
    args.output.mkdir(parents=True);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    raw=(args.root/'manifest.json').read_bytes();manifest=json.loads(raw)
    ids=np.array([r['id'] for r in manifest['records']])
    base,initial,_,view_mask=external_inputs(args.external_runs,manifest)
    features=[]
    for run,model_name in [(args.baseline_dino_run,'timm/vit_large_patch14_reg4_dinov2.lvd142m'),
                           (args.dino_run,'timm/vit_giant_patch14_reg4_dinov2.lvd142m')]:
        metadata=json.loads((run/'metrics.json').read_text())
        assert metadata['model']==model_name
        assert metadata['external_manifest_sha256']==hashlib.sha256(raw).hexdigest()
        with np.load(run/'external_features.npz') as a:
            assert 'labels' not in a.files and np.array_equal(a['ids'],ids)
            assert np.array_equal(a['view_mask'],view_mask)
            features.append(a['features'])
    baseline=evaluate(base,features[0],initial,.25,device)
    candidate=evaluate(base,features[1],initial,.25,device)
    # Confirmation labels enter only after both pipelines have produced scores.
    labels=np.array([r['label'] for r in manifest['records']])
    results=dict(baseline=accuracy(baseline,labels),candidate=accuracy(candidate,labels),
                 changes=changes(baseline,candidate,labels))
    for name,p in [('baseline',baseline),('candidate',candidate)]:save(args.output,name,p,ids,labels)
    passed=bool((candidate.argmax(1)==labels).sum()>(baseline.argmax(1)==labels).sum())
    metrics=dict(weight=.25,confirmation_passed=passed,results=results,
        confirmation_labels_used_for_selection=False,
        selection_sha256=hashlib.sha256(frozen).hexdigest(),manifest_sha256=hashlib.sha256(raw).hexdigest(),
        reference_public_accuracy=.97520,new_public_accuracy=None,
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()
                      for n in ['confirm_dino_giant.py','check_dino_giant.py','encode_dino_giant.py','DINO_GIANT_PLAN.md']})
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('CIFAR-10.2 paired confirmation:',results,'passed:',passed,flush=True)


if __name__=='__main__':main()
