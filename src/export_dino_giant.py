"""Export the frozen, confirmed DINOv2 giant candidate on unlabeled test."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from check_dino_geometry import evaluate
from graph_diffusion import load_split,save_predictions
from joint_graph_diffusion import fuse_features,mix_probabilities


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--confirmation',type=Path,required=True)
    parser.add_argument('--dino-test',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs=2,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    selection_bytes=(args.selection/'frozen_decision.json').read_bytes();selection=json.loads(selection_bytes)
    confirmation=json.loads((args.confirmation/'metrics.json').read_text())
    assert confirmation['confirmation_passed'] and confirmation['selection_sha256']==hashlib.sha256(selection_bytes).hexdigest()
    assert selection['pilot_passed']
    weight=.25;assert weight==confirmation['weight']
    if args.output.exists():raise FileExistsError('Preserve existing outputs.')
    args.output.mkdir(parents=True);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    inputs=[load_split(r,r,'test') for r in args.runs]
    ids=inputs[0]['ids'];sample=pd.read_csv(args.data/'sample_submission.csv')
    assert np.array_equal(ids,inputs[1]['ids']) and np.array_equal(ids,sample['id'])
    assert len(np.unique(ids))==len(ids)
    with np.load(args.dino_test/'test_features.npz') as d:
        assert 'labels' not in d.files and np.array_equal(ids,d['ids']);dino=d['features']
    base=fuse_features([a['features'] for a in inputs],np.array([.75,.25]))
    initial=mix_probabilities([a['logits'] for a in inputs],[.75,.25])
    probabilities=evaluate(base,dino,initial,weight,device)
    predicted=save_predictions(args.output,'test',dict(ids=ids),probabilities)
    csv=args.output/'submission.csv';pd.DataFrame(dict(id=ids,label=predicted)).to_csv(csv,index=False)
    with np.load(args.reference/'test_predictions.npz') as r:
        assert 'labels' not in r.files and np.array_equal(ids,r['ids']);changed=int((predicted!=r['predictions']).sum())
    with np.load(args.selection/'source_candidate.npz') as v:
        save_predictions(args.output,'validation',dict(ids=v['ids'],labels=v['labels']),v['probabilities'])
    metrics=dict(status='ready_not_submitted',dino_geometry_weight=weight,
        selection=str(args.selection),confirmation=str(args.confirmation),reference=str(args.reference),
        source_validation_accuracy=selection['results']['source']['candidate']['accuracy'],
        confirmation_results=confirmation['results'],reference_public_accuracy=.97520,
        public_accuracy=None,test_rows=len(ids),changed_test_predictions=changed,
        submission_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
        initial_probabilities_unchanged=True,true_test_labels_accessed=False)
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('Prepared DINO candidate:',metrics,flush=True)


if __name__=='__main__':main()
