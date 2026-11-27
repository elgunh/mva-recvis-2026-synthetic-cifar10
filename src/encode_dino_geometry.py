"""Cache frozen DINO features using the existing label-free reflected-view policy."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader,Subset

from data import NpzImageDataset
from encode_domainnet import ExternalImages
from external_cache import external_inputs

REPO='timm/vit_large_patch14_reg4_dinov2.lvd142m'
REVISION='f3c408e77602bb412aa65fb03dfa0d5f95cb3832'


def normalized(features):
    features=np.asarray(features,dtype=np.float32)
    norms=np.linalg.norm(features,axis=1,keepdims=True)
    if not np.isfinite(features).all() or np.any(norms<1e-8):
        raise ValueError('Invalid features.')
    return features/norms


@torch.inference_mode()
def encode(model,dataset,mask,device,batch_size,label):
    def run(subset,flip):
        loader=DataLoader(subset,batch_size=batch_size,shuffle=False,num_workers=4,
                          pin_memory=device.type=='cuda')
        encoded=[];start=time.time()
        for i,batch in enumerate(loader,1):
            images=batch[0] if isinstance(batch,(list,tuple)) else batch
            images=images.to(device)
            if flip:images=images.flip(-1)
            with torch.autocast(device.type,enabled=device.type=='cuda'):
                result=model(images)
            result=torch.nn.functional.normalize(result.float(),dim=1)
            encoded.append(result.cpu().numpy())
            if i==1 or i%50==0 or i==len(loader):
                print(label,'flip' if flip else 'original',min(i*batch_size,len(subset)),
                      '/',len(subset),f'{time.time()-start:.1f}s',flush=True)
        return np.concatenate(encoded)
    original=run(dataset,False);combined=original.copy()
    selected=np.flatnonzero(mask)
    if len(selected):
        reflected=run(Subset(dataset,selected.tolist()),True)
        combined[selected]=normalized(original[selected]+reflected)
    assert np.array_equal(combined[~mask],original[~mask])
    assert np.allclose(np.linalg.norm(combined,axis=1),1,atol=1e-5)
    return dict(features=combined,original_features=original,view_mask=mask)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--splits',nargs='*',default=[])
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--external-root',type=Path)
    parser.add_argument('--external-runs',nargs=2,type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=64)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous encoder outputs.')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading frozen DINOv2',REPO,REVISION,flush=True)
    model=timm.create_model('hf-hub:'+REPO+'@'+REVISION,pretrained=True,num_classes=0,img_size=224)
    model=model.to(device).eval()
    if device.type=='cuda':model=model.half()
    for parameter in model.parameters():parameter.requires_grad_(False)
    cfg=timm.data.resolve_model_data_config(model)
    default_size=cfg['input_size'];cfg.update(input_size=(3,224,224),crop_pct=1.0)
    transform=timm.data.create_transform(**cfg,is_training=False)
    provenance=json.loads(Path('models/dinov2_vitl14_reg4/provenance.json').read_text())
    metrics=dict(model=REPO,revision=REVISION,weights_sha256=provenance['weights_sha256'],
                 embedding='normalized class token, no classification head',
                 checkpoint_default_input_size=default_size,input_size=[3,224,224],
                 evaluation_resolution_overridden=True,preprocess=str(transform),
                 weight_dtype=str(next(model.parameters()).dtype),frozen=True,splits={})
    first=True
    def process(dataset,mask,ids,label):
        nonlocal first
        assert mask.dtype==np.bool_ and mask.shape==(len(dataset),)
        if first:
            images=torch.stack([dataset[i][0] if isinstance(dataset[i],tuple) else dataset[i]
                                for i in range(min(args.batch_size,len(dataset)))]).to(device)
            times=[]
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
            for i in range(3):
                if device.type=='cuda':torch.cuda.synchronize()
                start=time.perf_counter()
                with torch.autocast(device.type,enabled=device.type=='cuda'):result=model(images)
                assert torch.isfinite(result).all()
                if device.type=='cuda':torch.cuda.synchronize()
                if i:times.append(time.perf_counter()-start)
            metrics['preflight']=dict(batch_size=len(images),images_per_second=len(images)/np.mean(times),
                peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None)
            print('Preflight:',metrics['preflight'],flush=True);first=False;del images,result
        start=time.time();encoded=encode(model,dataset,mask,device,args.batch_size,label)
        encoded['ids']=ids
        np.savez(args.output/f'{label}_features.npz',**encoded)
        metrics['splits'][label]=dict(rows=len(ids),reflected_rows=int(mask.sum()),
              dimensions=encoded['features'].shape[1],seconds=time.time()-start,
              unselected_features_exactly_preserved=True,labels_saved=False)
        (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    if args.splits:
        if not args.selection:raise ValueError('Selection archive required.')
        with np.load(args.selection) as selected:
            for split in args.splits:
                dataset=NpzImageDataset(args.data/f'{split}.npz',transform,require_labels=False)
                if split=='test':assert dataset.labels is None
                assert np.array_equal(dataset.ids,selected[f'{split}_ids'])
                process(dataset,selected[f'{split}_mask'],dataset.ids,split)
    if args.external_root:
        manifest_bytes=(args.external_root/'manifest.json').read_bytes();manifest=json.loads(manifest_bytes)
        _,_,_,mask=external_inputs(args.external_runs,manifest)
        dataset=ExternalImages(args.external_root,manifest['records'],transform)
        ids=np.array([r['id'] for r in manifest['records']])
        metrics['external_manifest_sha256']=hashlib.sha256(manifest_bytes).hexdigest()
        process(dataset,mask,ids,'external')
    print('DINO encoding completed.',flush=True)


if __name__=='__main__':main()
