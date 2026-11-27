"""Prepare all official CIFAR-10.2 test images for a new fixed confirmation."""

import json
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image

from prepare_domainnet_cache import digest, write_json


def main():
    root=Path('external_data/cifar10_2_test');root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise FileExistsError('Preserve existing confirmation data.')
    repo='modestyachts/cifar-10.2'
    req=urllib.request.Request('https://api.github.com/repos/'+repo+'/commits/master',
                              headers={'User-Agent':'RecVis-academic-data-check'})
    revision=json.loads(urllib.request.urlopen(req,timeout=60).read())['sha']
    url=f'https://raw.githubusercontent.com/{repo}/{revision}/cifar102_test.npz'
    target=root/'cifar102_test.npz'
    with urllib.request.urlopen(url,timeout=60) as response:target.write_bytes(response.read())
    write_json(root/'download_provenance.json',dict(repository=repo,revision=revision,
        file='cifar102_test.npz',url=url,sha256=digest(target.read_bytes()),bytes=target.stat().st_size))
    with np.load(target,allow_pickle=False) as a:
        images=a['images'];labels=a['labels'].reshape(-1)
    assert images.shape==(2000,32,32,3) and images.dtype==np.uint8
    assert labels.shape==(2000,) and set(np.unique(labels))==set(range(10))
    forbidden=set()
    for split in ['train','validation']:
        with np.load(Path('data')/(split+'.npz')) as a:
            forbidden.update(digest(x.astype(np.uint8).tobytes()) for x in a['images'])
    for source in ['domainnet','dfn_check','cifar10_1_v6']:
        manifest=json.loads((Path('external_data')/source/'manifest.json').read_text())
        forbidden.update(r['pixels32_sha256'] for r in manifest['records'])
        forbidden.update(r['pixels32_sha256'] for r in manifest.get('excluded',[]))
    records,excluded,seen=[],[],set();(root/'images').mkdir(exist_ok=True)
    for i,(pixels,label) in enumerate(zip(images,labels)):
        sha=digest(pixels.tobytes())
        record=dict(id=f'cifar10_2_test/{i:04d}',path=f'images/{i:04d}.png',
            role='confirmation',domain='cifar10_2',label=int(label),pixels32_sha256=sha)
        reason=('provided_or_previous_external_duplicate' if sha in forbidden
                else 'within_check_duplicate' if sha in seen else None)
        if reason:excluded.append(dict(record,reason=reason));continue
        path=root/record['path'];Image.fromarray(pixels).save(path)
        records.append(dict(record,file_sha256=digest(path.read_bytes())));seen.add(sha)
    assert set(r['label'] for r in records)==set(range(10))
    write_json(root/'manifest.json',dict(dataset='CIFAR-10.2 test',repository=repo,revision=revision,
        provided_rows=2000,records=records,excluded=excluded,
        selection='all official test rows; exact pixel exclusion only; no replacements',
        fresh_relative_to_dinov3_experiment=True,pretraining_overlap_unknown=True,
        competition_test_images_or_labels_read=False))
    print('CIFAR-10.2 ready:',len(records),'images;',len(excluded),'excluded.',flush=True)


if __name__=='__main__':main()
