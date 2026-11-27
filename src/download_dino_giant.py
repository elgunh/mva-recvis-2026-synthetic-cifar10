"""Download and verify the pinned public DINOv2 giant checkpoint on the server."""

import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download

REPO='timm/vit_giant_patch14_reg4_dinov2.lvd142m'
REVISION='6994d74074a8f74e220de3f403dc49dd5d8ae704'
EXPECTED_SHA='6c038d1d0b91f20d551b052b85e66bf792e53a086a10ab2b054a8b1f185ed303'


def main():
    started=time.time();print('Downloading pinned public DINOv2 giant',REPO,REVISION,flush=True)
    config=Path(hf_hub_download(REPO,'config.json',revision=REVISION,token=False))
    weights=Path(hf_hub_download(REPO,'model.safetensors',revision=REVISION,token=False))
    digest=hashlib.sha256()
    with weights.open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):digest.update(block)
    assert digest.hexdigest()==EXPECTED_SHA and weights.stat().st_size==4545989736
    report=dict(repo=REPO,revision=REVISION,weights_path=str(weights),
        weights_bytes=weights.stat().st_size,weights_sha256=digest.hexdigest(),
        official_lfs_sha256_verified=True,config_path=str(config),download_seconds=time.time()-started)
    root=Path('models/dinov2_vitg14_reg4');root.mkdir(parents=True,exist_ok=True)
    (root/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified giant checkpoint:',json.dumps(report),flush=True)


if __name__=='__main__':main()
