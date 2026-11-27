"""Run fixed CIFAR-10.1 confirmation, then prepare at most one DINO candidate."""

import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    root=Path(__file__).resolve().parent;os.chdir(root)
    modern=os.path.expanduser('~/recvis/.venv-modern/bin/python')
    sig_python=os.path.expanduser('~/recvis/.venv-siglip2/bin/python')
    selected=json.loads(Path('runs/dinov2_geometry_comparison/frozen_selection.json').read_text())
    assert selected['pilot_gate_passed']
    def environment(gpu):
        return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',
                    TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4')
    jobs=[]
    for python,gpu,base,output in [(modern,1,'pe_bigg_imagenet80_fp16','cifar10_1_bigg'),
                                  (sig_python,0,'siglip2_gopt_imagenet80_fp16','cifar10_1_siglip2')]:
        cmd=[python,'-u','encode_domainnet.py','--root','external_data/cifar10_1_v6',
             '--base','runs/'+base,'--output','runs/'+output,'--batch-size','16']
        handle=open('logs/'+output+'.log','w')
        p=subprocess.Popen(cmd,env=environment(gpu),stdout=handle,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL)
        jobs.append((p,handle,output))
        Path('logs/'+output+'_process.json').write_text(json.dumps(dict(pid=p.pid,command=cmd,gpu=gpu,
            source_sha256=hashlib.sha256(Path('encode_domainnet.py').read_bytes()).hexdigest()),indent=2)+'\n')
        print('Started',output,'PID',p.pid,'GPU',gpu,flush=True)
    codes=[]
    for p,handle,output in jobs:
        code=p.wait();handle.close();codes.append(code);print(output,'exit',code,flush=True)
    if any(codes):raise RuntimeError('A confirmation encoder failed.')
    cmd=[modern,'-u','encode_dino_geometry.py','--external-root','external_data/cifar10_1_v6',
         '--external-runs','runs/cifar10_1_bigg','runs/cifar10_1_siglip2',
         '--output','runs/cifar10_1_dinov2','--batch-size','64']
    subprocess.run(cmd,env=environment(0),check=True)
    cmd=[modern,'-u','confirm_dino_geometry.py','--selection','runs/dinov2_geometry_comparison',
         '--root','external_data/cifar10_1_v6','--external-runs','runs/cifar10_1_bigg','runs/cifar10_1_siglip2',
         '--dino-run','runs/cifar10_1_dinov2','--output','runs/dinov2_geometry_confirmation']
    subprocess.run(cmd,env=environment(0),check=True)
    confirmation=json.loads(Path('runs/dinov2_geometry_confirmation/metrics.json').read_text())
    if not confirmation['confirmation_passed']:
        print('Independent check did not improve; no test candidate prepared.',flush=True);return
    cmd=[modern,'-u','encode_dino_geometry.py','--data','data','--splits','test',
         '--selection','runs/next3_selection_20261112.npz','--output','runs/dinov2_geometry_test',
         '--batch-size','64']
    subprocess.run(cmd,env=environment(0),check=True)
    cmd=[modern,'-u','export_dino_geometry.py','--selection','runs/dinov2_geometry_comparison',
         '--confirmation','runs/dinov2_geometry_confirmation','--dino-test','runs/dinov2_geometry_test',
         '--runs','runs/pe_bigg_selective_flip_v2','runs/siglip2_selective_flip_v2',
         '--reference','runs/joint_graph_selective_flip','--data','data',
         '--output','runs/dinov2_geometry_candidate']
    subprocess.run(cmd,env=environment(0),check=True)
    print('Confirmed candidate prepared. No automatic Kaggle upload.',flush=True)


if __name__=='__main__':main()
