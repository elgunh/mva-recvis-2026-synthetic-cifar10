"""Confirm a frozen giant pilot and export at most one candidate after a positive check."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    os.chdir(Path(__file__).resolve().parent)
    python=os.path.expanduser('~/recvis/.venv-modern/bin/python')
    sig_python=os.path.expanduser('~/recvis/.venv-siglip2/bin/python')
    selection=Path('runs/dinov2_giant_comparison/frozen_decision.json').read_bytes()
    assert json.loads(selection)['pilot_passed']
    print('Frozen selection SHA256:',hashlib.sha256(selection).hexdigest(),flush=True)
    def environment(gpu):
        return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',
                    TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4')
    def start(command,gpu,log):
        handle=open(log,'x')
        process=subprocess.Popen(command,env=environment(gpu),stdout=handle,
            stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        print('Started',log,'PID',process.pid,'GPU',gpu,flush=True)
        return process,handle
    def finish(job):
        process,handle=job;code=process.wait();handle.close()
        if code:raise RuntimeError(f'Encoder process {process.pid} exited {code}')
    sig=start([sig_python,'-u','encode_domainnet.py','--root','external_data/cifar10_2_test',
        '--base','runs/siglip2_gopt_imagenet80_fp16','--output','runs/cifar10_2_siglip2',
        '--batch-size','16'],0,'logs/cifar10_2_siglip2.log')
    finish(sig)
    started=time.monotonic();last_message=-1
    while not Path('runs/cifar10_2_bigg/metrics.json').exists():
        elapsed=time.monotonic()-started
        if elapsed>1800:raise TimeoutError('The precomputed bigG cache did not complete.')
        interval=int(elapsed//30)
        if interval!=last_message:
            print('Waiting for unscored bigG cache...',int(elapsed),'seconds',flush=True);last_message=interval
        time.sleep(5)
    common=['--external-root','external_data/cifar10_2_test','--external-runs',
            'runs/cifar10_2_bigg','runs/cifar10_2_siglip2']
    old=start([python,'-u','encode_dino_geometry.py',*common,'--output','runs/cifar10_2_dinov2_l',
        '--batch-size','64'],0,'logs/cifar10_2_dinov2_l.log')
    giant=start([python,'-u','encode_dino_giant.py',*common,'--output','runs/cifar10_2_dinov2_g',
        '--batch-size','32'],1,'logs/cifar10_2_dinov2_g.log')
    finish(old);finish(giant)
    subprocess.run([python,'-u','confirm_dino_giant.py','--selection','runs/dinov2_giant_comparison',
        '--root','external_data/cifar10_2_test','--external-runs','runs/cifar10_2_bigg',
        'runs/cifar10_2_siglip2','--baseline-dino-run','runs/cifar10_2_dinov2_l',
        '--dino-run','runs/cifar10_2_dinov2_g','--output','runs/dinov2_giant_confirmation'],
        env=environment(0),check=True)
    confirmation=json.loads(Path('runs/dinov2_giant_confirmation/metrics.json').read_text())
    if not confirmation['confirmation_passed']:
        print('New confirmation did not improve. No test export or upload.',flush=True);return
    subprocess.run([python,'-u','encode_dino_giant.py','--data','data','--splits','test',
        '--selection','runs/next3_selection_20261112.npz','--output','runs/dinov2_giant_test',
        '--batch-size','32'],env=environment(0),check=True)
    subprocess.run([python,'-u','export_dino_giant.py','--selection','runs/dinov2_giant_comparison',
        '--confirmation','runs/dinov2_giant_confirmation','--dino-test','runs/dinov2_giant_test',
        '--runs','runs/pe_bigg_selective_flip_v2','runs/siglip2_selective_flip_v2',
        '--reference','runs/dinov2_geometry_candidate','--data','data',
        '--output','runs/dinov2_giant_candidate'],env=environment(0),check=True)
    print('Confirmed giant CSV prepared; the upload to Kaggle is done separately.',flush=True)


if __name__=='__main__':main()
