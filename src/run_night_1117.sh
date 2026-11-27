#!/bin/bash
# Night of 16-17 November 2026, after the X1 results (SD 2.1 at 768px mixes with the test set).
# GPU 1: quick X2 on the seeds that are already complete ($1), then the rest of the odd generation jobs.
# GPU 0: finishes its running generation shard (started by run_x1.sh), then helps with the odd jobs.
# When seeds 0..119 are complete: full X2 on both GPUs. Afterwards both GPUs generate new seeds with the
# 32px image saved at generation time (no large files).
set -u
cd $HOME/recvis/A3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
GEN=$HOME/recvis/.venv-gen/bin/python
OUT=external_data/synth_style_gen/sd21_768
KFILE=external_data/x1_kernel_sd21_768.json
K=$1
echo "[night] start, complete seeds $K $(date)"
sha256sum run_night_1117.sh run_x2.sh generate_style_images.py prepare_synth_manifest.py train_synth_supervised.py x2_pilot_check.py x2_export.py
lane1() {
  bash run_x2.sh sd21_768 $K 1 1 > logs/x2_quick.log 2>&1
  echo "[night] quick X2 finished $(date)"
  CUDA_VISIBLE_DEVICES=1 $GEN -u generate_style_images.py --generator sd21_768 --output $OUT --start 24 --end 120 \
    --shard 1 4 --save-native >> logs/night_gen1.log 2>&1
  touch logs/night_lane1_ready
  echo "[night] lane 1 generation part done $(date)"
  CUDA_VISIBLE_DEVICES=1 $GEN -u generate_style_images.py --generator sd21_768 --output $OUT --start 120 --end 400 \
    --shard 1 2 --kernel-file $KFILE >> logs/night_more1.log 2>&1 &
  echo $! > logs/night_filler.pid
  wait
}
lane0() {
  until grep -q -E 'Completed sd21_768|FAILED' logs/x1_more_gpu0.log; do sleep 60; done
  echo "[night] GPU 0 finished its shard $(date)"
  CUDA_VISIBLE_DEVICES=0 $GEN -u generate_style_images.py --generator sd21_768 --output $OUT --start 24 --end 120 \
    --shard 3 4 --save-native >> logs/night_gen0.log 2>&1
  echo "[night] lane 0 generation part done $(date)"
}
lane1 &
lane0
until [ -f logs/night_lane1_ready ]; do sleep 30; done
if [ -f logs/night_filler.pid ]; then kill $(cat logs/night_filler.pid) 2> /dev/null; fi
sleep 30
echo "[night] seeds 0..119 complete, full X2 $(date)"
bash run_x2.sh sd21_768 120 0 1 > logs/x2_full.log 2>&1
echo "[night] full X2 finished $(date)"
for G in 0 1; do
  CUDA_VISIBLE_DEVICES=$G $GEN -u generate_style_images.py --generator sd21_768 --output $OUT --start 120 --end 400 \
    --shard $G 2 --kernel-file $KFILE >> logs/night_more$G.log 2>&1 &
done
wait
echo "[night] done $(date)"
