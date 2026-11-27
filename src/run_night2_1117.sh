#!/bin/bash
# 17 November 2026, replaces run_night_1117.sh after the first epoch of the quick X2 student (one DINOv2-g
# model reached 97.7% on held-out generated images after one epoch, above the frozen reference pipeline).
# GPU 1: waits for the quick X2 (already running), then trains bigG with labels on seeds 24..71 and encodes
#        the test set with it; then generates new seeds until GPU 0 is ready.
# GPU 0: finishes its generation shard (started by run_x1.sh), then generates the remaining odd jobs.
# When seeds 0..119 are complete: bigG with labels on all of them (GPU 0) and DINOv2-g with labels (GPU 1);
# afterwards both GPUs generate new seeds (32px saved at generation time).
set -u
cd $HOME/recvis/A3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
GEN=$HOME/recvis/.venv-gen/bin/python
MODERN=$HOME/recvis/.venv-modern/bin/python
OUT=external_data/synth_style_gen/sd21_768
KFILE=external_data/x1_kernel_sd21_768.json
QUICK=sd21_768_train72
FULL=sd21_768_train120
BASE=runs/adapted_bag3_graph_20261116
A="runs/selftrain_bigg_fold0_test runs/selftrain_bigg_fold1_test"
B="runs/selftrainB_bigg_fold0_test runs/selftrainB_bigg_fold1_test"
C="runs/selftrainC_bigg_fold0_test runs/selftrainC_bigg_fold1_test"
OTHERS="--siglip-encodes runs/selftrain_siglip2_fold0_test runs/selftrain_siglip2_fold1_test --dino-encodes runs/selftrain_dino_fold0_test runs/selftrain_dino_fold1_test"
echo "[night2] start $(date)"
sha256sum run_night2_1117.sh train_bigg_synth.py train_synth_supervised.py encode_test_selftrain.py encode_dino_selftrain.py \
  selftrain_export.py x2_export.py x2_pilot_check.py generate_style_images.py prepare_synth_manifest.py
more_seeds() {  # $1 gpu: new seeds, 32px saved at generation time, no large files
  CUDA_VISIBLE_DEVICES=$1 $GEN -u generate_style_images.py --generator sd21_768 --output $OUT --start 120 --end 400 \
    --shard $1 2 --kernel-file $KFILE >> logs/night_more$1.log 2>&1
}
bigg() {  # $1 gpu, $2 set name, $3 output; the student saw no test image, so it encodes the whole test set
  export CUDA_VISIBLE_DEVICES=$1
  $MODERN -u train_bigg_synth.py --sets $2 --clean top2 --epochs 2 --augment strong --output $3 \
    || { echo "[night2] FAILED bigG train $3"; return 1; }
  $MODERN -u encode_test_selftrain.py --student $3 --split test --reflected-rows selected --output ${3}_test \
    || { echo "[night2] FAILED bigG test encode $3"; return 1; }
  echo "[night2] bigG student $3 done $(date)"
}
bag() {  # $1 name, $2 encodes of the supervised bigG student: bag3 plus this student as a fourth bigG group
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $MODERN -u selftrain_export.py --name $1 --tag 20261117 \
    --encodes $A --encodes $B --encodes $C --encodes $2 $OTHERS > logs/x3_export_$1.log 2>&1 \
    || { echo "[night2] FAILED export $1"; return 1; }
  grep -v Neighbours logs/x3_export_$1.log | grep -E '^graph' | cut -c1-300
  $MODERN audit_synth_style_submission.py runs/adapted_${1}_graph_20261117 > /dev/null || echo "[night2] FAILED audit $1"
  echo "[night2] export $1 done $(date)"
}
lane1() {
  until grep -q -E "x2\] done $QUICK|FAILED" logs/x2_quick.log; do sleep 60; done
  echo "[night2] quick X2 finished $(date)"
  ( bigg 1 $QUICK runs/x3_bigg_sup72 ) > logs/x3_bigg_sup72.log 2>&1
  grep -h -E 'held-out generated|FAILED' logs/x3_bigg_sup72.log | cut -c1-300
  if [ -d runs/x3_bigg_sup72_test ]; then bag bag3sup72 runs/x3_bigg_sup72_test; fi
  touch logs/night2_lane1_ready
  more_seeds 1 &
  echo $! > logs/night2_filler.pid
  wait
}
lane0() {
  until grep -q -E 'Completed sd21_768|FAILED' logs/x1_more_gpu0.log; do sleep 60; done
  echo "[night2] GPU 0 finished its shard $(date)"
  CUDA_VISIBLE_DEVICES=0 $GEN -u generate_style_images.py --generator sd21_768 --output $OUT --start 24 --end 120 \
    --shard 1 2 --save-native >> logs/night_gen0.log 2>&1
  echo "[night2] seeds 0..119 generated $(date)"
}
lane1 &
lane0
CUDA_VISIBLE_DEVICES=0 $MODERN -u prepare_synth_manifest.py build --gen-roots $OUT --kernel-file $KFILE --role train \
  --max-index 120 --expected 15600 --output external_data/synth_style_$FULL > logs/x3_manifest120.log 2>&1 \
  && CUDA_VISIBLE_DEVICES=0 $MODERN -u judge_synth.py --root external_data/synth_style_$FULL \
    --output runs/synth_${FULL}_judge >> logs/x3_manifest120.log 2>&1 \
  || echo "[night2] FAILED manifest or judge $FULL"
gpu0_full() {
  ( bigg 0 $FULL runs/x3_bigg_sup120 ) > logs/x3_bigg_sup120.log 2>&1
  grep -h -E 'held-out generated|FAILED' logs/x3_bigg_sup120.log | cut -c1-300
  if [ -d runs/x3_bigg_sup120_test ]; then bag bag3sup120 runs/x3_bigg_sup120_test; fi
  more_seeds 0
}
gpu1_full() {
  until [ -f logs/night2_lane1_ready ]; do sleep 30; done
  if [ -f logs/night2_filler.pid ]; then pkill -P $(cat logs/night2_filler.pid) 2> /dev/null; kill $(cat logs/night2_filler.pid) 2> /dev/null; fi
  sleep 30
  export CUDA_VISIBLE_DEVICES=1
  S=runs/x2_dino_${FULL}_top2
  ( $MODERN -u train_synth_supervised.py --sets $FULL --clean top2 --output $S \
    && $MODERN -u encode_dino_selftrain.py --student $S --split validation --output ${S}_validation \
    && $MODERN -u encode_dino_selftrain.py --student $S --split test --output ${S}_test ) > logs/x2_${FULL}_top2.log 2>&1 \
    || echo "[night2] FAILED DINOv2 student $FULL"
  grep -h -E '^\{"epoch"|validation head accuracy' logs/x2_${FULL}_top2.log | cut -c1-400
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $MODERN -u x2_pilot_check.py --pilot sd21_768 --students $S \
    --output runs/x2_pilot_check_${FULL}.json 2>&1 | grep -v Neighbours
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $MODERN -u x2_export.py --name s1_$FULL --tag 20261117 --base $BASE \
    --base-public 0.98048 --student ${S}_test 2>&1 | grep -v Neighbours
  $MODERN audit_synth_style_submission.py runs/x2_s1_${FULL}_20261117 > /dev/null || echo "[night2] FAILED audit s1 $FULL"
  echo "[night2] DINOv2 student $FULL done $(date)"
  more_seeds 1
}
gpu0_full > logs/x3_gpu0_full.log 2>&1 &
gpu1_full > logs/x3_gpu1_full.log 2>&1 &
wait
echo "[night2] done $(date)"
