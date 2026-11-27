#!/bin/bash
# Queued after run_selftrain2.sh (15 November 2026) so that the GPUs do not wait.
#   1. DINOv2-g students, cross-fitted (teacher from runs/current_recipe.json), test and validation encodes.
#      Export 'dino': the recipe's bigG and SigLIP2 choice, with the DINOv2 students in the graph.
#   2. bigG round 3, cross-fitted, teacher read again from the recipe at its start.
#      Export 'crossfit3' with the recipe's SigLIP2 and DINOv2 choice.
# runs/current_recipe.json holds the best configuration so far; it is updated by hand after each public score.
set -u
cd $HOME/recvis/A3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
MODERN=$HOME/recvis/.venv-modern/bin/python
BIGG=runs/pe_bigg_imagenet80_fp16
TAG=20261115
recipe() {
  $MODERN -c "import json; v = json.load(open('runs/current_recipe.json'))['$1']; print(' '.join(v) if isinstance(v, list) else v)"
}
ready() {
  for TRY in 1 2 3 4 5 6 7 8 9 10; do
    if CUDA_VISIBLE_DEVICES=$1 $MODERN -c "import torch; torch.zeros(1, device='cuda')" 2> /dev/null; then return 0; fi
    sleep 15
  done
  return 1
}
export_graph() {  # $1 name, then selftrain_export.py arguments
  NAME=$1
  shift
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $MODERN -u selftrain_export.py --name $NAME --tag $TAG "$@" \
    > logs/q_export_$NAME.log 2>&1 || { echo "[q] FAILED export $NAME"; tail -n 5 logs/q_export_$NAME.log; return 1; }
  grep -v Neighbours logs/q_export_$NAME.log | grep -E '^(student|graph|prior|half|differences)' | cut -c1-400
  for V in prior graph half; do
    $MODERN audit_synth_style_submission.py runs/adapted_${NAME}_${V}_$TAG > /dev/null || echo "[q] FAILED audit $NAME $V"
  done
  echo "[q] export $NAME done $(date)"
}
dino() {  # $1 fold to train on
  $MODERN -u train_dino_selftrain.py --fold $1 --teacher $TEACHER --output runs/selftrain_dino_fold$1 \
    || { echo "[q] FAILED dino train fold $1"; return 1; }
  $MODERN -u encode_dino_selftrain.py --student runs/selftrain_dino_fold$1 --split test \
    --output runs/selftrain_dino_fold${1}_test || { echo "[q] FAILED dino encode fold $1"; return 1; }
  $MODERN -u encode_dino_selftrain.py --student runs/selftrain_dino_fold$1 --split validation \
    --output runs/selftrain_dino_fold${1}_validation || echo "[q] WARNING dino validation fold $1"
  echo "[q] finished dino fold $1 $(date)"
}
bigg() {  # $1 fold to train on
  $MODERN -u train_test_selftrain.py --data test --fold $1 --teacher $TEACHER --base $BIGG \
    --output runs/selftrain3_bigg_fold$1 || { echo "[q] FAILED bigg train fold $1"; return 1; }
  $MODERN -u encode_test_selftrain.py --student runs/selftrain3_bigg_fold$1 --split test --reflected-rows selected \
    --output runs/selftrain3_bigg_fold${1}_test || { echo "[q] FAILED bigg encode fold $1"; return 1; }
  echo "[q] finished bigg round 3 fold $1 $(date)"
}
echo "[q] waiting for run_selftrain2.sh $(date)"
sha256sum run_queue_dino_r3.sh train_dino_selftrain.py encode_dino_selftrain.py selftrain_export.py
for TRY in $(seq 1 2880); do
  if grep -q "\[r2\] done" logs/selftrain2.log; then break; fi
  sleep 30
done
TEACHER=$(recipe teacher)
echo "[q] dino start $(date) teacher $TEACHER"
ready 0
ready 1
( export CUDA_VISIBLE_DEVICES=0; dino 1 ) > logs/q_dino_gpu0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; dino 0 ) > logs/q_dino_gpu1.log 2>&1 &
wait
if grep -q FAILED logs/q_dino_gpu0.log logs/q_dino_gpu1.log; then
  grep -h FAILED logs/q_dino_gpu0.log logs/q_dino_gpu1.log
  echo "[q] dino failed $(date)"
else
  grep -h 'validation head accuracy' logs/q_dino_gpu0.log logs/q_dino_gpu1.log
  ( export_graph dino --encodes $(recipe bigg) --siglip-encodes $(recipe siglip) \
      --dino-encodes runs/selftrain_dino_fold0_test runs/selftrain_dino_fold1_test ) &
fi
TEACHER=$(recipe teacher)
echo "[q] round 3 start $(date) teacher $TEACHER"
ready 0
ready 1
( export CUDA_VISIBLE_DEVICES=0; bigg 1 ) > logs/q_bigg3_gpu0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; bigg 0 ) > logs/q_bigg3_gpu1.log 2>&1 &
wait
if grep -q FAILED logs/q_bigg3_gpu0.log logs/q_bigg3_gpu1.log; then
  grep -h FAILED logs/q_bigg3_gpu0.log logs/q_bigg3_gpu1.log
  echo "[q] round 3 failed $(date)"
else
  export_graph crossfit3 --encodes runs/selftrain3_bigg_fold0_test runs/selftrain3_bigg_fold1_test \
    --siglip-encodes $(recipe siglip) --dino-encodes $(recipe dino)
fi
echo "[q] done $(date)"
