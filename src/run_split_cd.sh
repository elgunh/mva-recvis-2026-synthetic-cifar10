#!/bin/bash
# 16 November 2026. Two more independent splits of the test set for bigG (seeds 20261117 and 20261118),
# teacher = the 0.97565 reference as in round 1, but with strong input noise (noisy-student style).
# Splits A and B (weak noise) with SigLIP2 and DINOv2 split-A students gave 0.97937.
#   C -> export 'bag3' = bigG A+B+C with SigLIP2 A and DINOv2 A
#   D -> export 'bag4' = bigG A+B+C+D with SigLIP2 A and DINOv2 A
set -u
cd $HOME/recvis/A3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
MODERN=$HOME/recvis/.venv-modern/bin/python
REFERENCE=runs/dinov2_giant_candidate
TAG=20261116
ready() {
  for TRY in 1 2 3 4 5 6 7 8 9 10; do
    if CUDA_VISIBLE_DEVICES=$1 $MODERN -c "import torch; torch.zeros(1, device='cuda')" 2> /dev/null; then return 0; fi
    sleep 15
  done
  return 1
}
export_graph() {
  NAME=$1
  shift
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $MODERN -u selftrain_export.py --name $NAME --tag $TAG "$@" \
    > logs/cd_export_$NAME.log 2>&1 || { echo "[cd] FAILED export $NAME"; tail -n 5 logs/cd_export_$NAME.log; return 1; }
  grep -v Neighbours logs/cd_export_$NAME.log | grep -E '^graph' | cut -c1-200
  $MODERN audit_synth_style_submission.py runs/adapted_${NAME}_graph_$TAG > /dev/null || echo "[cd] FAILED audit $NAME"
  echo "[cd] export $NAME done $(date)"
}
bigg() {  # $1 split letter, $2 split seed, $3 fold
  $MODERN -u train_test_selftrain.py --data test --fold $3 --split-seed $2 --teacher $REFERENCE --augment strong \
    --output runs/selftrain${1}_bigg_fold$3 || { echo "[cd] FAILED bigg $1 train fold $3"; return 1; }
  $MODERN -u encode_test_selftrain.py --student runs/selftrain${1}_bigg_fold$3 --split test --reflected-rows selected \
    --output runs/selftrain${1}_bigg_fold${3}_test || { echo "[cd] FAILED bigg $1 encode fold $3"; return 1; }
  echo "[cd] finished bigg $1 fold $3 $(date)"
}
split() {  # $1 letter, $2 seed
  ready 0
  ready 1
  ( export CUDA_VISIBLE_DEVICES=0; bigg $1 $2 1 ) > logs/cd_${1}_gpu0.log 2>&1 &
  ( export CUDA_VISIBLE_DEVICES=1; bigg $1 $2 0 ) > logs/cd_${1}_gpu1.log 2>&1 &
  wait
  if grep -q FAILED logs/cd_${1}_gpu0.log logs/cd_${1}_gpu1.log; then grep -h FAILED logs/cd_${1}_gpu0.log logs/cd_${1}_gpu1.log; return 1; fi
  echo "[cd] split $1 ok $(date)"
}
A="runs/selftrain_bigg_fold0_test runs/selftrain_bigg_fold1_test"
B="runs/selftrainB_bigg_fold0_test runs/selftrainB_bigg_fold1_test"
C="runs/selftrainC_bigg_fold0_test runs/selftrainC_bigg_fold1_test"
D="runs/selftrainD_bigg_fold0_test runs/selftrainD_bigg_fold1_test"
OTHERS="--siglip-encodes runs/selftrain_siglip2_fold0_test runs/selftrain_siglip2_fold1_test --dino-encodes runs/selftrain_dino_fold0_test runs/selftrain_dino_fold1_test"
echo "[cd] start $(date)"
sha256sum run_split_cd.sh train_test_selftrain.py encode_test_selftrain.py selftrain_export.py
C_OK=no
if split C 20261117; then
  C_OK=yes
  ( export_graph bag3 --encodes $A --encodes $B --encodes $C $OTHERS ) &
fi
if split D 20261118; then
  if [ $C_OK = yes ]; then
    export_graph bag4 --encodes $A --encodes $B --encodes $C --encodes $D $OTHERS
  else
    export_graph bag4 --encodes $A --encodes $B --encodes $D $OTHERS
  fi
fi
wait
echo "[cd] done $(date)"
