#!/bin/bash
# 15-16 November 2026. Second, independent split of the test set (split seed 20261116), so that every
# test image gets two held-out students per encoder; the export averages them.
#   1. bigG, round-1 recipe: teacher = the 0.97565 reference (the round-2 teacher, the 0.97761 graph
#      file, gave 0.97598, worse than round 1). Exports 'bagbigg' (bigG A+B) and 'bagbigg_sigdino'
#      (bigG A+B with the split-A SigLIP2 and DINOv2 students) start on CPU at once.
#   2. DINOv2 split B, 3. SigLIP2 split B (same recipes and teacher as split A: the 0.97761 file).
#   4. Export 'bagall': every encoder with students from both splits.
set -u
cd $HOME/recvis/A3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
MODERN=$HOME/recvis/.venv-modern/bin/python
SIGLIP=$HOME/recvis/.venv-siglip2/bin/python
REFERENCE=runs/dinov2_giant_candidate
GRAPH1=runs/adapted_crossfit_graph_20261115
SPLIT=20261116
TAG=20261116
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
    > logs/b_export_$NAME.log 2>&1 || { echo "[b] FAILED export $NAME"; tail -n 5 logs/b_export_$NAME.log; return 1; }
  grep -v Neighbours logs/b_export_$NAME.log | grep -E '^(graph|differences)' | cut -c1-300
  $MODERN audit_synth_style_submission.py runs/adapted_${NAME}_graph_$TAG > /dev/null || echo "[b] FAILED audit $NAME"
  echo "[b] export $NAME done $(date)"
}
bigg() {  # $1 fold
  $MODERN -u train_test_selftrain.py --data test --fold $1 --split-seed $SPLIT --teacher $REFERENCE \
    --output runs/selftrainB_bigg_fold$1 || { echo "[b] FAILED bigg train fold $1"; return 1; }
  $MODERN -u encode_test_selftrain.py --student runs/selftrainB_bigg_fold$1 --split test --reflected-rows selected \
    --output runs/selftrainB_bigg_fold${1}_test || { echo "[b] FAILED bigg encode fold $1"; return 1; }
  echo "[b] finished bigg fold $1 $(date)"
}
dino() {  # $1 fold
  $MODERN -u train_dino_selftrain.py --fold $1 --split-seed $SPLIT --teacher $GRAPH1 \
    --output runs/selftrainB_dino_fold$1 || { echo "[b] FAILED dino train fold $1"; return 1; }
  $MODERN -u encode_dino_selftrain.py --student runs/selftrainB_dino_fold$1 --split test \
    --output runs/selftrainB_dino_fold${1}_test || { echo "[b] FAILED dino encode fold $1"; return 1; }
  echo "[b] finished dino fold $1 $(date)"
}
siglip() {  # $1 fold
  $SIGLIP -u train_test_selftrain.py --data test --fold $1 --split-seed $SPLIT --teacher $GRAPH1 \
    --base runs/siglip2_gopt_imagenet80_fp16 --output runs/selftrainB_siglip2_fold$1 \
    || { echo "[b] FAILED siglip train fold $1"; return 1; }
  $SIGLIP -u encode_test_selftrain.py --student runs/selftrainB_siglip2_fold$1 --split test --reflected-rows selected \
    --output runs/selftrainB_siglip2_fold${1}_test || { echo "[b] FAILED siglip encode fold $1"; return 1; }
  echo "[b] finished siglip fold $1 $(date)"
}
stage() {  # $1 function name; runs fold 1 on GPU 0 and fold 0 on GPU 1, logs logs/b_$1_gpu*.log
  ready 0
  ready 1
  ( export CUDA_VISIBLE_DEVICES=0; $1 1 ) > logs/b_$1_gpu0.log 2>&1 &
  ( export CUDA_VISIBLE_DEVICES=1; $1 0 ) > logs/b_$1_gpu1.log 2>&1 &
  wait
  if grep -q FAILED logs/b_$1_gpu0.log logs/b_$1_gpu1.log; then grep -h FAILED logs/b_$1_gpu0.log logs/b_$1_gpu1.log; return 1; fi
  echo "[b] stage $1 ok $(date)"
}
A_BIGG="runs/selftrain_bigg_fold0_test runs/selftrain_bigg_fold1_test"
A_SIG="runs/selftrain_siglip2_fold0_test runs/selftrain_siglip2_fold1_test"
A_DINO="runs/selftrain_dino_fold0_test runs/selftrain_dino_fold1_test"
B_BIGG="runs/selftrainB_bigg_fold0_test runs/selftrainB_bigg_fold1_test"
B_SIG="runs/selftrainB_siglip2_fold0_test runs/selftrainB_siglip2_fold1_test"
B_DINO="runs/selftrainB_dino_fold0_test runs/selftrainB_dino_fold1_test"
echo "[b] start $(date)"
sha256sum run_bagging_b.sh train_test_selftrain.py encode_test_selftrain.py train_dino_selftrain.py encode_dino_selftrain.py selftrain_export.py
BIGG_OK=no; DINO_OK=no; SIG_OK=no
if stage bigg; then
  BIGG_OK=yes
  ( export_graph bagbigg --encodes $A_BIGG --encodes $B_BIGG
    export_graph bagbigg_sigdino --encodes $A_BIGG --encodes $B_BIGG --siglip-encodes $A_SIG --dino-encodes $A_DINO ) &
fi
if stage dino; then DINO_OK=yes; fi
if stage siglip; then SIG_OK=yes; fi
wait
ARGS="--encodes $A_BIGG --siglip-encodes $A_SIG --dino-encodes $A_DINO"
if [ $BIGG_OK = yes ]; then ARGS="$ARGS --encodes $B_BIGG"; fi
if [ $SIG_OK = yes ]; then ARGS="$ARGS --siglip-encodes $B_SIG"; fi
if [ $DINO_OK = yes ]; then ARGS="$ARGS --dino-encodes $B_DINO"; fi
echo "[b] bagall uses: bigg B $BIGG_OK, siglip B $SIG_OK, dino B $DINO_OK"
export_graph bagall $ARGS
echo "[b] done $(date)"
