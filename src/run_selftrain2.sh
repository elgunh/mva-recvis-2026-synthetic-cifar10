#!/bin/bash
# 15 November 2026. Second round of self-training and a SigLIP2 student.
#   Part A (both GPUs): bigG round 2. Same cross-fitted recipe as round 1, but the teacher is the
#     round-1 graph candidate (runs/adapted_crossfit_graph_20261115, public 0.97761). GPU 0 trains on
#     fold 1, GPU 1 on fold 0; each student encodes only the other half of the test set (reflected
#     view only on the rows that the reference reflects).
#   Export A (CPU, in the background): bigG round 2 with frozen SigLIP2 -> runs/adapted_crossfit2_*.
#   Part B (both GPUs): SigLIP2 students with the same recipe and the same teacher.
#   Export B (CPU): bigG round 1 or round 2 together with the SigLIP2 students.
set -u
cd $HOME/recvis/A3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
MODERN=$HOME/recvis/.venv-modern/bin/python
SIGLIP=$HOME/recvis/.venv-siglip2/bin/python
BIGG=runs/pe_bigg_imagenet80_fp16
SIG=runs/siglip2_gopt_imagenet80_fp16
TEACHER=runs/adapted_crossfit_graph_20261115
TAG=20261115
ready() {
  for TRY in 1 2 3 4 5 6 7 8 9 10; do
    if CUDA_VISIBLE_DEVICES=$1 $MODERN -c "import torch; torch.zeros(1, device='cuda')" 2> /dev/null; then return 0; fi
    sleep 15
  done
  return 1
}
student() {  # $1 python, $2 base run, $3 run prefix, $4 fold to train on
  $1 -u train_test_selftrain.py --data test --fold $4 --teacher $TEACHER --base $2 --output runs/${3}_fold$4 \
    || { echo "[r2] FAILED train $3 fold $4"; return 1; }
  $1 -u encode_test_selftrain.py --student runs/${3}_fold$4 --split test --reflected-rows selected \
    --output runs/${3}_fold${4}_test || { echo "[r2] FAILED encode $3 fold $4"; return 1; }
  echo "[r2] finished $3 fold $4 $(date)"
}
export_graph() {  # $1 name, then selftrain_export.py arguments
  NAME=$1
  shift
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $MODERN -u selftrain_export.py --name $NAME --tag $TAG "$@" \
    > logs/r2_export_$NAME.log 2>&1 || { echo "[r2] FAILED export $NAME"; tail -n 5 logs/r2_export_$NAME.log; return 1; }
  grep -v Neighbours logs/r2_export_$NAME.log | grep -E '^(student|graph|prior|half|differences)' | cut -c1-400
  for V in prior graph half; do
    $MODERN audit_synth_style_submission.py runs/adapted_${NAME}_${V}_$TAG > /dev/null || echo "[r2] FAILED audit $NAME $V"
  done
  echo "[r2] export $NAME done $(date)"
}
echo "[r2] start $(date)"
sha256sum run_selftrain2.sh train_test_selftrain.py encode_test_selftrain.py selftrain_export.py train_lowres_student.py
ready 0
ready 1
( export CUDA_VISIBLE_DEVICES=0; student $MODERN $BIGG selftrain2_bigg 1 ) > logs/r2_bigg_gpu0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; student $MODERN $BIGG selftrain2_bigg 0 ) > logs/r2_bigg_gpu1.log 2>&1 &
wait
R2="runs/selftrain2_bigg_fold0_test runs/selftrain2_bigg_fold1_test"
R1="runs/selftrain_bigg_fold0_test runs/selftrain_bigg_fold1_test"
PART_A=ok
if grep -q FAILED logs/r2_bigg_gpu0.log logs/r2_bigg_gpu1.log; then
  grep -h FAILED logs/r2_bigg_gpu0.log logs/r2_bigg_gpu1.log
  PART_A=failed
else
  ( export_graph crossfit2 --encodes $R2 ) &
fi
echo "[r2] part A $PART_A $(date)"
ready 0
ready 1
( export CUDA_VISIBLE_DEVICES=0; student $SIGLIP $SIG selftrain_siglip2 1 ) > logs/r2_siglip_gpu0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; student $SIGLIP $SIG selftrain_siglip2 0 ) > logs/r2_siglip_gpu1.log 2>&1 &
wait
if grep -q FAILED logs/r2_siglip_gpu0.log logs/r2_siglip_gpu1.log; then
  grep -h FAILED logs/r2_siglip_gpu0.log logs/r2_siglip_gpu1.log
  echo "[r2] part B failed $(date)"
else
  SG="runs/selftrain_siglip2_fold0_test runs/selftrain_siglip2_fold1_test"
  export_graph crossfit1sig --encodes $R1 --siglip-encodes $SG
  if [ $PART_A = ok ]; then export_graph crossfit2sig --encodes $R2 --siglip-encodes $SG; fi
fi
echo "[r2] done $(date)"
