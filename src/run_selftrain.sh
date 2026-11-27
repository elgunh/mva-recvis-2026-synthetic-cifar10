#!/bin/bash
# Night run: adapt bigG to the 32px competition images.
# Stage 1 (both GPUs, about 5 hours): cross-fitted self-training on the test set.
#   GPU 0 trains on fold 1 and encodes fold 0; GPU 1 trains on fold 0 and encodes fold 1.
# Stage 2 (until the morning):
#   GPU 0: supervised adaptation on CIFAR-10 train, then encodes the test set (original view).
#   GPU 1: self-training on all test images, then encodes the test set (both views);
#          afterwards it encodes the reflected view for the CIFAR-10 student.
# Starts when the pilot has finished.
set -u
cd $HOME/recvis/A3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
MODERN=$HOME/recvis/.venv-modern/bin/python
ready() {
  for TRY in 1 2 3 4 5 6 7 8 9 10; do
    if CUDA_VISIBLE_DEVICES=$1 $MODERN -c "import torch; torch.zeros(1, device='cuda')" 2> /dev/null; then return 0; fi
    sleep 15
  done
  return 1
}
echo "[selftrain] waiting for the pilot $(date)"
for TRY in $(seq 1 720); do
  if grep -q "student. done\|FAILED before" logs/student_pilot.log; then break; fi
  sleep 15
done
sleep 10
echo "[selftrain] start $(date)"
sha256sum train_test_selftrain.py encode_test_selftrain.py train_lowres_student.py
ready 0
ready 1
smoke() {  # every configuration of the night is checked before the long runs
  for SPLIT in test validation; do  # encoder code check with the pilot student, 32 images
    $MODERN -u encode_test_selftrain.py --student runs/lowres_student_kandinsky22 --split $SPLIT --limit 32 \
      --output runs/_smoke_encode_${SPLIT}_fold$1 || echo "[selftrain] WARNING encode smoke failed $SPLIT"
  done
  $MODERN -u train_test_selftrain.py --data test --fold $1 --output runs/_smoke_selftrain --smoke-steps 2 \
    && $MODERN -u train_test_selftrain.py --data $2 --fold -1 --output runs/_smoke_selftrain --smoke-steps 2
}
stage1() {  # $1 = fold to train on
  $MODERN -u train_test_selftrain.py --data test --fold $1 --output runs/selftrain_bigg_fold$1 \
    || { echo "[selftrain] FAILED train fold $1"; return 1; }
  $MODERN -u encode_test_selftrain.py --student runs/selftrain_bigg_fold$1 --split test \
    --output runs/selftrain_bigg_fold${1}_test || { echo "[selftrain] FAILED encode test fold $1"; return 1; }
  $MODERN -u encode_test_selftrain.py --student runs/selftrain_bigg_fold$1 --split validation \
    --output runs/selftrain_bigg_fold${1}_validation || echo "[selftrain] FAILED encode validation fold $1"
  echo "[selftrain] stage 1 finished fold $1 $(date)"
}
(
  export CUDA_VISIBLE_DEVICES=0
  smoke 1 train || { echo "[selftrain] FAILED smoke gpu0"; exit 1; }
  echo "[selftrain] training started gpu0 $(date)"
  stage1 1
  $MODERN -u train_test_selftrain.py --data train --output runs/adapt_bigg_cifar_train \
    || { echo "[selftrain] FAILED train cifar"; exit 1; }
  $MODERN -u encode_test_selftrain.py --student runs/adapt_bigg_cifar_train --split validation \
    --output runs/adapt_bigg_cifar_train_validation || echo "[selftrain] FAILED encode validation cifar"
  $MODERN -u encode_test_selftrain.py --student runs/adapt_bigg_cifar_train --split test --views original \
    --output runs/adapt_bigg_cifar_train_test_original || echo "[selftrain] FAILED encode test cifar original"
  echo "[selftrain] gpu0 finished $(date)"
) > logs/selftrain_gpu0.log 2>&1 &
(
  export CUDA_VISIBLE_DEVICES=1
  smoke 0 test || { echo "[selftrain] FAILED smoke gpu1"; exit 1; }
  echo "[selftrain] training started gpu1 $(date)"
  stage1 0
  $MODERN -u train_test_selftrain.py --data test --fold -1 --output runs/selftrain_bigg_all \
    || { echo "[selftrain] FAILED train all"; exit 1; }
  $MODERN -u encode_test_selftrain.py --student runs/selftrain_bigg_all --split test \
    --output runs/selftrain_bigg_all_test || echo "[selftrain] FAILED encode test all"
  for TRY in $(seq 1 480); do
    if [ -f runs/adapt_bigg_cifar_train/lora.pt ] || grep -q "FAILED train cifar" logs/selftrain_gpu0.log; then break; fi
    sleep 30
  done
  if [ -f runs/adapt_bigg_cifar_train/lora.pt ]; then
    $MODERN -u encode_test_selftrain.py --student runs/adapt_bigg_cifar_train --split test --views reflected \
      --output runs/adapt_bigg_cifar_train_test_reflected || echo "[selftrain] FAILED encode test cifar reflected"
  fi
  echo "[selftrain] gpu1 finished $(date)"
) > logs/selftrain_gpu1.log 2>&1 &
wait
grep -h FAILED logs/selftrain_gpu0.log logs/selftrain_gpu1.log
echo "[selftrain] done $(date)"
