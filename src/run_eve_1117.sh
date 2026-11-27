#!/bin/bash
# Evening of 17 November 2026, after file S1 scored 0.98701 (second place; leader 0.98727).
# Phase 1: both GPUs generate the missing even-class jobs of seeds 120..185 (the odd-class jobs are done).
# Phase 2: DINOv2-g with labels on seeds 24..185 (GPU 1) -> file S1 with this student.
#          Joint cross-fitted DINOv2-g students: the same labelled images plus one test fold with the soft
#          targets of the 0.98701 file (fold 0 on GPU 0, fold 1 on GPU 1 after the supervised student).
# Phase 3: both GPUs generate new seeds.
set -u
cd $HOME/recvis/A3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
GEN=$HOME/recvis/.venv-gen/bin/python
MODERN=$HOME/recvis/.venv-modern/bin/python
OUT=external_data/synth_style_gen/sd21_768
KFILE=external_data/x1_kernel_sd21_768.json
END=186
NAME=sd21_768_train$END
BASE=runs/adapted_bag3_graph_20261116
TEACHER=runs/x2_s1_sd21_768_train120_20261117
TAG=20261117
echo "[eve] start $(date)"
sha256sum run_eve_1117.sh train_synth_supervised.py encode_dino_selftrain.py x2_export.py x2_pilot_check.py \
  generate_style_images.py prepare_synth_manifest.py
gen() {  # $1 gpu, $2 start, $3 end, $4 shard, $5 shards, $6 log
  CUDA_VISIBLE_DEVICES=$1 $GEN -u generate_style_images.py --generator sd21_768 --output $OUT --start $2 --end $3 \
    --shard $4 $5 --kernel-file $KFILE >> $6 2>&1
}
gen 0 120 $END 0 4 logs/eve_gen0.log &
gen 1 120 $END 2 4 logs/eve_gen1.log &
wait
echo "[eve] seeds 0..$((END-1)) generated $(date)"
CUDA_VISIBLE_DEVICES=0 $MODERN -u prepare_synth_manifest.py build --gen-roots $OUT --kernel-file $KFILE --role train \
  --max-index $END --expected $((130*END)) --output external_data/synth_style_$NAME > logs/eve_manifest.log 2>&1 \
  && CUDA_VISIBLE_DEVICES=0 $MODERN -u judge_synth.py --root external_data/synth_style_$NAME \
    --output runs/synth_${NAME}_judge >> logs/eve_manifest.log 2>&1 \
  || { echo "[eve] FAILED manifest or judge"; }
encodes() {  # $1 student folder: CIFAR validation (guard) and the test rows the student may encode
  $MODERN -u encode_dino_selftrain.py --student $1 --split validation --output ${1}_validation \
    && $MODERN -u encode_dino_selftrain.py --student $1 --split test --output ${1}_test
}
supervised() {
  export CUDA_VISIBLE_DEVICES=1
  S=runs/x2_dino_${NAME}_top2
  ( $MODERN -u train_synth_supervised.py --sets $NAME --clean top2 --output $S && encodes $S ) > logs/eve_sup.log 2>&1 \
    || echo "[eve] FAILED supervised student"
  grep -h -E '^\{"epoch"|validation head accuracy' logs/eve_sup.log | cut -c1-400
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $MODERN -u x2_pilot_check.py --pilot sd21_768 --students $S \
    --output runs/x2_pilot_check_${NAME}.json 2>&1 | grep -v Neighbours
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $MODERN -u x2_export.py --name s1_$NAME --tag $TAG --base $BASE \
    --base-public 0.98048 --student ${S}_test 2>&1 | grep -v Neighbours
  $MODERN audit_synth_style_submission.py runs/x2_s1_${NAME}_$TAG > /dev/null || echo "[eve] FAILED audit s1"
  echo "[eve] supervised student done $(date)"
}
joint() {  # $1 gpu, $2 test fold
  export CUDA_VISIBLE_DEVICES=$1
  S=runs/x3_dino_joint_${NAME}_fold$2
  ( $MODERN -u train_synth_supervised.py --sets $NAME --clean top2 --test-fold $2 --teacher $TEACHER --output $S \
    && encodes $S ) > logs/eve_joint$2.log 2>&1 || echo "[eve] FAILED joint student fold $2"
  grep -h -E '^\{"epoch"|validation head accuracy' logs/eve_joint$2.log | cut -c1-400
  echo "[eve] joint student fold $2 done $(date)"
}
rm -f logs/eve_joint0_ready logs/eve_joint1_ready
( joint 0 0; touch logs/eve_joint0_ready; gen 0 186 500 0 2 logs/eve_more0.log ) &
( supervised; joint 1 1; touch logs/eve_joint1_ready; gen 1 186 500 1 2 logs/eve_more1.log ) &
until [ -f logs/eve_joint0_ready ] && [ -f logs/eve_joint1_ready ]; do sleep 60; done
J=runs/x3_dino_joint_${NAME}
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $MODERN -u x2_pilot_check.py --pilot sd21_768 --students ${J}_fold0 ${J}_fold1 \
  --output runs/x2_pilot_check_joint_${NAME}.json 2>&1 | grep -v Neighbours
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $MODERN -u x2_export.py --name s1joint_$NAME --tag $TAG --base $BASE \
  --base-public 0.98048 --student ${J}_fold0_test ${J}_fold1_test 2>&1 | grep -v Neighbours
$MODERN audit_synth_style_submission.py runs/x2_s1joint_${NAME}_$TAG > /dev/null || echo "[eve] FAILED audit s1joint"
echo "[eve] joint file done $(date)"
wait
echo "[eve] done $(date)"
