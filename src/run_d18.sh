#!/bin/bash
# 18 November 2026. Best public file: 0.98897 (S1 rule with a DINOv2-g student trained on 20,953 own
# labelled images; leader 0.98916). 438 seeds (56,940 images) are generated.
# GPU 0: manifest and judge for seeds 0..437, then a DINOv2-g student on seeds 24..399 (about 48,000
#        images; seeds 400..437 stay unused as a second held-out set) and the S1 file with it.
# GPU 1: input-size test: the 12,417-image student of 17 November (98.55% held-out at 224px) again at
#        336px, everything else equal; then its test encode and S1 file.
# Afterwards each GPU generates new seeds.
set -u
cd $HOME/recvis/A3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
GEN=$HOME/recvis/.venv-gen/bin/python
MODERN=$HOME/recvis/.venv-modern/bin/python
OUT=external_data/synth_style_gen/sd21_768
KFILE=external_data/x1_kernel_sd21_768.json
BASE=runs/adapted_bag3_graph_20261116
TAG=20261118
ALL=sd21_768_train438
echo "[d18] start $(date)"
sha256sum run_d18.sh train_synth_supervised.py train_dino_selftrain.py encode_dino_selftrain.py x2_export.py \
  x2_pilot_check.py prepare_synth_manifest.py
gen() {  # $1 gpu: new seeds, 32px saved at generation time
  CUDA_VISIBLE_DEVICES=$1 $GEN -u generate_style_images.py --generator sd21_768 --output $OUT --start 438 --end 900 \
    --shard $1 2 --kernel-file $KFILE >> logs/d18_more$1.log 2>&1
}
student() {  # $1 gpu, $2 set, $3 student folder, $4 file name, $5.. extra training flags
  export CUDA_VISIBLE_DEVICES=$1
  SET=$2; S=$3; FILE=$4; shift 4
  $MODERN -u train_synth_supervised.py --sets $SET --clean top2 --output $S "$@" \
    && $MODERN -u encode_dino_selftrain.py --student $S --split validation --output ${S}_validation \
    && $MODERN -u encode_dino_selftrain.py --student $S --split test --output ${S}_test \
    || { echo "[d18] FAILED student $S"; return 1; }
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $MODERN -u x2_pilot_check.py --pilot sd21_768 --students $S \
    --output runs/x2_pilot_check_$FILE.json 2>&1 | grep -v Neighbours
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $MODERN -u x2_export.py --name $FILE --tag $TAG --base $BASE \
    --base-public 0.98048 --student ${S}_test 2>&1 | grep -v Neighbours
  $MODERN audit_synth_style_submission.py runs/x2_${FILE}_$TAG > /dev/null || echo "[d18] FAILED audit $FILE"
  echo "[d18] student $S done $(date)"
}
lane0() {
  CUDA_VISIBLE_DEVICES=0 $MODERN -u prepare_synth_manifest.py build --gen-roots $OUT --kernel-file $KFILE --role train \
    --max-index 438 --expected 56940 --output external_data/synth_style_$ALL > logs/d18_manifest.log 2>&1 \
    && CUDA_VISIBLE_DEVICES=0 $MODERN -u judge_synth.py --root external_data/synth_style_$ALL \
      --output runs/synth_${ALL}_judge >> logs/d18_manifest.log 2>&1 \
    || echo "[d18] FAILED manifest or judge"
  ( student 0 $ALL runs/x2_dino_sd21_768_train400_top2 s1_sd21_768_train400 --max-train-index 400 ) > logs/d18_big.log 2>&1
  grep -h -E '^\{"epoch"|validation head accuracy|X2_PILOT|FAILED|d18\]' logs/d18_big.log | cut -c1-700
  echo "[d18] lane 0 done $(date)"
  gen 0
}
lane1() {
  ( student 1 sd21_768_train120 runs/x2_dino_sd21_768_train120_top2_r336 s1_sd21_768_train120_r336 \
      --image-size 336 --batch-size 16 ) > logs/d18_r336.log 2>&1
  grep -h -E '^\{"epoch"|validation head accuracy|X2_PILOT|FAILED|d18\]' logs/d18_r336.log | cut -c1-700
  echo "[d18] lane 1 done $(date)"
  gen 1
}
lane0 &
lane1 &
wait
echo "[d18] done $(date)"
