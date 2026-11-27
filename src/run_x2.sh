#!/bin/bash
# X2 of the 16 November 2026 plan: DINOv2-g with LoRA trained with cross-entropy on my own generated
# images of one recipe (labels = prompt classes), then file S1 (x2_export.py). Two training sets:
# cleaned by the judge (top 2; this student makes the file) and uncleaned (used for the agreement check).
# Usage: run_x2.sh <generator> <end seed index> [gpu of the cleaned student] [gpu of the uncleaned student]
# With the same GPU twice the two students are trained one after the other.
# Seeds below 24 are held out and scored with labels (x2_pilot_check.py).
set -u
cd $HOME/recvis/A3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
MODERN=$HOME/recvis/.venv-modern/bin/python
GEN=$1
END=$2
GA=${3:-0}
GB=${4:-1}
NAME=${GEN}_train$END
TAG=20261117
ROOT=external_data/synth_style_$NAME
BASE=runs/adapted_bag3_graph_20261116
echo "[x2] start $NAME gpus $GA $GB $(date)"
sha256sum run_x2.sh train_synth_supervised.py encode_dino_selftrain.py x2_export.py x2_pilot_check.py prepare_synth_manifest.py
if [ ! -d $ROOT ]; then
  CUDA_VISIBLE_DEVICES=$GA $MODERN -u prepare_synth_manifest.py build --gen-roots external_data/synth_style_gen/$GEN \
    --kernel-file external_data/x1_kernel_$GEN.json --role train --max-index $END --expected $((130*END)) \
    --output $ROOT || { echo "[x2] FAILED manifest $NAME"; exit 1; }
fi
if [ ! -d runs/synth_${NAME}_judge ]; then
  CUDA_VISIBLE_DEVICES=$GA $MODERN -u judge_synth.py --root $ROOT --output runs/synth_${NAME}_judge \
    || { echo "[x2] FAILED judge $NAME"; exit 1; }
fi
student() {  # $1 gpu, $2 cleaning
  export CUDA_VISIBLE_DEVICES=$1
  S=runs/x2_dino_${NAME}_$2
  $MODERN -u train_synth_supervised.py --sets $NAME --clean $2 --output $S || { echo "[x2] FAILED train $2"; return 1; }
  $MODERN -u encode_dino_selftrain.py --student $S --split validation --output ${S}_validation || { echo "[x2] FAILED validation $2"; return 1; }
  $MODERN -u encode_dino_selftrain.py --student $S --split test --output ${S}_test || { echo "[x2] FAILED test $2"; return 1; }
  echo "[x2] student $2 done $(date)"
}
if [ $GA = $GB ]; then
  ( student $GA top2 ) > logs/x2_${NAME}_top2.log 2>&1
  ( student $GB none ) > logs/x2_${NAME}_none.log 2>&1
else
  ( student $GA top2 ) > logs/x2_${NAME}_top2.log 2>&1 &
  ( student $GB none ) > logs/x2_${NAME}_none.log 2>&1 &
  wait
fi
grep -h -E '^\{"epoch"|validation head accuracy|head scale' logs/x2_${NAME}_top2.log logs/x2_${NAME}_none.log | cut -c1-400
if grep -q FAILED logs/x2_${NAME}_top2.log logs/x2_${NAME}_none.log; then
  grep -h FAILED logs/x2_${NAME}_top2.log logs/x2_${NAME}_none.log
  exit 1
fi
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $MODERN -u x2_pilot_check.py --pilot $GEN \
  --students runs/x2_dino_${NAME}_top2 runs/x2_dino_${NAME}_none --output runs/x2_pilot_check_${NAME}.json 2>&1 \
  | grep -v Neighbours || echo "[x2] FAILED pilot check"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $MODERN -u x2_export.py --name s1_$NAME --tag $TAG \
  --base $BASE --base-public 0.98048 \
  --student runs/x2_dino_${NAME}_top2_test --check-student runs/x2_dino_${NAME}_none_test 2>&1 | grep -v Neighbours \
  || echo "[x2] FAILED export"
$MODERN audit_synth_style_submission.py runs/x2_s1_${NAME}_$TAG > /dev/null || echo "[x2] FAILED audit"
echo "[x2] done $NAME $(date)"
