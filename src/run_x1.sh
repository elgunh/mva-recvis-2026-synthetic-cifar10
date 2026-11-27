#!/bin/bash
# X1 of the 16 November 2026 plan (EXPERIMENTS.md): own pilots with SD-Turbo, SD 2.1-base and SD 2.1 (768),
# 13 published templates x 10 classes, scored by the frozen reference pipeline (synth_signature.py).
# Each GPU starts only after split C has finished on it.
set -u
cd $HOME/recvis/A3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
GEN=$HOME/recvis/.venv-gen/bin/python
MODERN=$HOME/recvis/.venv-modern/bin/python
SIGLIP=$HOME/recvis/.venv-siglip2/bin/python
GENROOT=external_data/synth_style_gen

wait_gpu() {  # $1 gpu, $2 log of split C on that gpu
  until grep -q -E 'finished bigg C|FAILED' $2; do sleep 60; done
  for TRY in $(seq 1 40); do
    if CUDA_VISIBLE_DEVICES=$1 $MODERN -c "import torch; torch.zeros(1, device='cuda')" 2> /dev/null; then return 0; fi
    sleep 15
  done
  return 1
}
generate() {  # $1 generator, $2 first seed index, $3 end, [$4 shard, $5 shards], [$6 = no: skip native files]
  FLAG=--save-native
  if [ "${6:-yes}" = no ]; then FLAG=; fi
  echo "[x1] generate $1 $2-$3 shard ${4:-0}/${5:-1} $(date)"
  $GEN -u generate_style_images.py --generator $1 --output $GENROOT/$1 --start $2 --end $3 \
    --shard ${4:-0} ${5:-1} $FLAG || { echo "[x1] FAILED generate $1 $2-$3"; return 1; }
}
score() {  # $1 generator, $2 set name, $3 number of seeds per cell
  NAME=$2
  ROOT=external_data/synth_style_$NAME
  KERNEL=external_data/x1_kernel_$1.json
  echo "[x1] score $NAME $(date)"
  if [ ! -f $KERNEL ]; then
    CUDA_VISIBLE_DEVICES= $MODERN -u prepare_synth_manifest.py kernel --gen-roots $GENROOT/$1 --pilot-end 8 \
      --sources hires native --output $KERNEL || { echo "[x1] FAILED kernel $1"; return 1; }
  fi
  $MODERN -u prepare_synth_manifest.py build --gen-roots $GENROOT/$1 --kernel-file $KERNEL --role pilot \
    --max-index $3 --expected $((130*$3)) --output $ROOT || { echo "[x1] FAILED manifest $NAME"; return 1; }
  $MODERN -u judge_synth.py --root $ROOT --output runs/synth_${NAME}_judge || { echo "[x1] FAILED judge $NAME"; return 1; }
  $MODERN -u encode_domainnet.py --root $ROOT --base runs/pe_bigg_imagenet80_fp16 \
    --output runs/synth_${NAME}_bigg || { echo "[x1] FAILED bigG $NAME"; return 1; }
  $SIGLIP -u encode_domainnet.py --root $ROOT --base runs/siglip2_gopt_imagenet80_fp16 \
    --output runs/synth_${NAME}_siglip2 || { echo "[x1] FAILED SigLIP2 $NAME"; return 1; }
  $MODERN -u encode_dino_giant.py --external-root $ROOT --external-runs runs/synth_${NAME}_bigg runs/synth_${NAME}_siglip2 \
    --output runs/synth_${NAME}_dinov2_g || { echo "[x1] FAILED DINOv2 $NAME"; return 1; }
  $MODERN -u synth_signature.py --name $NAME --mixing-index $3 || { echo "[x1] FAILED signature $NAME"; return 1; }
  echo "[x1] scored $NAME $(date)"
}
chain0() {
  export CUDA_VISIBLE_DEVICES=0
  wait_gpu 0 logs/cd_C_gpu0.log || { echo "[x1] FAILED GPU 0 not ready"; return 1; }
  echo "[x1] chain 0 start $(date)"
  generate sd_turbo 0 24 && score sd_turbo sd_turbo 24
  generate sd21_768 0 8 && score sd21_768 sd21_768_p8 8
  generate sd21_768 8 24 0 2
  echo "[x1] chain 0 done $(date)"
}
chain1() {
  export CUDA_VISIBLE_DEVICES=1
  wait_gpu 1 logs/cd_C_gpu1.log || { echo "[x1] FAILED GPU 1 not ready"; return 1; }
  echo "[x1] chain 1 start $(date)"
  generate sd21_base 0 24 && score sd21_base sd21_base 24
  generate sd21_768 8 24 1 2
  echo "[x1] chain 1 done $(date)"
}
echo "[x1] queued $(date)"
sha256sum run_x1.sh generate_style_images.py prepare_synth_manifest.py synth_signature.py x1_pick.py
chain0 > logs/x1_gpu0.log 2>&1 &
chain1 > logs/x1_gpu1.log 2>&1 &
wait
( export CUDA_VISIBLE_DEVICES=0; score sd21_768 sd21_768 24 ) > logs/x1_768_full.log 2>&1
echo "[x1] pilots done $(date)"
grep -h -E 'FAILED|X1_SIGNATURE' logs/x1_gpu0.log logs/x1_gpu1.log logs/x1_768_full.log | cut -c1-1500

# Fallback so that the GPUs are not idle overnight: more seeds of the best-ranked recipe.
# This is data for X2 only; it does not decide anything and can be stopped at any time.
PICK=$($MODERN x1_pick.py)
echo "[x1] fallback generation: $PICK"
set -- $PICK
if [ $# -eq 3 ]; then
  ( export CUDA_VISIBLE_DEVICES=0; generate $1 24 $2 0 2 $3 ) > logs/x1_more_gpu0.log 2>&1 &
  ( export CUDA_VISIBLE_DEVICES=1; generate $1 24 $2 1 2 $3 ) > logs/x1_more_gpu1.log 2>&1 &
  wait
fi
echo "[x1] done $(date)"
