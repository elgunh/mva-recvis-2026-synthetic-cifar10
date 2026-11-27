#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

gpu=${1:?Pass GPU 0 or 1.}
[[ $gpu == 0 || $gpu == 1 ]]
exec 9>"logs/d22_gen$gpu.lock"
flock -n 9
busy=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z $busy ]] || { printf 'GPU %s is occupied: %s\n' "$gpu" "$busy" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
printf '[%s] GPU %s starting seeds 1050-1399\n' "$(TZ=Europe/Paris date -Is)" "$gpu"
$HOME/recvis/.venv-gen/bin/python -u generate_style_images.py \
    --generator sd21_768 --output external_data/synth_style_gen/sd21_768 \
    --start 1050 --end 1400 --shard "$gpu" 2 \
    --kernel-file external_data/x1_kernel_sd21_768.json --reserve-disk-bytes 2147483648
printf '[%s] GPU %s generation complete\n' "$(TZ=Europe/Paris date -Is)" "$gpu"
