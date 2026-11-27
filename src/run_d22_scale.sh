#!/usr/bin/env bash
# Continue the 0.99223 supervised recipe with the larger generated training set.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1

modern=$HOME/recvis/.venv-modern/bin/python
siglip=$HOME/recvis/.venv-siglip2/bin/python
source=external_data/synth_style_gen/sd21_768
kernel=external_data/x1_kernel_sd21_768.json
base_run=runs/adapted_bag3_graph_20261116
previous_set=sd21_768_train800
previous_root=external_data/synth_style_$previous_set
extra_set=sd21_768_800_1349_d22
extra_root=external_data/synth_style_$extra_set
fresh_root=external_data/synth_style_s1_fresh_1350_1387_20261122
work=runs/d22_scale1350
fresh_work=runs/s1_fresh_1350_1387_20261122
incumbent=runs/x2_dino_sd21_768_train800_top2
incumbent_file=runs/x2_s1_sd21_768_train800_20261120
student=runs/x2_dino_sd21_768_train1350_top2
candidate=runs/x2_s1_sd21_768_train1350_20261122
assessment=("$fresh_root" "$fresh_work" "$incumbent" "$incumbent_file" "$student" "$candidate")

exec 9>logs/d22_scale.lock
flock -n 9
mkdir "$work"

log() { printf '[%s] %s\n' "$(TZ=Europe/Paris date -Is)" "$*"; }

wait_gpu() {
    local gpu=$1
    log "GPU $gpu waiting for its generation lane"
    while tmux has-session -t "recvis_d22_gen$gpu" 2>/dev/null; do sleep 30; done
    local busy
    while true; do
        busy=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
        [[ -z $busy ]] && break
        sleep 30
    done
}

snapshot() {
    "$modern" - "$source" "$@" <<'PY'
import json
import sys
from pathlib import Path

from generate_style_images import jobs

source, output = map(Path, sys.argv[1:3])
start, stop, omit_start, omit_stop = map(int, sys.argv[3:])
expected = {r['id'] for r in jobs('sd21_768', start, stop)
            if not omit_start <= r['index'] < omit_stop}
records = {}
for path in sorted(source.glob('records_*.jsonl')):
    raw = path.read_bytes()
    for line in raw[:raw.rfind(b'\n') + 1].splitlines():
        if line:
            record = json.loads(line)
            if record['id'] in expected:
                records[record['id']] = line
if len(records) != len(expected):
    print(f'Generation has {len(records)} of {len(expected)} required records for {output.name}', flush=True)
    sys.exit(3)

output.mkdir()
for name in ('hires', 'small', 'settings.json'):
    path = source / name
    assert path.exists()
    (output / name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
(output / 'records_fixed.jsonl').write_bytes(b'\n'.join(records[key] for key in sorted(records)) + b'\n')
print(f'Fixed snapshot contains {len(records)} records for {output.name}', flush=True)
PY
}

wait_for_file() {
    local path=$1 producer=$2
    until [[ -e $path ]]; do
        if ! kill -0 "$producer" 2>/dev/null; then
            [[ -e $path ]] && return 0
            log "Training process ended before writing $path"
            return 1
        fi
        sleep 30
    done
}

evaluate() {
    local py=$1 stage=$2
    "$py" -u check_s1_fresh_20261120.py "$stage" --root "${assessment[0]}" --work "${assessment[1]}" \
        --seed-range 1350 1388 --incumbent-public .99223 \
        --incumbent-student "${assessment[2]}" --incumbent-submission "${assessment[3]}" \
        --candidate-student "${assessment[4]}" --candidate-submission "${assessment[5]}"
}

(
    wait_gpu 0
    export CUDA_VISIBLE_DEVICES=0
    until snapshot "$work/train_snapshot" 800 1350 900 938; do
        rc=$?
        [[ $rc == 3 ]] || exit "$rc"
        if ! tmux has-session -t recvis_d22_gen1 2>/dev/null; then
            log 'Missing training records after both generators exited.'
            exit 1
        fi
        sleep 60
    done
    log 'Building and judging the new training rows on GPU 0'
    "$modern" -u prepare_synth_manifest.py build --gen-roots "$work/train_snapshot" \
        --kernel-file "$kernel" --role train --expected 66560 --output "$extra_root" \
        --exclude-manifest "$previous_root/manifest.json"
    touch "$work/train_manifest_ready"
    "$modern" -u judge_synth.py --root "$extra_root" --output "runs/synth_${extra_set}_judge"

    log 'Training one 224 px DINOv2-g student on both manifests'
    "$modern" -u train_synth_supervised.py --sets "$previous_set" "$extra_set" \
        --clean top2 --output "$student" --val-seeds 24 --max-train-index 1350 \
        --exclude-train-range 400 438 --image-size 224 --epochs 3 --batch-size 32
    "$modern" -u encode_dino_selftrain.py --student "$student" --split validation --output "${student}_validation"
    "$modern" -u encode_dino_selftrain.py --student "$student" --split test --output "${student}_test"

    export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
    "$modern" -u x2_pilot_check.py --pilot sd21_768 --students "$incumbent" "$student" --output "$work/pilot.json"
    "$modern" -u x2_export.py --name s1_sd21_768_train1350 --tag 20261122 --base "$base_run" \
        --base-public .98048 --threshold .9 --weight .5 --student "${student}_test"
    "$modern" -u audit_synth_style_submission.py "$candidate"
    touch "$work/candidate_ready"
    log "Audited CSV ready at $candidate"
) >> logs/d22_train.log 2>&1 &
train_pid=$!

(
    wait_gpu 1
    export CUDA_VISIBLE_DEVICES=1
    wait_for_file "$work/train_manifest_ready" "$train_pid"
    mkdir "$fresh_work"
    snapshot "$fresh_work/snapshot" 1350 1388 0 0
    log 'Preparing the fresh 4,940-image holdout on GPU 1'
    "$modern" -u prepare_synth_manifest.py build --gen-roots "$fresh_work/snapshot" \
        --kernel-file "$kernel" --role confirmation --expected 4940 --output "$fresh_root" \
        --exclude-manifest "$previous_root/manifest.json" --exclude-manifest "$extra_root/manifest.json" \
        --exclude-manifest external_data/synth_style_s1_fresh_900_937_20261120/manifest.json
    "$modern" -u judge_synth.py --root "$fresh_root" --output "$fresh_work/judge"
    "$modern" -u encode_domainnet.py --root "$fresh_root" --base runs/pe_bigg_imagenet80_fp16 \
        --original-only --output "$fresh_work/frozen_bigg"
    "$siglip" -u encode_domainnet.py --root "$fresh_root" --base runs/siglip2_gopt_imagenet80_fp16 \
        --original-only --output "$fresh_work/frozen_siglip"
    evaluate "$siglip" siglip
    evaluate "$modern" bigg

    log 'Adapted base ready on the fresh holdout; waiting for the candidate'
    wait_for_file "$work/candidate_ready" "$train_pid"
    evaluate "$modern" dino
    export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
    evaluate "$modern" score
    log "Finished offline comparisons at $fresh_work/result.json"
) >> logs/d22_fresh.log 2>&1 &
fresh_pid=$!

train_status=0
fresh_status=0
wait "$train_pid" || train_status=$?
wait "$fresh_pid" || fresh_status=$?
log "Training exit $train_status; fresh comparison exit $fresh_status"
(( train_status == 0 && fresh_status == 0 ))
