#!/usr/bin/env bash
# 23 November 2026, final round. Best public file: 0.99321 (S1 rule with one DINOv2-g student trained on
# 161,656 own labelled images). The deadline is 24 November 23:59:59 Paris time.
# Generation is stopped. Seeds below N are complete. Four students with the unchanged recipe (224px, LoRA
# rank 8, cosine head, 3 epochs, batch 32) are trained on all labelled images except the held-out seeds
# (0-23, 400-437, 900-937, 1350-1387): a and c on GPU 0, b and d on GPU 1, with different seeds.
# After each round the S1 files are built (single student and geometric-mean ensembles), audited, and
# scored on the two fresh holdouts against the 161,656-image student.
# A rerun skips finished steps.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1

N=${1:?Pass the first seed index that is not complete (training uses seeds below it).}
modern=$HOME/recvis/.venv-modern/bin/python
source=external_data/synth_style_gen/sd21_768
kernel=external_data/x1_kernel_sd21_768.json
base_run=runs/adapted_bag3_graph_20261116
set1=sd21_768_train800
set2=sd21_768_800_1349_d22
set3=sd21_768_1388_${N}_d23
root3=external_data/synth_style_$set3
incumbent=runs/x2_dino_sd21_768_train1350_top2
fresh1_root=external_data/synth_style_s1_fresh_1350_1387_20261122
fresh1_work=runs/s1_fresh_1350_1387_20261122
fresh2_root=external_data/synth_style_s1_fresh_900_937_20261120
fresh2_work=runs/s1_fresh_900_937_20261120
work=runs/d23_final
steps=$work/steps
tag=20261123

mkdir -p "$steps" logs
exec 9>logs/d23_final.lock
flock -n 9

log() { printf '[%s] %s\n' "$(TZ=Europe/Paris date -Is)" "$*"; }
student_dir() { printf 'runs/x2_dino_sd21_768_final%s_%s' "$N" "$1"; }

step() {  # step NAME OUTPUT COMMAND...: skip when done, otherwise remove a partial output and run
    local name=$1 out=${2:?}
    shift 2
    if [[ -e $steps/$name.done ]]; then
        log "Step $name already done"
        return 0
    fi
    rm -rf -- "$out"
    log "Step $name starting"
    "$@" || { log "Step $name FAILED"; return 1; }
    touch "$steps/$name.done"
    log "Step $name done"
}

snapshot() {
    "$modern" - "$source" "$work/train_snapshot" 1388 "$N" <<'PY'
import json
import sys
from pathlib import Path

from generate_style_images import jobs

source, output = map(Path, sys.argv[1:3])
start, stop = map(int, sys.argv[3:])
expected = {r['id'] for r in jobs('sd21_768', start, stop)}
records = {}
for path in sorted(source.glob('records_*.jsonl')):
    raw = path.read_bytes()
    for line in raw[:raw.rfind(b'\n') + 1].splitlines():
        if line:
            record = json.loads(line)
            if record['id'] in expected:
                records[record['id']] = line
assert len(records) == len(expected), f'{len(records)} of {len(expected)} records below seed {stop}'
output.mkdir()
for name in ('hires', 'small', 'settings.json'):
    path = source / name
    assert path.exists()
    (output / name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
(output / 'records_fixed.jsonl').write_bytes(b'\n'.join(records[key] for key in sorted(records)) + b'\n')
print(f'Fixed snapshot contains {len(records)} records', flush=True)
PY
}

fresh_encode() {  # NAME=FOLDER on both fresh holdouts (GPU of the calling lane)
    "$modern" -u x4_fresh_ensemble.py encode --root "$fresh1_root" --work "$fresh1_work" --student "$1" \
        && "$modern" -u x4_fresh_ensemble.py encode --root "$fresh2_root" --work "$fresh2_work" --student "$1"
}

student() {  # NAME SEED, on the GPU of the calling lane
    local name=$1 seed=$2 s
    s=$(student_dir "$name")
    step "train_$name" "$s" "$modern" -u train_synth_supervised.py --sets "$set1" "$set2" "$set3" \
            --clean top2 --output "$s" --val-seeds 24 --max-train-index "$N" --exclude-train-range 400 438 \
            --image-size 224 --epochs 3 --batch-size 32 --seed "$seed" \
        && step "validation_$name" "${s}_validation" "$modern" -u encode_dino_selftrain.py --student "$s" \
            --split validation --output "${s}_validation" \
        && step "test_$name" "${s}_test" "$modern" -u encode_dino_selftrain.py --student "$s" \
            --split test --output "${s}_test" \
        && step "fresh_$name" "$work/fresh_$name.unused" fresh_encode "fin_$name=$s"
}

export_file() {  # FILE_NAME STUDENT_NAME...
    local file=$1 flags=() name
    shift
    for name in "$@"; do flags+=(--student "$(student_dir "$name")_test"); done
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 step "export_$file" "runs/x2_${file}_$tag" "$modern" -u x2_export.py \
            --name "$file" --tag "$tag" --base "$base_run" --base-public .98048 --threshold .9 --weight .5 "${flags[@]}" \
        && CUDA_VISIBLE_DEVICES= step "audit_$file" "runs/x2_${file}_$tag/independent_audit.json" \
            "$modern" -u audit_synth_style_submission.py "runs/x2_${file}_$tag"
}

score_fresh() {  # TAG COMBO...
    local t=$1
    shift
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 step "score1_$t" "$fresh1_work/x4_result_$t.json" "$modern" -u x4_fresh_ensemble.py score \
        --root "$fresh1_root" --work "$fresh1_work" --saved d1350=candidate --combos d1350 "$@" --incumbent d1350 --tag "$t" || true
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 step "score2_$t" "$fresh2_work/x4_result_$t.json" "$modern" -u x4_fresh_ensemble.py score \
        --root "$fresh2_root" --work "$fresh2_work" --combos d1350 "$@" --incumbent d1350 --tag "$t" || true
}

log "Final round, training on seeds below $N"
sha256sum run_d23_final.sh train_synth_supervised.py encode_dino_selftrain.py x2_export.py x4_fresh_ensemble.py \
    check_s1_fresh_20261120.py prepare_synth_manifest.py judge_synth.py audit_synth_style_submission.py

export CUDA_VISIBLE_DEVICES=0
step snapshot "$work/train_snapshot" snapshot
step manifest "$root3" "$modern" -u prepare_synth_manifest.py build --gen-roots "$work/train_snapshot" \
    --kernel-file "$kernel" --role train --expected "$((130 * (N - 1388)))" --output "$root3" \
    --exclude-manifest "external_data/synth_style_$set1/manifest.json" \
    --exclude-manifest "external_data/synth_style_$set2/manifest.json" \
    --exclude-manifest "$fresh1_root/manifest.json" --exclude-manifest "$fresh2_root/manifest.json"
step judge "runs/synth_${set3}_judge" "$modern" -u judge_synth.py --root "$root3" --output "runs/synth_${set3}_judge"
# The current best student on the second fresh holdout (it is already scored on the first one).
step fresh_incumbent "$work/fresh_incumbent.unused" "$modern" -u x4_fresh_ensemble.py encode \
    --root "$fresh2_root" --work "$fresh2_work" --student "d1350=$incumbent"

( export CUDA_VISIBLE_DEVICES=0; student a 20261117 || log 'student a did not finish' ) >> logs/d23_lane0.log 2>&1 &
lane0=$!
( export CUDA_VISIBLE_DEVICES=1; student b 20261123 || log 'student b did not finish' ) >> logs/d23_lane1.log 2>&1 &
lane1=$!
wait "$lane0" || true
# GPU 0 is free first: it starts the third student at once; the single-student file is built on CPU.
( export CUDA_VISIBLE_DEVICES=0; student c 20261124 || log 'student c did not finish' ) >> logs/d23_lane0.log 2>&1 &
lane0=$!
if [[ -e $steps/test_a.done ]]; then
    export_file s1_final_a a || true
    score_fresh round1a fin_a
fi
wait "$lane1" || true
( export CUDA_VISIBLE_DEVICES=1; student d 20261125 || log 'student d did not finish' ) >> logs/d23_lane1.log 2>&1 &
lane1=$!
if [[ -e $steps/test_a.done && -e $steps/test_b.done ]]; then
    export_file s1_final_ab a b || true
    score_fresh round1ab fin_a fin_b fin_a+fin_b fin_a+fin_b+d1350
fi
wait "$lane0" || true
wait "$lane1" || true
if [[ -e $steps/test_c.done && -e $steps/test_d.done && -e $steps/test_a.done && -e $steps/test_b.done ]]; then
    export_file s1_final_abcd a b c d || true
    score_fresh round2 fin_a+fin_b fin_a+fin_b+fin_c+fin_d fin_c fin_d
elif [[ -e $steps/test_c.done && -e $steps/test_a.done && -e $steps/test_b.done ]]; then
    export_file s1_final_abc a b c || true
    score_fresh round2 fin_a+fin_b fin_a+fin_b+fin_c fin_c
fi
log 'Final round finished'
