#!/usr/bin/env bash
# Version 2 of run_d20_scale.sh (19 November 2026).
# Changes: (1) a generator that stops early is resumed: GPU 0 completes its shard below seed 900 before the
# snapshot, and the extra generation of each lane starts at seed 438, so gaps are filled before seeds 900-1049;
# (2) the snapshot wait has a deadline and then stops the train lane loudly; (3) every step leaves a marker,
# so a rerun skips finished steps and removes unfinished outputs; (4) the release check implements the gate
# written in EXPERIMENTS.md before the result exists (G0-G5).
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1

existing_gpu0=${1:?Pass the PID of the D18 GPU 0 generator.}
existing_gpu1=${2:?Pass the PID of the D18 GPU 1 generator.}
venvs=$HOME/recvis
python="$venvs/.venv-modern/bin/python"
gen_python="$venvs/.venv-gen/bin/python"
source_root=external_data/synth_style_gen/sd21_768
kernel=external_data/x1_kernel_sd21_768.json
end=800
snapshot=external_data/synth_style_gen/d20_train800_snapshot
set_name=sd21_768_train800
manifest=external_data/synth_style_$set_name
incumbent=runs/x2_dino_sd21_768_train400_top2
incumbent_file=runs/x2_s1_sd21_768_train400_20261118/submission.csv
student=runs/x2_dino_sd21_768_train800_top2
candidate=runs/x2_s1_sd21_768_train800_20261120
pilot=runs/x2_pilot_check_scale800_20261120.json
fresh=runs/x2_fresh_holdout_scale800_20261120.json
steps=logs/d20_steps
snapshot_deadline=${SNAPSHOT_DEADLINE:-2026-11-20 16:00}

mkdir -p logs "$steps"
exec 9>logs/d20_scale.lock
flock -n 9

log() { printf '[%s] %s\n' "$(TZ=Europe/Paris date -Is)" "$*"; }

step() {  # step NAME OUTPUT COMMAND...: skip when done, otherwise remove a partial output and run
    local name=$1 out=${2:?}
    shift 2
    if [[ -e $steps/$name.done ]]; then
        log "Step $name already done"
        return 0
    fi
    rm -rf -- "$out"
    log "Step $name starting"
    "$@"
    touch "$steps/$name.done"
    log "Step $name done"
}

wait_for_generator() {
    local gpu=$1 pid=$2 command
    log "GPU $gpu waiting for D18 generator PID $pid"
    while command=$(ps -p "$pid" -o args= -ww) &&
          [[ $command == *"generate_style_images.py"* && $command == *"--start 438 --end 900 --shard $gpu 2"* ]]; do
        sleep 45
    done
    log "GPU $gpu: D18 generator PID $pid has ended"
}

wait_for_idle_gpu() {
    local gpu=$1 busy
    busy=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits) || return
    while [[ -n $busy ]]; do
        sleep 45
        busy=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits) || return
    done
}

resume_shard() {  # completes shard $1 of seeds 438-899; returns at once when nothing is missing
    log "GPU $1 completing shard $1 of seeds 438-899 if anything is missing"
    CUDA_VISIBLE_DEVICES="$1" "$gen_python" -u generate_style_images.py --generator sd21_768 \
        --output "$source_root" --start 438 --end 900 --shard "$1" 2 --kernel-file "$kernel" \
        >> "logs/d20_resume$1.log" 2>&1
}

snapshot_records() {
    "$python" - "$source_root" "$snapshot" "$end" <<'PY'
import json
import shutil
import sys
from pathlib import Path

from generate_style_images import jobs

source, output, end = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
if (output / 'records_snapshot.jsonl').exists():
    print('Reusing the existing snapshot.', flush=True)
    sys.exit(0)
expected = {job['id'] for job in jobs('sd21_768', 0, end)}
records = {}
for index in sorted(source.glob('records_*.jsonl')):
    content = index.read_bytes()
    complete_lines = content[:content.rfind(b'\n') + 1].splitlines()
    for line in complete_lines:
        if line:
            row = json.loads(line)
            if row['id'] in expected:
                records[row['id']] = line
missing = expected - records.keys()
if missing:
    print(f'Waiting for {len(missing)} generated records below seed {end}.', flush=True)
    sys.exit(3)

partial = output.with_name(output.name + '.partial')
shutil.rmtree(partial, ignore_errors=True)
partial.mkdir()
for name in ('settings.json', 'hires', 'small', 'native'):
    path = source / name
    if path.exists():
        (partial / name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
with (partial / 'records_snapshot.jsonl').open('xb') as stream:
    for key in sorted(records):
        stream.write(records[key] + b'\n')
partial.rename(output)
print(f'Saved a fixed snapshot of {len(records)} generated records.', flush=True)
PY
}

generate_next() {
    local gpu=$1
    wait_for_idle_gpu "$gpu" || return
    log "GPU $gpu generating shard $gpu of seeds 438-1049 (missing images below 900 first)"
    CUDA_VISIBLE_DEVICES="$gpu" "$gen_python" -u generate_style_images.py \
        --generator sd21_768 --output "$source_root" --start 438 --end 1050 \
        --shard "$gpu" 2 --kernel-file "$kernel" >> "logs/d20_more$gpu.log" 2>&1 || return
    log "GPU $gpu additional generation finished"
}

release_check() {
    "$python" - "$pilot" "$fresh" "$candidate" "$incumbent" "$student" "$incumbent_file" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pilot, fresh_path, folder = map(Path, sys.argv[1:4])
incumbent, student, incumbent_file = sys.argv[4:7]
students = json.loads(pilot.read_text())['students']
fresh = json.loads(fresh_path.read_text())
audit = json.loads((folder / 'independent_audit.json').read_text())
metrics = json.loads((folder / 'metrics.json').read_text())

old, new = (students[name]['s1_rule']['accuracy'] for name in (incumbent, student))
rows_old, rows_new = (students[name]['kept_rows'] for name in (incumbent, student))
old_errors, new_errors = round(rows_old * (1 - old)), round(rows_new * (1 - new))


def held_out(name):
    with np.load(Path(name) / 'validation_logits.npz') as data:
        keep = data['top2'].astype(bool)
        right = data['logits'].argmax(1) == data['labels']
        return dict(zip(data['ids'][keep].tolist(), right[keep].tolist()))


a, b = held_out(incumbent), held_out(student)
shared = sorted(a.keys() & b.keys())
fix0 = sum(b[k] and not a[k] for k in shared)
err0 = sum(a[k] and not b[k] for k in shared)
fix1, err1 = fresh['top2']['fixes'], fresh['top2']['new_errors']
pooled = (fix0 + fix1 - err0 - err1) / max(fix0 + fix1 + err0 + err1, 1) ** .5
differ = int((pd.read_csv(folder / 'submission.csv')['label'].to_numpy()
              != pd.read_csv(incumbent_file)['label'].to_numpy()).sum())
gate = dict(
    G0_s1_rule_z_vs_reference_at_least_2=students[student]['s1_rule']['z'] >= 2,
    G1_audit_and_hash=bool(audit['passed']) and audit['submission_sha256'] == metrics['submission_sha256'],
    G2_fresh_400_437_more_fixes_than_new_errors=all(fresh[s]['fixes'] > fresh[s]['new_errors'] for s in ('all', 'top2')),
    G3_s1_rule_seeds_0_23_not_worse=new_errors <= old_errors,
    G4_pooled_student_z_at_least_1=pooled >= 1,
    G5_at_least_30_labels_differ=differ >= 30)
result = dict(
    incumbent_public=0.99066, gate=gate, evidence_supports_submission=all(gate.values()),
    s1_rule_seeds_0_23=dict(rows=[rows_old, rows_new], incumbent_errors=old_errors, candidate_errors=new_errors),
    student_alone_seeds_0_23=dict(rows=len(shared), fixes=fix0, new_errors=err0),
    fresh_holdout=fresh, pooled_student_z=pooled, labels_differ_from_incumbent_file=differ,
    public_noise_unit_points=100 * (0.3 * max(differ, 1)) ** .5 / 15323,
    submission_sha256=audit['submission_sha256'])
with (folder / 'release_check.json').open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print('RELEASE_CHECK', json.dumps(result), flush=True)
PY
}

(
    wait_for_generator 0 "$existing_gpu0"
    wait_for_idle_gpu 0
    resume_shard 0 || log "GPU 0 resume of shard 0 failed; the snapshot wait decides what happens next"
    deadline=$(TZ=Europe/Paris date -d "$snapshot_deadline" +%s)
    until snapshot_records; do
        status=$?
        if [[ $status != 3 ]]; then
            exit "$status"
        fi
        if (( $(date +%s) > deadline )); then
            log "Snapshot deadline ($snapshot_deadline) passed with records below seed $end missing; the train lane stops"
            exit 4
        fi
        sleep 60
    done

    export CUDA_VISIBLE_DEVICES=0
    log "GPU 0 building the 104,000-image manifest and applying the D18 label filter"
    step manifest "$manifest" "$python" -u prepare_synth_manifest.py build --gen-roots "$snapshot" \
        --kernel-file "$kernel" --role train --max-index "$end" --expected "$((130 * end))" --output "$manifest"
    step judge "runs/synth_${set_name}_judge" "$python" -u judge_synth.py --root "$manifest" \
        --output "runs/synth_${set_name}_judge"

    log "GPU 0 training DINOv2-g at 224px, holding out seeds 0-23 and 400-437"
    step train "$student" "$python" -u train_synth_supervised.py --sets "$set_name" --clean top2 --output "$student" \
        --val-seeds 24 --max-train-index "$end" --exclude-train-range 400 438 \
        --image-size 224 --epochs 3 --batch-size 32
    step fresh "$fresh" "$python" -u compare_synth_students.py --set sd21_768_train438 --seed-range 400 438 \
        --incumbent "$incumbent" --candidate "$student" --output "$fresh"
    step encode_val "${student}_validation" "$python" -u encode_dino_selftrain.py --student "$student" \
        --split validation --output "${student}_validation"
    step encode_test "${student}_test" "$python" -u encode_dino_selftrain.py --student "$student" \
        --split test --output "${student}_test"

    export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
    log "GPU 0 inference finished; scoring and auditing the CSV"
    step pilot "$pilot" "$python" -u x2_pilot_check.py --pilot sd21_768 --students "$incumbent" "$student" --output "$pilot"
    step export "$candidate" "$python" -u x2_export.py --name s1_sd21_768_train800 --tag 20261120 \
        --base runs/adapted_bag3_graph_20261116 --base-public 0.98048 --student "${student}_test"
    step audit "$candidate/independent_audit.json" "$python" -u audit_synth_style_submission.py "$candidate"
    if [[ ! -e $candidate/release_check.json ]]; then
        release_check
    fi
    log "CSV and release check ready at $candidate"
) &
train_lane=$!

(
    wait_for_generator 1 "$existing_gpu1"
    generate_next 1
) &
generation_lane=$!

train_status=0
wait "$train_lane" || train_status=$?
if (( train_status != 0 )); then
    log "Training or export failed (exit code $train_status); see this log before using any partial result"
fi
gen_status=0
generate_next 0 || gen_status=$?
wait "$generation_lane" || gen_status=$?
if (( train_status != 0 )); then
    exit "$train_status"
fi
exit "$gen_status"
