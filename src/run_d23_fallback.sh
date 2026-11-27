#!/usr/bin/env bash
# Safety net for run_d23_final.sh (23 November). It waits until the
# final-round script has ended. Then it (1) scores a planned file on the fresh holdouts if that score is
# missing, and (2) if the planned ensemble of round 2 does not exist, builds one file from all students
# that finished their test encode (s1_final_best), audits and scores it.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
N=1642
modern=$HOME/recvis/.venv-modern/bin/python
base_run=runs/adapted_bag3_graph_20261116
fresh1_root=external_data/synth_style_s1_fresh_1350_1387_20261122
fresh1_work=runs/s1_fresh_1350_1387_20261122
fresh2_root=external_data/synth_style_s1_fresh_900_937_20261120
fresh2_work=runs/s1_fresh_900_937_20261120
work=runs/d23_final
steps=$work/steps
tag=20261123
log() { printf '[%s] %s\n' "$(TZ=Europe/Paris date -Is)" "$*"; }
student_dir() { printf 'runs/x2_dino_sd21_768_final%s_%s' "$N" "$1"; }

exec 8>logs/d23_fallback.lock
flock -n 8 || { echo 'another fallback instance is running'; exit 0; }
log 'fallback waiting for the final-round script'
until flock -n logs/d23_final.lock true; do sleep 120; done
log 'final-round script has ended'

score() {  # TAG COMBO...: on both fresh holdouts, only where the result is missing
    local t=$1
    shift
    if [[ ! -e $fresh1_work/x4_result_$t.json ]]; then
        CUDA_VISIBLE_DEVICES= "$modern" -u x4_fresh_ensemble.py score --root "$fresh1_root" --work "$fresh1_work" \
            --saved d1350=candidate --combos d1350 "$@" --incumbent d1350 --tag "$t" || log "score $t on fresh 1350 FAILED"
    fi
    if [[ ! -e $fresh2_work/x4_result_$t.json ]]; then
        CUDA_VISIBLE_DEVICES= "$modern" -u x4_fresh_ensemble.py score --root "$fresh2_root" --work "$fresh2_work" \
            --combos d1350 "$@" --incumbent d1350 --tag "$t" || log "score $t on fresh 900 FAILED"
    fi
}

available=()
for s in a b c d; do
    if [[ -e $steps/test_$s.done ]]; then
        available+=("$s")
        # the fresh encodes of a finished student, if that step failed (GPU 0 is free now; existing files are kept)
        for pair in "$fresh1_root $fresh1_work" "$fresh2_root $fresh2_work"; do
            set -- $pair
            if [[ ! -e $2/x4_student_fin_$s.npz ]]; then
                CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 "$modern" -u x4_fresh_ensemble.py encode --root "$1" --work "$2" \
                    --student "fin_$s=$(student_dir "$s")" || log "fresh encode of student $s FAILED"
            fi
        done
    fi
done
log "students with a test encode: ${available[*]:-none}"

# (1) planned files that exist but have no fresh score
[[ -e runs/x2_s1_final_a_$tag/independent_audit.json ]] && score round1a fin_a
[[ -e runs/x2_s1_final_ab_$tag/independent_audit.json ]] && score round1ab fin_a fin_b fin_a+fin_b
[[ -e runs/x2_s1_final_abcd_$tag/independent_audit.json ]] && score round2 fin_a+fin_b+fin_c+fin_d
[[ -e runs/x2_s1_final_abc_$tag/independent_audit.json ]] && score round2 fin_a+fin_b+fin_c

# (2) one file from all finished students, unless exactly that file already exists
joined=$(IFS=; echo "${available[*]:-}")
if [[ -z $joined ]]; then
    log 'no student finished; nothing to build'
elif [[ -e runs/x2_s1_final_${joined}_$tag/independent_audit.json ]]; then
    log "the planned file s1_final_$joined exists; nothing to build"
else
    flags=()
    combo=
    for s in "${available[@]}"; do
        flags+=(--student "$(student_dir "$s")_test")
        combo+="${combo:++}fin_$s"
    done
    out=runs/x2_s1_final_best_$tag
    rm -rf -- "$out"
    log "building s1_final_best from students ${available[*]}"
    if CUDA_VISIBLE_DEVICES= "$modern" -u x2_export.py --name s1_final_best --tag "$tag" --base "$base_run" \
            --base-public .98048 --threshold .9 --weight .5 "${flags[@]}" \
        && CUDA_VISIBLE_DEVICES= "$modern" -u audit_synth_style_submission.py "$out" > /dev/null; then
        rm -f "$fresh1_work/x4_result_fallback.json" "$fresh2_work/x4_result_fallback.json"
        score fallback "$combo"
        printf '%s\n' "$combo" > "$work/fallback_combo.txt"
        log "s1_final_best ready with $combo"
    else
        log 's1_final_best FAILED'
    fi
fi
log 'fallback finished'
