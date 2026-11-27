#!/usr/bin/env bash
# Compare the two completed S1 rules on unseen generation seeds 900-937.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1

modern=$HOME/recvis/.venv-modern/bin/python
siglip=$HOME/recvis/.venv-siglip2/bin/python
work=runs/s1_fresh_900_937_20261120
root=external_data/synth_style_s1_fresh_900_937_20261120
source=external_data/synth_style_gen/sd21_768
export CUDA_VISIBLE_DEVICES=0

log() { printf '[%s] %s\n' "$(TZ=Europe/Paris date -Is)" "$*"; }

if [[ ${1:-} == prep ]]; then
    mkdir "$work"
    log "Snapshotting the 4,940 rows fixed before scoring"
    "$modern" - "$source" "$work/snapshot" <<'PY'
import json
import sys
from pathlib import Path

from generate_style_images import jobs

source, output = map(Path, sys.argv[1:])
expected = {job['id'] for job in jobs('sd21_768', 900, 938)}
rows = {}
for path in sorted(source.glob('records_*.jsonl')):
    raw = path.read_bytes()
    for line in raw[:raw.rfind(b'\n') + 1].splitlines():
        if line:
            record = json.loads(line)
            if record['id'] in expected:
                rows[record['id']] = line
if set(rows) != expected:
    raise RuntimeError(f'Holdout incomplete: {len(rows)} of {len(expected)} records')
output.mkdir()
for name in ('hires', 'small', 'settings.json'):
    path = source / name
    assert path.exists() and not (output / name).exists()
    (output / name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
(output / 'records_fixed.jsonl').write_bytes(b'\n'.join(rows[key] for key in sorted(rows)) + b'\n')
PY
    "$modern" -u prepare_synth_manifest.py build --gen-roots "$work/snapshot" --output "$root" \
        --role confirmation --expected 4940 --kernel-file external_data/x1_kernel_sd21_768.json
    "$modern" - "$root/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

fresh = json.loads(Path(sys.argv[1]).read_text())['records']
previous = json.loads(Path('external_data/synth_style_sd21_768_train800/manifest.json').read_text())['records']
assert {r['index'] for r in fresh} == set(range(900, 938))
assert not ({r['pixels32_sha256'] for r in fresh} & {r['pixels32_sha256'] for r in previous}), 'Fresh pixels overlap previous data.'
print(f'{len(fresh)} fresh unique rows; no exact pixel overlap with the training manifest.', flush=True)
PY
    "$modern" -u judge_synth.py --root "$root" --output "$work/judge"
    log "Frozen bigG original views for reflection routing"
    "$modern" -u encode_domainnet.py --root "$root" --base runs/pe_bigg_imagenet80_fp16 \
        --original-only --output "$work/frozen_bigg"
    log "Frozen SigLIP2 original views for reflection routing"
    "$siglip" -u encode_domainnet.py --root "$root" --base runs/siglip2_gopt_imagenet80_fp16 \
        --original-only --output "$work/frozen_siglip"
    touch "$work/prep_done"
    log "Fresh inputs ready"
elif [[ ${1:-} == score ]]; then
    test -f "$work/prep_done"
    for encoder in dino siglip bigg; do
        log "Encoding $encoder for the adapted base"
        py=$modern
        if [[ $encoder == siglip ]]; then py=$siglip; fi
        "$py" -u check_s1_fresh_20261120.py "$encoder" --root "$root" --work "$work"
    done
    log "Rebuilding the adapted graph and scoring both fixed rules"
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$modern" -u \
        check_s1_fresh_20261120.py score --root "$root" --work "$work"
    log "Paired results ready in $work/result.json"
else
    printf 'Usage: %s prep|score\n' "$0" >&2
    exit 2
fi
