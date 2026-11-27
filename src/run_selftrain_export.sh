#!/bin/bash
# Builds the three candidates from the cross-fitted students as soon as both
# halves of the test set are encoded. Runs on the server under nohup, on CPU,
# so it keeps running after I log out.
set -u
cd $HOME/recvis/A3
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
MODERN=$HOME/recvis/.venv-modern/bin/python
A=runs/selftrain_bigg_fold0_test
B=runs/selftrain_bigg_fold1_test
echo "[export] waiting $(date)"
for TRY in $(seq 1 1440); do
  if [ -f $A/metrics.json ] && [ -f $B/metrics.json ]; then break; fi
  sleep 30
done
if [ ! -f $A/metrics.json ] || [ ! -f $B/metrics.json ]; then
  echo "[export] FAILED encodes missing $(date)"
  exit 1
fi
VAL=""
for V in runs/selftrain_bigg_fold0_validation runs/selftrain_bigg_fold1_validation; do
  if [ -f $V/metrics.json ]; then VAL="$VAL $V"; fi
done
TAG=$(date +%Y%m%d)
echo "[export] start $(date) validation:$VAL"
CUDA_VISIBLE_DEVICES= $MODERN -u selftrain_export.py --name crossfit --encodes $A $B --validation $VAL --tag $TAG \
  > logs/selftrain_export_python.log 2>&1 || {
  echo "[export] FAILED export $(date)"
  tail -n 6 logs/selftrain_export_python.log
  exit 1
}
grep -v Neighbours logs/selftrain_export_python.log | tail -n 8
for V in prior graph half; do
  $MODERN audit_synth_style_submission.py runs/adapted_crossfit_${V}_$TAG | grep -E '"passed"|changed_from_reference' \
    || echo "[export] FAILED audit $V"
done
echo "[export] done $(date) tag $TAG"
