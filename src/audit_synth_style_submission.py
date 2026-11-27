"""Format and consistency check of a prepared submission. It recomputes nothing from the models."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_SHA256 = '16ded85df3246c46579c9ca717480a203b9dac25fd946558053054fb927a0045'
REFERENCE_SHA256 = '7de5fa73b11573189f3ae44385ec6c4ee3736e5f915890a170f156320d7e553a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--sample', type=Path, default=Path('data/sample_submission.csv'))
    parser.add_argument('--reference', type=Path, default=Path('runs/dinov2_giant_candidate'))
    args = parser.parse_args()
    assert sha(args.sample) == SAMPLE_SHA256, 'Not the canonical competition sample.'
    assert sha(args.reference/'submission.csv') == REFERENCE_SHA256, 'Reference CSV changed.'
    sample = pd.read_csv(args.sample)
    csv = args.root/'submission.csv'
    frame = pd.read_csv(csv)
    reference = pd.read_csv(args.reference/'submission.csv')
    checks = dict(
        columns_exact=list(frame.columns) == ['id', 'label'],
        rows_51077=len(frame) == 51077 == len(sample),
        ids_equal_sample_in_order=bool(np.array_equal(frame['id'].to_numpy(), sample['id'].to_numpy())),
        ids_unique=bool(frame['id'].is_unique),
        labels_integer=bool(pd.api.types.is_integer_dtype(frame['label'])),
        labels_in_0_9=bool(frame['label'].between(0, 9).all()),
        no_missing=bool(not frame.isna().any().any()))
    with np.load(args.root/'test_predictions.npz') as saved:
        probabilities = saved['probabilities']
        checks.update(
            no_labels_in_test_predictions='labels' not in saved.files,
            prediction_ids_equal=bool(np.array_equal(saved['ids'], frame['id'].to_numpy())),
            probabilities_finite=bool(np.isfinite(probabilities).all()),
            probabilities_normalised=bool(np.allclose(probabilities.sum(1), 1, atol=1e-4)),
            probabilities_nonnegative=bool((probabilities >= 0).all()),
            argmax_equals_saved_predictions=bool(np.array_equal(probabilities.argmax(1), saved['predictions'])),
            argmax_equals_csv=bool(np.array_equal(probabilities.argmax(1), frame['label'].to_numpy())))
    metrics = json.loads((args.root/'metrics.json').read_text())
    digest = sha(csv)
    changed = int((frame['label'].to_numpy() != reference['label'].to_numpy()).sum())
    checks.update(sha_matches_metrics=digest == metrics['submission_sha256'],
                  changed_count_matches_metrics=changed == metrics['changed_test_predictions'],
                  status_not_submitted=metrics['status'] == 'ready_not_submitted')
    audit = dict(passed=bool(all(checks.values())), checks=checks, submission_sha256=digest,
                 changed_from_reference=changed,
                 class_counts=np.bincount(frame['label'].to_numpy(), minlength=10).tolist(),
                 reference_class_counts=np.bincount(reference['label'].to_numpy(), minlength=10).tolist(),
                 sample_sha256=SAMPLE_SHA256, reference_sha256=REFERENCE_SHA256,
                 note='Format and consistency only; this does not measure accuracy.')
    (args.root/'independent_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))
    assert audit['passed']


if __name__ == '__main__':
    main()
