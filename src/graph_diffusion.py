"""Smooth zero-shot probabilities over image-feature neighbours, without label seeds."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@torch.inference_mode()
def neighbours(features, k, temperature, device, chunk_size):
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    if not torch.isfinite(x).all() or (x.norm(dim=1) < 1e-8).any():
        raise ValueError('Expected finite, nonzero image features.')
    x = torch.nn.functional.normalize(x, dim=1)
    if not 0 < k < len(x):
        raise ValueError('Neighbour count must be smaller than the dataset.')
    indices, weights = [], []
    start = time.time()
    for offset in range(0, len(x), chunk_size):
        scores = x[offset:offset + chunk_size] @ x.T
        rows = torch.arange(len(scores), device=device)
        scores[rows, offset + rows] = -torch.inf
        values, index = scores.topk(k, dim=1)
        indices.append(index)
        weights.append((values / temperature).softmax(dim=1))
        if offset == 0 or offset + chunk_size >= len(x):
            print(f'Neighbours: {min(offset+chunk_size, len(x))}/{len(x)}, '
                  f'{time.time()-start:.1f}s', flush=True)
    indices, weights = torch.cat(indices), torch.cat(weights)
    if (indices == torch.arange(len(x), device=device)[:, None]).any():
        raise ValueError('Self-neighbours must be excluded.')
    return indices, weights


@torch.inference_mode()
def diffuse(logits, indices, weights, alpha, device, iterations=50):
    # This function has no access to validation labels or test labels.
    initial = torch.as_tensor(logits, dtype=torch.float32, device=device).softmax(dim=1)
    p = initial.clone()
    steps = 0
    for steps in range(1, iterations + 1):
        updated = (1-alpha)*initial + alpha*(weights[:, :, None]*p[indices]).sum(dim=1)
        change = float((updated-p).abs().max())
        p = updated
        if change < 1e-6:
            break
    if not torch.isfinite(p).all() or not torch.allclose(
            p.sum(dim=1), torch.ones(len(p), device=device), atol=1e-5):
        raise ValueError('Invalid class probabilities after diffusion.')
    return p.cpu().numpy(), steps


def load_split(base, features, split):
    with np.load(base / f'{split}_predictions.npz') as p, np.load(
            features / f'{split}_features.npz') as f:
        if not np.array_equal(p['ids'], f['ids']):
            raise ValueError('Feature IDs and prediction IDs differ.')
        if split == 'test' and ('labels' in p.files or 'labels' in f.files):
            raise ValueError('Test inputs must not contain labels.')
        result = {'ids': p['ids'], 'logits': p['logits'], 'features': f['features']}
        if result['logits'].shape != (len(result['ids']), 10):
            raise ValueError('Expected logits for the ten classes in the original order.')
        if not np.isfinite(result['logits']).all():
            raise ValueError('Non-finite input logits.')
        if split == 'validation':
            if not np.array_equal(p['labels'], f['labels']):
                raise ValueError('Validation labels differ between caches.')
            result['labels'] = p['labels']
        return result


def save_predictions(output, split, inputs, probabilities):
    saved = {'ids': inputs['ids'], 'probabilities': probabilities,
             'logits': np.log(np.maximum(probabilities, 1e-12)),
             'predictions': probabilities.argmax(axis=1)}
    if split == 'validation':
        saved['labels'] = inputs['labels']
    np.savez_compressed(output / f'{split}_predictions.npz', **saved)
    return saved['predictions']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--neighbours', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--chunk-size', type=int, default=512)
    args = parser.parse_args()
    if args.temperature <= 0 or args.chunk_size <= 0:
        raise ValueError('Temperature and chunk size must be positive.')
    if (args.output / 'metrics.json').exists():
        raise FileExistsError('Use a new output folder to preserve earlier results.')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    validation = load_split(args.base, args.features, 'validation')
    index, weights = neighbours(validation['features'], args.neighbours,
                                args.temperature, device, args.chunk_size)
    comparisons, best_probabilities = [], None
    best_accuracy, best_alpha = -1, None
    for alpha in [0, 0.25, 0.5, 0.75]:
        probabilities, iterations = diffuse(validation['logits'], index, weights, alpha, device)
        accuracy = float(np.mean(probabilities.argmax(1) == validation['labels']))
        comparisons.append({'alpha': alpha, 'validation_accuracy': accuracy,
                            'iterations': iterations})
        if accuracy > best_accuracy:
            best_accuracy, best_alpha, best_probabilities = accuracy, alpha, probabilities
    metrics = {'base_run': str(args.base), 'feature_run': str(args.features),
               'method': 'anchored diffusion of zero-shot probabilities on a directed kNN graph',
               'transductive': True, 'ground_truth_label_seeds': False,
               'neighbours': args.neighbours, 'neighbour_temperature': args.temperature,
               'logit_temperature': 1, 'maximum_iterations': 50, 'tolerance': 1e-6,
               'validation_comparisons': comparisons, 'alpha': best_alpha,
               'validation_accuracy': best_accuracy,
               'logit_representation': 'log of diffused probabilities'}
    save_predictions(args.output, 'validation', validation, best_probabilities)
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print('Validation:', comparisons, 'Selected alpha:', best_alpha, flush=True)
    test = load_split(args.base, args.features, 'test')
    sample = pd.read_csv(args.data / 'sample_submission.csv')
    if not np.array_equal(test['ids'], sample['id']) or len(np.unique(test['ids'])) != len(test['ids']):
        raise ValueError('Test IDs must match the complete submission template.')
    if best_alpha:
        index, weights = neighbours(test['features'], args.neighbours,
                                    args.temperature, device, args.chunk_size)
        probabilities, iterations = diffuse(test['logits'], index, weights, best_alpha, device)
    else:
        probabilities = torch.as_tensor(test['logits']).softmax(dim=1).numpy()
        iterations = 0
    predictions = save_predictions(args.output, 'test', test, probabilities)
    csv = args.output / 'submission.csv'
    pd.DataFrame({'id': test['ids'], 'label': predictions}).to_csv(csv, index=False)
    metrics.update(test_predictions=len(predictions), test_iterations=iterations,
                   changed_test_predictions=int(np.sum(predictions != test['logits'].argmax(1))),
                   submission_sha256=hashlib.sha256(csv.read_bytes()).hexdigest())
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print('Saved', csv, flush=True)


if __name__ == '__main__':
    main()
