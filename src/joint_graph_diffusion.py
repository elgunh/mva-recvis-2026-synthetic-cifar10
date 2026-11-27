"""Build one neighbour graph from two frozen encoders, then diffuse their mean predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graph_diffusion import diffuse, load_split, neighbours, save_predictions


def fuse_features(features, weights):
    parts = []
    for feature, weight in zip(features, weights):
        feature = np.asarray(feature, dtype=np.float32)
        norms = np.linalg.norm(feature, axis=1, keepdims=True)
        if not np.isfinite(feature).all() or np.any(norms < 1e-8):
            raise ValueError('Features must be finite and nonzero.')
        parts.append(np.sqrt(weight) * feature / norms)
    # The dot product is the weighted mean of the encoders' cosine similarities.
    return np.concatenate(parts, axis=1)


def mix_probabilities(logits, weights):
    result = np.zeros_like(logits[0], dtype=np.float64)
    for scores, weight in zip(logits, weights):
        scores = np.asarray(scores, dtype=np.float64)
        probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        result += weight * probabilities
    return result / result.sum(axis=1, keepdims=True)


@torch.inference_mode()
def mutual_edges(index, weights):
    """Keep reciprocal kNN edges; retain original rows if none are reciprocal."""
    neighbours_np = index.cpu().numpy()
    n = len(neighbours_np)
    rows = np.arange(n, dtype=np.int64)[:, None]
    encoded = np.sort((rows * n + neighbours_np).reshape(-1))
    reverse = neighbours_np * n + rows
    positions = np.searchsorted(encoded, reverse)
    keep = (positions < len(encoded)) & (encoded[np.minimum(positions, len(encoded)-1)] == reverse)
    empty = ~keep.any(axis=1)
    reciprocal_count = int(keep.sum())
    keep[empty] = True
    filtered = weights * torch.as_tensor(keep, device=weights.device)
    filtered = filtered / filtered.sum(1, keepdim=True)
    return filtered, dict(reciprocal_edges=reciprocal_count,
                         original_edges=int(neighbours_np.size),
                         fallback_rows=int(empty.sum()),
                         retained_edges_with_fallback=int(keep.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs=2, required=True)
    parser.add_argument('--weights', type=float, nargs=2, default=[.75, .25])
    parser.add_argument('--prior-weights', type=float, nargs=2,
                        help='Initial probability weights; defaults to feature weights.')
    parser.add_argument('--mutual', action='store_true', help='Use reciprocal neighbour edges.')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alpha', type=float, default=.5)
    parser.add_argument('--neighbours', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=.07)
    parser.add_argument('--chunk-size', type=int, default=512)
    args = parser.parse_args()
    weights = np.asarray(args.weights, dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Weights must be finite, nonnegative and have a positive sum.')
    if not 0 <= args.alpha < 1 or args.temperature <= 0 or args.chunk_size <= 0:
        raise ValueError('Invalid diffusion settings.')
    weights /= weights.sum()
    prior_weights = np.asarray(args.prior_weights if args.prior_weights is not None
                               else weights, dtype=np.float64)
    if not np.isfinite(prior_weights).all() or np.any(prior_weights < 0) or prior_weights.sum() <= 0:
        raise ValueError('Invalid initial probability weights.')
    prior_weights /= prior_weights.sum()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory.')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = dict(runs=[str(r) for r in args.runs], weights=weights.tolist(),
                   prior_weights=prior_weights.tolist(), mutual_neighbours=args.mutual,
                   reference=str(args.reference), method='joint feature graph',
                   feature_similarity='weighted mean of normalized encoder cosine similarities',
                   initial_probabilities='weighted mean of softmax(input logits); see input run metadata',
                   alpha=args.alpha, neighbours=args.neighbours,
                   neighbour_temperature=args.temperature, maximum_iterations=50,
                   tolerance=1e-6, parameter_selection='fixed before this experiment',
                   transductive=True, ground_truth_label_seeds=False, splits={})
    for split in ['validation', 'test']:
        inputs = [load_split(run, run, split) for run in args.runs]
        ids = inputs[0]['ids']
        if not np.array_equal(ids, inputs[1]['ids']) or len(np.unique(ids)) != len(ids):
            raise ValueError('Encoder rows must have the same unique IDs.')
        if split == 'validation' and not np.array_equal(inputs[0]['labels'], inputs[1]['labels']):
            raise ValueError('Validation labels must match.')
        if split == 'test':
            sample = pd.read_csv(args.data / 'sample_submission.csv')
            if not np.array_equal(ids, sample['id']):
                raise ValueError('Test IDs must match the submission template.')
        initial = mix_probabilities([r['logits'] for r in inputs], prior_weights)
        features = fuse_features([r['features'] for r in inputs], weights)
        index, edge_weights = neighbours(features, args.neighbours, args.temperature,
                                         device, args.chunk_size)
        graph_stats = {}
        if args.mutual:
            edge_weights, graph_stats = mutual_edges(index, edge_weights)
        if split == 'validation':
            y = torch.as_tensor(inputs[0]['labels'], device=device)
            graph_stats['cross_class_edge_weight_fraction'] = float(
                (edge_weights * (y[:, None] != y[index])).sum() / len(y))
        probabilities, iterations = diffuse(np.log(np.maximum(initial, 1e-30)),
                                           index, edge_weights, args.alpha, device)
        predictions = save_predictions(args.output, split, inputs[0], probabilities)
        with np.load(args.reference / f'{split}_predictions.npz') as reference:
            if not np.array_equal(ids, reference['ids']):
                raise ValueError('Reference IDs differ.')
            old = reference['predictions']
            result = dict(rows=len(ids), iterations=iterations,
                          changed_from_reference=int(np.sum(predictions != old)),
                          feature_dimensions=features.shape[1],
                          graph=graph_stats,
                          finite_normalized_probabilities=True)
            if split == 'validation':
                labels = inputs[0]['labels']
                if not np.array_equal(labels, reference['labels']):
                    raise ValueError('Reference labels differ.')
                result.update(accuracy=float(np.mean(predictions == labels)),
                              initial_accuracy=float(np.mean(initial.argmax(1) == labels)),
                              reference_accuracy=float(np.mean(old == labels)),
                              corrected=int(np.sum((old != labels) & (predictions == labels))),
                              new_errors=int(np.sum((old == labels) & (predictions != labels))))
                metrics['validation_accuracy'] = result['accuracy']
            else:
                if 'labels' in reference.files:
                    raise ValueError('Reference test data must not contain labels.')
                csv = args.output / 'submission.csv'
                pd.DataFrame({'id': ids, 'label': predictions}).to_csv(csv, index=False)
                metrics['submission_sha256'] = hashlib.sha256(csv.read_bytes()).hexdigest()
        metrics['splits'][split] = result
        (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print(split, result, flush=True)
        del inputs, features, index, edge_weights
    print('Saved', args.output / 'submission.csv', flush=True)


if __name__ == '__main__':
    main()
