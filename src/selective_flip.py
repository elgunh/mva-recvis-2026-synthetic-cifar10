"""Add a reflected view to a fixed, label-free selection using a cached encoder run."""

import argparse
import json
import re
from pathlib import Path
import time

import numpy as np
import open_clip
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from clip_baseline import text_prototypes
from data import NpzImageDataset
from joint_graph_diffusion import mix_probabilities


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new run directory.')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_metrics = json.loads((args.base / 'metrics.json').read_text())
    model_name = base_metrics['model']
    if not model_name.startswith('hf-hub:'):
        raise ValueError('This experiment expects the two cached HF-hub checkpoints.')
    print('Loading', model_name, flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=None, precision=base_metrics['weight_precision'], device=device)
    model = model.to(device).eval()
    prototypes, prompts, prompt_source = text_prototypes(
        model, open_clip.get_tokenizer(model_name), 'imagenet', device)
    canonical_preprocess = lambda value: re.sub(r'0x[0-9a-fA-F]+', '<address>', value)
    compatibility = dict(prompts_match=prompts == base_metrics['prompts'],
        preprocess_match=canonical_preprocess(str(preprocess)) ==
                         canonical_preprocess(base_metrics['preprocess']),
        expected_preprocess=base_metrics['preprocess'], actual_preprocess=str(preprocess),
        ignored_description_field='function memory addresses only')
    (args.output / 'compatibility.json').write_text(json.dumps(compatibility,indent=2))
    if not compatibility['prompts_match'] or not compatibility['preprocess_match']:
        raise ValueError('Checkpoint preprocessing or prompts differ from the base run.')
    metrics = dict(model=model_name, checkpoint=base_metrics['checkpoint'],
                   base_run=str(args.base), selection=str(args.selection),
                   transformation='horizontal reflection of preprocessed tensor',
                   view_weights=[.5, .5], unselected_rows='original probabilities and features',
                   feature_aggregation='unit-normalized mean of original and reflected unit features',
                   logit_representation='log of view-averaged class probabilities',
                   prompt_source=prompt_source, prompts=prompts, preprocess=str(preprocess),
                   weight_precision=base_metrics['weight_precision'], splits={})
    metrics['compatibility'] = compatibility
    with np.load(args.selection) as selection:
        for split in ['validation', 'test']:
            started = time.time()
            dataset = NpzImageDataset(args.data / f'{split}.npz', transform=preprocess,
                                      require_labels=split == 'validation')
            with np.load(args.base / f'{split}_predictions.npz') as old, np.load(
                    args.base / f'{split}_features.npz') as cached:
                ids = old['ids']
                if not np.array_equal(ids, cached['ids']) or not np.array_equal(ids, dataset.ids):
                    raise ValueError('Prediction, feature and dataset IDs differ.')
                if not np.array_equal(ids, selection[f'{split}_ids']):
                    raise ValueError('Selection IDs differ.')
                mask = selection[f'{split}_mask']
                if mask.dtype != np.bool_ or mask.shape != (len(ids),):
                    raise ValueError('Invalid selection mask.')
                if split == 'test' and ('labels' in old.files or 'labels' in cached.files
                                        or dataset.labels is not None):
                    raise ValueError('Test inputs must be unlabeled.')
                if split == 'validation' and (not np.array_equal(dataset.labels, old['labels'])
                        or not np.array_equal(old['labels'], cached['labels'])):
                    raise ValueError('Validation labels differ.')
                original_logits = old['logits']
                features = cached['features'].copy()
            selected = np.flatnonzero(mask)
            loader = DataLoader(Subset(dataset, selected.tolist()), batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=device.type == 'cuda')
            reflected_features, reflected_logits = [], []
            for batch, (images, _) in enumerate(loader, 1):
                with torch.autocast(device.type, enabled=device.type == 'cuda'):
                    encoded = model.encode_image(images.to(device).flip(-1))
                encoded = torch.nn.functional.normalize(encoded.float(), dim=1)
                reflected_features.append(encoded.cpu().numpy())
                reflected_logits.append((100 * encoded @ prototypes.T).cpu().numpy())
                if batch == 1 or batch % 10 == 0 or batch == len(loader):
                    print(f'{split}: reflected {min(batch*args.batch_size,len(selected))}/'
                          f'{len(selected)}, {time.time()-started:.1f}s', flush=True)
            reflected_features = np.concatenate(reflected_features)
            reflected_logits = np.concatenate(reflected_logits)
            combined_features = features[selected] + reflected_features
            combined_features /= np.linalg.norm(combined_features, axis=1, keepdims=True)
            features[selected] = combined_features
            probabilities = mix_probabilities([original_logits], [1.])
            probabilities[selected] = mix_probabilities(
                [original_logits[selected], reflected_logits], [.5, .5])
            if (not np.isfinite(features).all() or not np.isfinite(probabilities).all()
                    or not np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-5)
                    or not np.allclose(probabilities.sum(1),1,atol=1e-8)):
                raise ValueError('Invalid averaged features or probabilities.')
            predicted = probabilities.argmax(1)
            saved = dict(ids=ids, probabilities=probabilities, predictions=predicted,
                         logits=np.log(np.maximum(probabilities,1e-30)))
            cached = dict(ids=ids, features=features)
            result = dict(rows=len(ids), reflected_rows=len(selected), seconds=time.time()-started,
                          changed_raw_labels=int(np.sum(predicted != original_logits.argmax(1))))
            if split == 'validation':
                saved['labels'] = dataset.labels
                cached['labels'] = dataset.labels
                result['accuracy'] = float(np.mean(predicted == dataset.labels))
                metrics['validation_accuracy'] = result['accuracy']
            else:
                sample = pd.read_csv(args.data / 'sample_submission.csv')
                if not np.array_equal(ids, sample['id']):
                    raise ValueError('Sample ID order differs.')
                pd.DataFrame({'id':ids, 'label':predicted}).to_csv(
                    args.output / 'submission.csv', index=False)
            np.savez_compressed(args.output / f'{split}_predictions.npz', **saved)
            np.savez(args.output / f'{split}_features.npz', **cached)
            metrics['splits'][split] = result
            (args.output / 'metrics.json').write_text(json.dumps(metrics,indent=2))
            print(split, result, flush=True)
    print('Completed', args.output, flush=True)


if __name__ == '__main__':
    main()
