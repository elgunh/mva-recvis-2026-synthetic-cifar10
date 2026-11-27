"""Evaluate the starter's open-source CLIP baseline on validation and test."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import CLASS_NAMES, NpzImageDataset
from main import validate_inputs


@torch.inference_mode()
def text_prototypes(model, tokenizer, templates, device):
    if templates == 'starter':
        prompts = [f'a photo of a {name}.' for name in CLASS_NAMES]
        features = model.encode_text(tokenizer(prompts).to(device)).float()
        features /= features.norm(dim=-1, keepdim=True)
        return features, prompts, 'teacher starter data.py/clip_zero_shot'
    from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES
    prompts, prototypes = {}, []
    for name in CLASS_NAMES:
        prompts[name] = [template(name) for template in OPENAI_IMAGENET_TEMPLATES]
        features = model.encode_text(tokenizer(prompts[name]).to(device)).float()
        features /= features.norm(dim=-1, keepdim=True)
        prototype = features.mean(dim=0)
        prototypes.append(prototype / prototype.norm())
    return (torch.stack(prototypes), prompts,
            'open_clip.zero_shot_metadata.OPENAI_IMAGENET_TEMPLATES (80 existing templates)')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='ViT-B-32')
    parser.add_argument('--pretrained', default='laion2b_s34b_b79k')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--preflight', action='store_true', help='Benchmark one validation batch; do not create a submission.')
    parser.add_argument('--save-features', action='store_true', help='Cache image features on the server for later classifiers.')
    parser.add_argument('--templates', choices=['starter', 'imagenet'], default='starter')
    args = parser.parse_args()
    validate_inputs(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hub_model = args.model.startswith('hf-hub:')
    checkpoint = args.model[7:] if hub_model else args.pretrained
    print(f'Loading open-source {args.model}, checkpoint {checkpoint}', flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=None if hub_model else args.pretrained,
        precision=args.precision, device=device,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    # Exact existing template from the teacher's data.py/clip_zero_shot,
    # starter commit 92aad957eb112dc99d62038bd2858cab3c14636e.
    text_features, prompts, prompt_source = text_prototypes(
        model, tokenizer, args.templates, device)
    metrics = {
        'model': args.model, 'checkpoint': checkpoint,
        'prompts': prompts, 'prompt_source': prompt_source,
        'pretrained_config': ({'hf_hub': args.model[7:]} if hub_model else
                              open_clip.get_pretrained_cfg(args.model, args.pretrained)),
        'preprocess': str(preprocess), 'mixed_precision': device.type == 'cuda',
        'weight_precision': args.precision,
    }
    if args.preflight:
        dataset = NpzImageDataset(args.data/'validation.npz', transform=preprocess, require_labels=True)
        images = torch.stack([dataset[i][0] for i in range(args.batch_size)]).to(device)
        elapsed = []
        for repeat in range(4):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                features = model.encode_image(images)
            features = features.float()
            features /= features.norm(dim=-1, keepdim=True)
            logits = 100 * features @ text_features.T
            if not torch.isfinite(logits).all():
                raise ValueError('Non-finite preflight logits.')
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if repeat:
                elapsed.append(time.perf_counter()-start)
        result = {'preflight_only': True, 'batch_size': args.batch_size,
                  'images_per_second': float(args.batch_size / np.mean(elapsed)),
                  'peak_gpu_memory_mb': (torch.cuda.max_memory_allocated()/2**20
                                         if device.type == 'cuda' else None),
                  'finite_logits': True}
        metrics.update(result)
        (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2))
        print(json.dumps(result), flush=True)
        return
    for split in ['validation', 'test']:
        started = time.time()
        dataset = NpzImageDataset(args.data/f'{split}.npz', transform=preprocess,
                                  require_labels=split == 'validation')
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=device.type == 'cuda')
        logits, image_features = [], []
        for batch, (images, _) in enumerate(loader, 1):
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                features = model.encode_image(images.to(device))
            features = features.float()
            features /= features.norm(dim=-1, keepdim=True)
            if args.save_features:
                image_features.append(features.cpu().numpy())
            logits.append((100 * features @ text_features.T).cpu().numpy())
            if batch == 1 or batch % 20 == 0 or batch == len(loader):
                print(f'{split}: batch {batch}/{len(loader)}, {time.time()-started:.1f}s', flush=True)
        logits = np.concatenate(logits)
        predicted = logits.argmax(axis=1)
        saved = {'logits': logits, 'ids': dataset.ids, 'predictions': predicted}
        if split == 'validation':
            saved['labels'] = dataset.labels
            accuracy = float(np.mean(predicted == dataset.labels))
            metrics['validation_accuracy'] = accuracy
            print(f'CLIP validation accuracy: {accuracy:.6f}', flush=True)
        else:
            if dataset.labels is not None:
                raise ValueError('Competition test data must not contain labels.')
            sample = pd.read_csv(args.data/'sample_submission.csv')
            submission = pd.DataFrame({'id': dataset.ids, 'label': predicted})
            if submission['id'].duplicated().any() or set(submission['id']) != set(sample['id']):
                raise ValueError('Prediction IDs do not match the submission template.')
            submission = submission.set_index('id').loc[sample['id']].reset_index()
            submission.to_csv(args.output/'submission.csv', index=False)
            metrics['test_predictions'] = len(submission)
        np.savez_compressed(args.output/f'{split}_predictions.npz', **saved)
        if args.save_features:
            cached = {'features': np.concatenate(image_features), 'ids': dataset.ids}
            if split == 'validation':
                cached['labels'] = dataset.labels
            np.savez(args.output/f'{split}_features.npz', **cached)
            metrics['image_features_saved'] = True
        metrics[f'{split}_seconds'] = time.time()-started
        (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f'Saved {args.output / "submission.csv"}', flush=True)


if __name__ == '__main__':
    main()
