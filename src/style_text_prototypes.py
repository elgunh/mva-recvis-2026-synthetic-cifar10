"""Text embeddings for the style-matched captions, for one frozen encoder.

Run in the encoder's own environment. Saves the 80-template class prototypes
(control, checked against the saved logits) and the 13 x 10 caption embeddings.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import torch

from clip_baseline import text_prototypes
from data import CLASS_NAMES
from generate_style_images import STYLES


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory.')
    args.output.mkdir(parents=True)
    base = json.loads((args.base/'metrics.json').read_text())
    precision = base.get('weight_precision', 'fp32')
    device = torch.device('cuda')
    model, _, _ = open_clip.create_model_and_transforms(
        base['model'], pretrained=None, precision=precision, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(base['model'])
    prototypes, prompts, source = text_prototypes(model, tokenizer, 'imagenet', device)
    assert prompts == base['prompts'], 'Control templates differ from the saved run.'
    with np.load(args.base/'validation_features.npz') as f, \
            np.load(args.base/'validation_predictions.npz') as p:
        assert np.array_equal(f['ids'], p['ids'])
        recomputed = 100*f['features'].astype(np.float64)@prototypes.double().cpu().numpy().T
        control_error = float(np.abs(recomputed-p['logits']).max())
        same_argmax = bool(np.array_equal(recomputed.argmax(1), p['logits'].argmax(1)))
    assert control_error < 2e-2 and same_argmax, control_error
    joint, captions = [], {}
    for key, template in STYLES:
        captions[key] = [template.format(name) for name in CLASS_NAMES]
        features = model.encode_text(tokenizer(captions[key]).to(device)).float()
        joint.append(features/features.norm(dim=-1, keepdim=True))
    joint = torch.stack(joint).cpu().numpy()
    assert joint.shape[:2] == (13, 10) and np.isfinite(joint).all()
    np.savez(args.output/'text_prototypes.npz', imagenet80=prototypes.cpu().numpy(),
             style_joint=joint, style_keys=np.array([k for k, _ in STYLES]))
    metrics = dict(model=base['model'], weight_precision=precision, base_run=str(args.base),
                   control_prompt_source=source, control_max_logit_error=control_error,
                   control_argmax_identical=same_argmax, captions=captions,
                   caption_source='CIFAR-10-Warehouse paper, Table 4 (templates only; no data)',
                   dimensions=int(joint.shape[2]))
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps({k: metrics[k] for k in ['model', 'control_max_logit_error',
                                               'control_argmax_identical', 'dimensions']}), flush=True)


if __name__ == '__main__':
    main()
