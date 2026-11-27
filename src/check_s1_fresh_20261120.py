"""Compare two fixed S1 recipes on newly generated images using the adapted base."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

import selftrain_export as X
import synth_style_experiment as E
from check_dino_geometry import geometry
from external_cache import smooth
from graph_diffusion import neighbours
from joint_graph_diffusion import fuse_features, mix_probabilities
from train_dino_selftrain import Head, add_lora as add_dino_lora, features, load_dino
from train_lowres_student import add_lora as add_clip_lora, load_lora, load_model
from train_test_selftrain import Images, folds
from x2_pilot_check import counts

BASE = Path('runs/adapted_bag3_graph_20261116')
KEYS = {'bigg': 'PE-Core-bigG', 'siglip': 'SigLIP2', 'dino': 'dinov2'}
GROUP_COUNTS = {'bigg': 3, 'siglip': 1, 'dino': 1}
STUDENTS = {
    'incumbent': Path('runs/x2_dino_sd21_768_train400_top2'),
    'candidate': Path('runs/x2_dino_sd21_768_train800_top2'),
}
SUBMISSIONS = {
    'incumbent': Path('runs/x2_s1_sd21_768_train400_20261118'),
    'candidate': Path('runs/x2_s1_sd21_768_train800_20261120'),
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def groups(encoder):
    encodes = [row for row in read_json(BASE / 'metrics.json')['encodes'] if KEYS[encoder] in row['model']]
    if len(encodes) != 2 * GROUP_COUNTS[encoder]:
        raise ValueError(f'Unexpected number of {encoder} checkpoints.')
    pairs = [encodes[i:i + 2] for i in range(0, len(encodes), 2)]
    if any({row['student_fold'] for row in pair} != {0, 1} for pair in pairs):
        raise ValueError(f'Incomplete {encoder} fold pair.')
    return pairs


def load_fresh(root, work):
    records = read_json(root / 'manifest.json')['records']
    ids = np.array([row['id'] for row in records])
    with np.load(work / 'judge' / 'judge.npz') as data:
        assert np.array_equal(data['ids'], ids)
        kept = data['top2'].astype(bool)
    frozen_logits = []
    for name in ('bigg', 'siglip'):
        with np.load(work / f'frozen_{name}' / 'external_features.npz') as data:
            assert np.array_equal(data['ids'], ids)
            frozen_logits.append(data['logits'])
    frozen_prior = mix_probabilities(frozen_logits, [.75, .25])
    selected = (frozen_logits[0].argmax(1) != frozen_logits[1].argmax(1)) | (frozen_prior.max(1) < .9)
    return records, ids, kept, selected


def read_pixels(root, records):
    images = []
    for row in records:
        with Image.open(root / row['path']) as image:
            assert image.size == (32, 32)
            images.append(np.asarray(image.convert('RGB'), dtype=np.uint8))
    return np.stack(images)


def encode(images, rows, selected, transform, forward, device, batch_size, arrays):
    loader = DataLoader(Images(images, rows, transform, False), batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    for batch_index, (batch, index) in enumerate(loader, 1):
        index = index.numpy()
        batch = batch.to(device)
        encoded = [('original', index, batch)]
        reflect = selected[index]
        if reflect.any():
            encoded.append(('reflected', index[reflect], batch[reflect].flip(-1)))
        for name, positions, view in encoded:
            feature, probability_logits = forward(view)
            for kind, values in [('features', feature), ('logits', probability_logits)]:
                key = f'{name}_{kind}'
                array = values.float().cpu().numpy()
                if key not in arrays:
                    arrays[key] = np.zeros((len(images), array.shape[1]), dtype=np.float32)
                arrays[key][positions] = array
        if batch_index % 30 == 0 or batch_index == len(loader):
            print(f'{min(batch_index * batch_size, len(rows))}/{len(rows)} rows', flush=True)
    return arrays


def encoder_views(encoder, pairs, images, selected, device, students, first_holdout_seed):
    first = read_json(Path(pairs[0][0]['student']) / 'metrics.json')
    base_rank = first['rank']
    is_dino = encoder == 'dino'
    if is_dino:
        model, transform = load_dino(device, first.get('image_size', 224))
        blocks = add_dino_lora(model, base_rank)
    else:
        model, transform, prototypes = load_model(read_json(Path(first['base_run']) / 'metrics.json'), device)
        blocks = add_clip_lora(model, base_rank)
        model.set_grad_checkpointing(False)

    def encode_checkpoint(folder, rows, arrays):
        meta = read_json(folder / 'metrics.json')
        assert meta['model'] == first['model'] and meta['rank'] == base_rank
        if is_dino:
            assert meta.get('image_size', 224) == first.get('image_size', 224)
            state = torch.load(folder / 'student.pt', map_location='cpu', weights_only=True)
            expected = {k for k in blocks.state_dict() if k.rsplit('.', 1)[-1] in ('down', 'up')}
            assert set(state['lora']) == expected
            blocks.load_state_dict(state['lora'], strict=False)
            head = Head(state['head'], state['scale']).to(device).eval()

            def forward(batch):
                embedding = F.normalize(features(model, batch, device), dim=1)
                return embedding, head(embedding)
        else:
            assert meta['base_run'] == first['base_run']
            load_lora(blocks, folder / 'lora.pt')

            def forward(batch):
                with torch.autocast('cuda'):
                    embedding = model.encode_image(batch)
                embedding = F.normalize(embedding.float(), dim=1)
                return embedding, 100 * embedding @ prototypes.T

        print(f'{folder}: {len(rows)} fresh rows', flush=True)
        return encode(images, rows, selected, transform, forward, device, 64 if is_dino else 16, arrays)

    combined = []
    for pair in pairs:
        arrays = {}
        metas = [read_json(Path(entry['student']) / 'metrics.json') for entry in pair]
        assert metas[0]['fold_seed'] == metas[1]['fold_seed']
        allocation = folds(len(images), metas[0]['fold_seed'])
        for entry in pair:
            rows = np.flatnonzero(allocation != entry['student_fold'])
            encode_checkpoint(Path(entry['student']), rows, arrays)
        combined.append(X.view_rule(arrays, selected))
    student_probs = {}
    if is_dino:
        for name, folder in students.items():
            meta = read_json(folder / 'metrics.json')
            assert meta['fold'] == -1 and meta['max_train_index'] <= first_holdout_seed
            full = encode_checkpoint(folder, np.arange(len(images)), {})
            student_probs[name] = X.view_rule(full, selected)[0]
    return combined, student_probs


def combine(parts):
    if len(parts) == 1:
        return parts[0]
    probabilities, unit_features = zip(*parts)
    features_mean = np.mean(unit_features, axis=0)
    return np.mean(probabilities, axis=0), features_mean / np.linalg.norm(features_mean, axis=1, keepdims=True)


@torch.inference_mode()
def make_encoder(args, records, ids, kept, selected):
    output = args.work / f'{args.stage}.npz'
    if output.exists():
        raise FileExistsError(output)
    images = read_pixels(args.root, records)
    pairs = groups(args.stage)
    parts, student = encoder_views(args.stage, pairs, images, selected, torch.device('cuda'),
                                   args.students, args.seed_range[0])
    prior, embedding = combine(parts)
    assert np.isfinite(prior).all() and np.isfinite(embedding).all()
    if student:
        truth = np.array([row['label'] for row in records])
        right = {k: v.argmax(1) == truth for k, v in student.items()}
        print('STUDENTS_ONLY_FRESH', counts(right['candidate'], right['incumbent']), flush=True)
    np.savez(output, ids=ids, selected=selected, probabilities=prior, features=embedding, **student)


def adapted_inputs(encodes):
    big, small, dino = [encodes[k] for k in ('bigg', 'siglip', 'dino')]
    mix = .75 * big[0] + .25 * small[0]
    feature = geometry(fuse_features([big[1], small[1]], [.75, .25]), dino[1], .25)
    return mix / mix.sum(1, keepdims=True), feature


def s1(base, student):
    predictions = base.copy()
    low = base.max(1) < .9
    predictions[low] = E.softmax(.5 * np.log(np.maximum(base[low], 1e-12)) +
                                 .5 * np.log(np.maximum(student[low], 1e-12)))
    return predictions.argmax(1)


def score(args, records, ids, kept, selected):
    out = args.work / 'result.json'
    if out.exists():
        raise FileExistsError(out)
    with np.load(E.SELECTION) as selection:
        test_ids, test_selected = selection['test_ids'], selection['test_mask']
    with np.load(BASE / 'test_predictions.npz') as prediction:
        assert 'labels' not in prediction.files and np.array_equal(test_ids, prediction['ids'])
        saved_base = prediction['probabilities'].astype(np.float64)

    test, fresh = {}, {}
    for encoder in KEYS:
        parts = [X.view_rule(X.merge([Path(r['folder']) for r in pair], test_ids, test_selected, KEYS[encoder])[0],
                             test_selected) for pair in groups(encoder)]
        test[encoder] = combine(parts)
        with np.load(args.work / f'{encoder}.npz') as data:
            assert np.array_equal(data['ids'], ids) and np.array_equal(data['selected'], selected)
            fresh[encoder] = data['probabilities'], data['features']
    test_prior, test_feature = adapted_inputs(test)
    fresh_prior, fresh_feature = adapted_inputs(fresh)
    rebuilt = smooth(test_prior, *neighbours(test_feature, 20, .07, 'cpu', 512), 'cpu')
    base_control = int((rebuilt.argmax(1) != saved_base.argmax(1)).sum())
    base_error = float(np.abs(rebuilt - saved_base).max())
    if base_control or base_error > 1e-4:
        raise ValueError(f'Base control failed: {base_control} changing labels, max error {base_error}')

    n = len(test_ids)
    joined = smooth(np.concatenate([test_prior, fresh_prior]),
                    *neighbours(np.concatenate([test_feature, fresh_feature]), 20, .07, 'cpu', 512), 'cpu')
    target_base = joined[n:]
    labels = np.array([r['label'] for r in records])
    predictions = {}
    with np.load(args.work / 'dino.npz') as data:
        for name, student in args.students.items():
            original, _ = X.merge([student.with_name(student.name + '_test')], test_ids, test_selected, 'dinov2')
            control = s1(saved_base, X.view_rule(original, test_selected)[0])
            with np.load(args.submissions[name] / 'test_predictions.npz') as saved:
                assert np.array_equal(saved['ids'], test_ids) and 'labels' not in saved.files
                assert np.array_equal(control, saved['predictions']), f'{name} S1 control failed'
            predictions[name] = s1(target_base, data[name])

    result = dict(seed_range=args.seed_range, rows=len(ids), base=str(BASE),
                  incumbent_public=args.incumbent_public, incumbent_run=str(args.submissions['incumbent']),
                  candidate_run=str(args.submissions['candidate']),
                  base_test_control_label_differences=base_control, base_test_control_max_probability_error=base_error,
                  original_s1_test_controls_match=True, reflected_rows=int(selected.sum()),
                  base_low_confidence_rows=int((target_base.max(1) < .9).sum()))
    for name, mask in [('all', np.ones(len(ids), bool)), ('top2', kept)]:
        old, new = (predictions[k][mask] == labels[mask] for k in ('incumbent', 'candidate'))
        result[name] = dict(rows=int(mask.sum()), incumbent_errors=int((~old).sum()),
                            candidate_errors=int((~new).sum()), **counts(new, old))
    result['evidence_supports_submission'] = result['all']['z'] >= 2 and result['top2']['z'] > 0
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('S1_FRESH_RESULT', json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=[*KEYS, 'score'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--seed-range', type=int, nargs=2, default=[900, 938], metavar=('START', 'STOP'))
    parser.add_argument('--incumbent-public', type=float, default=.99066)
    for name in STUDENTS:
        parser.add_argument(f'--{name}-student', type=Path, default=STUDENTS[name])
        parser.add_argument(f'--{name}-submission', type=Path, default=SUBMISSIONS[name])
    args = parser.parse_args()
    start, stop = args.seed_range
    if not 0 <= start < stop:
        parser.error('--seed-range requires 0 <= START < STOP.')
    args.students = {k: getattr(args, f'{k}_student') for k in STUDENTS}
    args.submissions = {k: getattr(args, f'{k}_submission') for k in STUDENTS}
    torch.set_num_threads(8 if args.stage == 'score' else 4)
    fresh = load_fresh(args.root, args.work)
    if {r['index'] for r in fresh[0]} != set(range(start, stop)):
        raise ValueError('The generated manifest does not match the full holdout seed range.')
    if args.stage == 'score':
        score(args, *fresh)
    else:
        make_encoder(args, *fresh)


if __name__ == '__main__':
    main()
