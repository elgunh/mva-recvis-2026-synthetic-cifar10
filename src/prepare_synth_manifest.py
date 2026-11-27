"""Choose the 32px kernel without labels, then build a manifest of generated images.

Stage `kernel`: compare low-level statistics of pilot images with the test
images (pixels only) and freeze one downsampling kernel.
Stage `build`: write 32x32 PNG files and `manifest.json` in the format used by
encode_domainnet.py.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PIL_KERNELS = {'pil_nearest': Image.Resampling.NEAREST, 'pil_box': Image.Resampling.BOX,
               'pil_bilinear': Image.Resampling.BILINEAR, 'pil_hamming': Image.Resampling.HAMMING,
               'pil_bicubic': Image.Resampling.BICUBIC, 'pil_lanczos': Image.Resampling.LANCZOS}
TORCH_KERNELS = {'torch_bilinear_noaa': 'bilinear', 'torch_bicubic_noaa': 'bicubic',
                 'torch_area': 'area'}
KERNELS = list(PIL_KERNELS)+list(TORCH_KERNELS)
BANDS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 23)]


def split_choice(chosen):
    """'native:pil_nearest' -> ('native', 'pil_nearest'); a bare kernel name means the 224px file."""
    source, _, kernel = chosen.rpartition(':')
    return (source or 'hires'), kernel


def source_path(root, row, source):
    return root/(row['native_path'] if source == 'native' else row['path'])


def pixels32(root, row, source, kernel, chosen):
    """The 32px image of one record: saved at generation time with the same frozen choice, or computed here."""
    if row.get('small_kernel') == chosen:
        with Image.open(root/row['small_path']) as image:
            pixels = np.asarray(image.convert('RGB'), dtype=np.uint8)
        assert pixels.shape == (32, 32, 3)
        return pixels
    return downsample(Image.open(source_path(root, row, source)).convert('RGB'), kernel)


def downsample(image, kernel):
    """Square RGB PIL image (the 224px file or the generated size) -> (32, 32, 3) uint8."""
    assert image.size[0] == image.size[1] and image.mode == 'RGB'
    if kernel in PIL_KERNELS:
        return np.asarray(image.resize((32, 32), PIL_KERNELS[kernel]), dtype=np.uint8)
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1)[None]
    mode = TORCH_KERNELS[kernel]
    options = {} if mode == 'area' else dict(align_corners=False, antialias=False)
    small = torch.nn.functional.interpolate(pixels, size=(32, 32), mode=mode, **options)
    return small[0].permute(1, 2, 0).clamp(0, 255).round().numpy().astype(np.uint8)


def low_level(images):
    """Label-free statistics of (N, 32, 32, 3) uint8 images, in float64."""
    grey = images.astype(np.float64).mean(-1)/255
    centred = grey-grey.mean((1, 2), keepdims=True)
    power = (np.abs(np.fft.fftshift(np.fft.fft2(centred), axes=(1, 2)))**2).mean(0)
    yy, xx = np.mgrid[-16:16, -16:16]
    radius = np.sqrt(yy**2+xx**2)
    bands = [float(power[(radius >= a) & (radius < b)].sum()/power.sum()) for a, b in BANDS]
    gradient = (np.abs(np.diff(grey, axis=2)).mean((1, 2))
                + np.abs(np.diff(grey, axis=1)).mean((1, 2)))
    lag1 = (centred[:, :, 1:]*centred[:, :, :-1]).sum((1, 2))/((centred**2).sum((1, 2))+1e-12)
    colour = images.astype(np.float64)/255
    return dict(bands=bands, gradient_median=float(np.median(gradient)),
                lag1_autocorrelation_median=float(np.median(lag1)),
                image_mean_sd=float(colour.mean((1, 2, 3)).std()),
                grey_fraction=float(((colour.max(-1)-colour.min(-1)).mean((1, 2)) < .01).mean()),
                rows=int(len(images)))


def records_of(root):
    rows = {}
    for log in sorted(root.glob('records_*.jsonl')):
        for line in log.read_text().splitlines():
            if line:
                row = json.loads(line)
                rows[row['id']] = row
    return [rows[key] for key in sorted(rows)]


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def choose_kernel(args):
    if args.output.exists():
        raise FileExistsError('The kernel choice is frozen; it is never overwritten.')
    test = np.load(args.data/'test.npz')['images']
    target = low_level(test)
    pilots = []
    for root in args.gen_roots:
        rows = [r for r in records_of(root) if r['index'] < args.pilot_end and r['pixel_std'] >= .02]
        assert len(rows) >= .95*130*args.pilot_end, f'Pilot incomplete for {root}'
        pilots += [(root, r) for r in rows]
    results, names = {}, []
    for source in args.sources:  # 'hires' = saved 224px file, 'native' = generated size
        images = [Image.open(source_path(root, r, source)).convert('RGB') for root, r in pilots]
        for kernel in KERNELS:
            name = kernel if source == 'hires' else f'{source}:{kernel}'
            small = np.stack([downsample(image, kernel) for image in images])
            stats = low_level(small)
            stats['l1_band_distance'] = float(np.abs(np.array(stats['bands'])-np.array(target['bands'])).sum())
            results[name] = stats
            names.append(name)
            print(name, json.dumps(stats), flush=True)
        del images
    chosen = min(names, key=lambda k: (results[k]['l1_band_distance'], names.index(k)))
    decision = dict(chosen=chosen, criterion='smallest L1 distance between five-band radial power spectra',
                    sources=args.sources,
                    bands=BANDS, test=target, candidates=results, pilot_images=len(pilots),
                    pilot_roots=[str(r) for r in args.gen_roots], pilot_end=args.pilot_end,
                    labels_used=False, test_pixels_read=True)
    args.output.write_text(json.dumps(decision, indent=2)+'\n')
    print('Chosen kernel:', chosen, 'test:', json.dumps(target), flush=True)


@torch.inference_mode()
def nearest_test_rmse(images, test, device):
    a = torch.as_tensor(images.reshape(len(images), -1), dtype=torch.float32, device=device)/255
    b = torch.as_tensor(test.reshape(len(test), -1), dtype=torch.float32, device=device)/255
    b2 = (b*b).sum(1)
    best = []
    for offset in range(0, len(a), 1024):
        chunk = a[offset:offset+1024]
        squared = (chunk*chunk).sum(1, keepdim=True)+b2[None]-2*chunk@b.T
        best.append((squared.min(1).values.clamp_min(0)/a.shape[1]).sqrt().cpu())
    return torch.cat(best).numpy()


def build(args):
    if args.output.exists():
        raise FileExistsError('Use a new manifest directory.')
    decision_bytes = args.kernel_file.read_bytes()
    chosen = json.loads(decision_bytes)['chosen']
    source, kernel = split_choice(chosen)
    root = args.gen_roots[0]
    rows = records_of(root)
    if args.max_index is not None:
        rows = [r for r in rows if r['index'] < args.max_index]
    settings = json.loads((root/'settings.json').read_text())
    assert len(rows) == args.expected, f'{len(rows)} records, expected {args.expected}'
    assert len({r['id'] for r in rows}) == len(rows)
    forbidden = set()
    for split in ['train', 'validation', 'test']:
        forbidden.update(sha(image) for image in np.load(args.data/f'{split}.npz')['images'])
    previous, previous_sources = set(), {}
    for path in args.exclude_manifest:
        previous_bytes = path.read_bytes()
        old_manifest = json.loads(previous_bytes)
        previous.update(r['pixels32_sha256'] for r in old_manifest['records'])
        previous_sources[str(path)] = hashlib.sha256(previous_bytes).hexdigest()
    small = np.stack([pixels32(root, r, source, kernel, chosen) for r in rows])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    distance = nearest_test_rmse(small, np.load(args.data/'test.npz')['images'], device)
    near = float((distance < .02).mean())
    # Safeguard of the 16 November plan: the same model and prompt could reproduce a test image.
    assert near <= .001, f'{near:.4%} of the images are near-duplicates of test images: stop and review.'
    kept, excluded, seen = [], [], set()
    for row, pixels, rmse in zip(rows, small, distance):
        digest = sha(pixels)
        reason = None
        if row['pixel_std'] < .02:
            reason = 'flat_image'
        elif digest in forbidden:
            reason = 'provided_train_validation_or_test_duplicate'
        elif digest in previous:
            reason = 'previous_generated_manifest_duplicate'
        elif digest in seen:
            reason = 'within_set_duplicate'
        elif rmse < .02:
            reason = 'near_duplicate_of_test_image'
        record = dict(id=row['id'], path=f"images/{row['style']}/{row['category']}/{row['index']:04d}.png",
                      role=args.role, domain=row['style'], category=row['category'], label=row['label'],
                      generator=row['generator'], index=row['index'], seed=row['seed'], prompt=row['prompt'],
                      hires_path=str(root/row['path']), file_sha256=row['file_sha256'],
                      pixels32_sha256=digest, nearest_test_rmse=round(float(rmse), 5))
        if reason:
            excluded.append(dict(record, reason=reason))
            continue
        seen.add(digest)
        target = args.output/record['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels, mode='RGB').save(target, format='PNG')
        with Image.open(target) as check:
            assert check.size == (32, 32) and np.array_equal(np.asarray(check.convert('RGB')), pixels)
        kept.append(record)
    reasons = {}
    for row in excluded:
        reasons[row['reason']] = reasons.get(row['reason'], 0)+1
    manifest = dict(source='own generation with an open-weight text-to-image model; see SYNTH_STYLE_PLAN.md',
                    generator_settings=settings, role=args.role, kernel=kernel, kernel_source=source,
                    kernel_decision_sha256=hashlib.sha256(decision_bytes).hexdigest(),
                    resize=(f'generated size -> 32 with {kernel}' if source == 'native' else
                            f'native -> 224 PIL bicubic (saved PNG) -> 32 with {kernel}'),
                    near_duplicate_of_test_fraction=near,
                    records=kept, excluded=excluded,
                    counts=dict(generated=len(rows), kept=len(kept), excluded=reasons),
                    nearest_test_rmse_quantiles=[float(q) for q in np.quantile(distance, [0, .01, .5])],
                    low_level=low_level(np.stack([p for p, r in zip(small, rows)])),
                    competition_test_labels_accessed=False,
                    competition_test_images_read='pixels only: duplicate exclusion and low-level statistics',
                    warehouse_data_used=False, all_ids_unique=True, all_pixels32_unique=True)
    if previous_sources:
        manifest['generated_duplicate_exclusion_sources'] = previous_sources
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(manifest['counts'], kernel=kernel, low_level=manifest['low_level'],
                          nearest_test_rmse_quantiles=manifest['nearest_test_rmse_quantiles'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['kernel', 'build'])
    parser.add_argument('--gen-roots', type=Path, nargs='+', required=True)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot-end', type=int, default=8)
    parser.add_argument('--kernel-file', type=Path)
    parser.add_argument('--role', choices=['dev', 'confirmation', 'pilot', 'train'])
    parser.add_argument('--expected', type=int, default=8320)
    parser.add_argument('--sources', nargs='+', choices=['hires', 'native'], default=['hires'],
                        help='kernel stage: reduce from the saved 224px file and/or from the generated size.')
    parser.add_argument('--max-index', type=int, help='build stage: use only images with index below this.')
    parser.add_argument('--exclude-manifest', type=Path, action='append', default=[],
                        help='Build stage: exclude exact pixels from an earlier generated manifest. Repeatable.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'kernel':
        choose_kernel(args)
    else:
        assert args.kernel_file and args.role and len(args.gen_roots) == 1
        build(args)


if __name__ == '__main__':
    main()
