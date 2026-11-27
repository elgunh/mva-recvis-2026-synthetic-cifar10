"""Generate labelled style images with an open-weight text-to-image model.

Settings of the first three generators are fixed in SYNTH_STYLE_PLAN.md. Until
16 November 2026 I used only these three (sd15, kandinsky22, sdxl_turbo). From
that date I also generate with Stable Diffusion 2.x (sd_turbo, sd21_base,
sd21_768; see EXPERIMENTS.md), the model named in the CIFAR-10-Warehouse paper.
Only my own generated images are used; the CIFAR-10-Warehouse data is never
downloaded or used. The run is resumable: an image
with a saved record is skipped.
"""

import argparse
import hashlib
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CLASS_NAMES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
STYLES = (
    ('cage', 'photo of {} in a cage'),
    ('table', 'photo of {} on a table'),
    ('box', 'photo of {} painted on a box'),
    ('tshirt', 'photo of {} painted on a t-shirt'),
    ('sketch', 'sketch of {}'),
    ('messy', 'photo of {} with messy background'),
    ('toy', 'photo of toy model of {}'),
    ('model3d', 'photo of 3D model of {}'),
    ('origami', 'photo of origami of {}'),
    ('sculpture', 'photo of sculpture of {}'),
    ('lego', 'photo of lego model of {}'),
    ('wooden', 'photo of wooden model of {}'),
    ('oil', 'oil painting of {}'),
)
MODELS = {
    'sd15': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
    'kandinsky22': 'kandinsky-community/kandinsky-2-2-decoder',
    'kandinsky22_prior': 'kandinsky-community/kandinsky-2-2-prior',
    'sdxl_turbo': 'stabilityai/sdxl-turbo',
    'sd_turbo': 'stabilityai/sd-turbo',
    'sd21_base': 'sd2-community/stable-diffusion-2-1-base',
    'sd21_768': 'sd2-community/stable-diffusion-2-1',
}
SD2_FAMILY = ('sd_turbo', 'sd21_base', 'sd21_768')
SETTINGS = {
    'sd15': dict(height=512, width=512, steps=20, guidance=7.5, batch=8,
                 scheduler='DPMSolverMultistepScheduler(use_karras_sigmas=True)'),
    'kandinsky22': dict(height=512, width=512, steps=50, guidance=4.0, batch=4,
                        prior_steps=25, prior_guidance=4.0, scheduler='shipped DDPMScheduler'),
    'sdxl_turbo': dict(height=512, width=512, steps=4, guidance=0.0, batch=4, vae_slicing=True,
                       scheduler='shipped EulerAncestralDiscreteScheduler'),
    # The Warehouse paper names only the model (Stable-diffusion-2-1) and the prompts; sampler,
    # steps, guidance and size are not stated. These are common defaults, not recovered settings.
    'sd_turbo': dict(height=512, width=512, steps=4, guidance=0.0, batch=8,
                     scheduler='shipped scheduler'),
    'sd21_base': dict(height=512, width=512, steps=25, guidance=7.5, batch=8,
                      scheduler='DPMSolverMultistepScheduler (model card example)'),
    'sd21_768': dict(height=768, width=768, steps=25, guidance=7.5, batch=2, vae_slicing=True,
                     scheduler='DPMSolverMultistepScheduler (model card example)'),
}


def seed_of(generator, style, name, index):
    text = f'{generator}/{style}/{name}/{index}'
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def jobs(generator, start, end):
    for index in range(start, end):
        for style, template in STYLES:
            for label, name in enumerate(CLASS_NAMES):
                yield dict(id=f'{generator}/{style}/{name}/{index:04d}', generator=generator,
                           style=style, category=name, label=label, index=index,
                           prompt=template.format(name),
                           seed=seed_of(generator, style, name, index),
                           path=f'hires/{style}/{name}/{index:04d}.png')


def cpu_generators(batch):
    return [torch.Generator('cpu').manual_seed(j['seed']) for j in batch]


def revision(repo):
    try:  # commit of the cached snapshot; works offline
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo, local_files_only=True)).name
    except Exception as error:  # recorded, not fatal
        return f'unavailable: {type(error).__name__}'


def build(generator, device, output):
    cfg = SETTINGS[generator]
    if generator == 'sd15':
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        pipe = StableDiffusionPipeline.from_pretrained(
            MODELS['sd15'], torch_dtype=torch.float16, variant='fp16', use_safetensors=True,
            safety_checker=None, requires_safety_checker=False)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True)
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)

        def run(batch):
            return pipe([j['prompt'] for j in batch], height=cfg['height'], width=cfg['width'],
                        num_inference_steps=cfg['steps'], guidance_scale=cfg['guidance'],
                        generator=cpu_generators(batch)).images
        return run
    if generator in SD2_FAMILY:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        pipe = StableDiffusionPipeline.from_pretrained(
            MODELS[generator], torch_dtype=torch.float16, variant='fp16', use_safetensors=True,
            safety_checker=None, requires_safety_checker=False)
        if generator != 'sd_turbo':  # keeps the prediction type of the checkpoint (v-prediction at 768)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)
        if cfg.get('vae_slicing'):
            pipe.enable_vae_slicing()

        def run(batch):
            return pipe([j['prompt'] for j in batch], height=cfg['height'], width=cfg['width'],
                        num_inference_steps=cfg['steps'], guidance_scale=cfg['guidance'],
                        generator=cpu_generators(batch)).images
        return run
    if generator == 'sdxl_turbo':
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODELS['sdxl_turbo'], torch_dtype=torch.float16, variant='fp16', use_safetensors=True)
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)
        # The VAE is upcast to float32 for decoding; batch 8 ran out of memory on 11 GB.
        pipe.enable_vae_slicing()

        def run(batch):
            return pipe([j['prompt'] for j in batch], height=cfg['height'], width=cfg['width'],
                        num_inference_steps=cfg['steps'], guidance_scale=cfg['guidance'],
                        generator=cpu_generators(batch)).images
        return run
    if generator == 'kandinsky22':
        from diffusers import KandinskyV22Pipeline
        with np.load(output/'prior_embeddings.npz') as saved:
            lookup = {i: e for i, e in zip(saved['ids'].tolist(), saved['image_embeds'])}
            zero = torch.as_tensor(saved['negative_image_embeds'])
        pipe = KandinskyV22Pipeline.from_pretrained(MODELS['kandinsky22'], torch_dtype=torch.float16)
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)

        def run(batch):
            embeds = torch.as_tensor(np.stack([lookup[j['id']] for j in batch]))
            embeds = embeds.to(device, torch.float16)
            negative = zero.to(device, torch.float16).expand(len(batch), -1)
            return pipe(image_embeds=embeds, negative_image_embeds=negative,
                        height=cfg['height'], width=cfg['width'],
                        num_inference_steps=cfg['steps'], guidance_scale=cfg['guidance'],
                        generator=cpu_generators(batch)).images
        return run
    raise ValueError(generator)


@torch.inference_mode()
def kandinsky_prior(args, device):
    """Stage 1 for Kandinsky: one image embedding per image seed, saved to disk."""
    from diffusers import KandinskyV22PriorPipeline
    cfg = SETTINGS['kandinsky22']
    target = args.output/'prior_embeddings.npz'
    if target.exists():
        raise FileExistsError('Prior embeddings already exist; they are never overwritten.')
    pipe = KandinskyV22PriorPipeline.from_pretrained(
        MODELS['kandinsky22_prior'], torch_dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    todo = list(jobs('kandinsky22', args.start, args.end))
    ids, embeds, zero, started = [], [], None, time.time()
    for offset in range(0, len(todo), 16):
        batch = todo[offset:offset+16]
        image_embeds, negative = pipe(
            [j['prompt'] for j in batch], num_inference_steps=cfg['prior_steps'],
            guidance_scale=cfg['prior_guidance'], generator=cpu_generators(batch),
            return_dict=False)
        assert torch.isfinite(image_embeds).all() and torch.isfinite(negative).all()
        if zero is None:
            zero = negative[:1].float().cpu()
        assert torch.allclose(negative.float().cpu(), zero.expand(len(batch), -1), atol=1e-3)
        ids += [j['id'] for j in batch]
        embeds.append(image_embeds.float().cpu().numpy())
        if offset == 0 or (offset//16) % 50 == 0 or offset+16 >= len(todo):
            print(f'prior {offset+len(batch)}/{len(todo)} {time.time()-started:.1f}s', flush=True)
    np.savez(target, ids=np.array(ids), image_embeds=np.concatenate(embeds),
             negative_image_embeds=zero.numpy())
    print('Prior embeddings saved', target, flush=True)


def save_image(image, path, native=None, small=None):
    image = image.convert('RGB')
    if native is not None:  # pilot only: the image at its generated size, for the choice of the 32px kernel
        native.parent.mkdir(parents=True, exist_ok=True)
        image.save(native, format='PNG')
    if small is not None:  # (path, kernel): the 32px image, reduced from the generated size with the frozen kernel
        from prepare_synth_manifest import downsample
        small[0].parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(downsample(image, small[1])).save(small[0], format='PNG')
    small = image.resize((224, 224), Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    small.save(buffer, format='PNG')
    data = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    pixels = np.asarray(small, dtype=np.float32)/255
    return hashlib.sha256(data).hexdigest(), float(pixels.std())


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generator', choices=list(SETTINGS), required=True)
    parser.add_argument('--stage', choices=['prior', 'images'], default='images')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=64)
    parser.add_argument('--shard', type=int, nargs=2, default=[0, 1], metavar=('INDEX', 'COUNT'))
    parser.add_argument('--limit', type=int, help='Stop after this many new images (preflight).')
    parser.add_argument('--reserve-disk-bytes', type=int, default=0,
                        help='Stop cleanly if free space falls below this amount.')
    parser.add_argument('--save-native', action='store_true',
                        help='Also keep each image at its generated size under native/ (pilot runs).')
    parser.add_argument('--kernel-file', type=Path,
                        help='Frozen 32px kernel choice (from the generated size): also save the 32px image under small/.')
    args = parser.parse_args()
    if args.reserve_disk_bytes < 0:
        parser.error('--reserve-disk-bytes must be nonnegative.')
    chosen = kernel = None
    if args.kernel_file:
        from prepare_synth_manifest import split_choice
        chosen = json.loads(args.kernel_file.read_text())['chosen']
        source, kernel = split_choice(chosen)
        assert source == 'native', 'Only a kernel applied to the generated size can be used at generation time.'
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda')
    if args.stage == 'prior':
        assert args.generator == 'kandinsky22'
        return kandinsky_prior(args, device)
    done = set()
    for log in args.output.glob('records_*.jsonl'):
        done.update(json.loads(line)['id'] for line in log.read_text().splitlines() if line)
    todo = [j for n, j in enumerate(jobs(args.generator, args.start, args.end))
            if n % args.shard[1] == args.shard[0] and j['id'] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f'{args.generator}: {len(todo)} images to generate, {len(done)} already recorded', flush=True)
    if not todo:
        return
    settings_file = args.output/'settings.json'
    settings = dict(generator=args.generator, model=MODELS[args.generator],
                    model_revision=revision(MODELS[args.generator]), **SETTINGS[args.generator],
                    saved_size=224, saved_resize='PIL bicubic', saved_native=bool(args.save_native),
                    seed_rule='sha256(generator/style/class/index)[:8]',
                    templates=dict(STYLES), class_names=CLASS_NAMES, torch=torch.__version__)
    if args.generator == 'kandinsky22':
        settings.update(prior_model=MODELS['kandinsky22_prior'],
                        prior_revision=revision(MODELS['kandinsky22_prior']))
    if not settings_file.exists():
        settings_file.write_text(json.dumps(settings, indent=2)+'\n')
    run = build(args.generator, device, args.output)
    batch_size = SETTINGS[args.generator]['batch']
    log = (args.output/f'records_{args.shard[0]}of{args.shard[1]}.jsonl').open('a')
    started, flat = time.time(), 0
    torch.cuda.reset_peak_memory_stats()
    for offset in range(0, len(todo), batch_size):
        if (args.reserve_disk_bytes and (offset // batch_size) % 25 == 0
                and shutil.disk_usage(args.output).free < args.reserve_disk_bytes):
            log.close()
            raise SystemExit('Generation stopped at the requested free-space reserve.')
        batch = todo[offset:offset+batch_size]
        images = run(batch)
        for job, image in zip(batch, images):
            native = 'native'+job['path'][len('hires'):] if args.save_native else None
            small = 'small'+job['path'][len('hires'):] if kernel else None
            digest, spread = save_image(image, args.output/job['path'],
                                        args.output/native if native else None,
                                        (args.output/small, kernel) if small else None)
            flat += spread < .02
            record = dict(job, file_sha256=digest, pixel_std=round(spread, 5))
            if native:
                record['native_path'] = native
            if small:
                record.update(small_path=small, small_kernel=chosen)
            log.write(json.dumps(record)+'\n')
        log.flush()
        count = offset+len(batch)
        if offset == 0 or (offset//batch_size) % 25 == 0 or count == len(todo):
            elapsed = time.time()-started
            print(f'{args.generator}: {count}/{len(todo)} {elapsed:.0f}s '
                  f'{elapsed/count:.2f}s/img flat={flat} '
                  f'peak={torch.cuda.max_memory_allocated()/2**20:.0f}MB', flush=True)
    print('Completed', args.generator, flush=True)


if __name__ == '__main__':
    main()
