"""Select and download a small, fixed DomainNet support/calibration/style holdout."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

MAPPING = {'airplane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'dog': 5,
           'frog': 6, 'horse': 7, 'cruise_ship': 8, 'truck': 9}
SOURCES = [('support', 'clipart', 'train', 64),
           ('support', 'painting', 'train', 64),
           ('calibration', 'clipart', 'test', 32),
           ('calibration', 'painting', 'test', 32),
           ('holdout', 'sketch', 'test', 64)]
BASE = 'https://csr.bu.edu/ftp/visda/2019/multi-source/'


def archive_url(domain):
    # The cleaned clipart/painting links differ from the original-layout archives.
    return BASE + ('groundtruth/' if domain in ['clipart', 'painting'] else '') + domain + '.zip'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def select(indices):
    records, index_hashes = [], {}
    for role, domain, split, cap in SOURCES:
        filename = f'{domain}_{split}.txt'
        raw = (indices / filename).read_bytes()
        index_hashes[filename] = digest(raw)
        for category, label in MAPPING.items():
            paths = sorted(line.split()[0] for line in raw.decode().splitlines()
                           if line.split()[0].split('/')[1] == category)
            seed = int(digest(f'20261113/{filename}/{category}'.encode())[:16], 16)
            random.Random(seed).shuffle(paths)
            for path in paths[:cap]:
                records.append(dict(id=path, path='images/' + path, role=role,
                                    domain=domain, category=category, label=label,
                                    source_index=filename, archive=archive_url(domain)))
    return records, index_hashes


def check_image(payload):
    with Image.open(io.BytesIO(payload)) as image:
        rgb = image.convert('RGB')
        small = rgb.resize((32, 32), Image.Resampling.BICUBIC)
        return digest(np.asarray(small).tobytes())


def download_group(root, domain, records):
    from remotezip import RemoteZip
    with RemoteZip(archive_url(domain), initial_buffer_size=2**22, timeout=60) as archive:
        names = set(archive.namelist())
        for i, record in enumerate(records, 1):
            name = record['id']
            if name not in names:
                raise ValueError('Official index path missing from cleaned archive: '+name)
            for retry in range(3):
                try:
                    payload = archive.read(name)
                    check_image(payload)
                    break
                except Exception:
                    if retry == 2:
                        raise
                    time.sleep(1)
            target = root/record['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            if i == 1 or i % 25 == 0 or i == len(records):
                print(domain, 'worker progress', i, '/', len(records), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path, help='Provided competition data for train/validation duplicate exclusion.')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    records, hashes = select(args.root / 'indices')
    planned = dict(seed=20261113, mapping=MAPPING, absent_class='deer',
                   source='official cleaned DomainNet', source_page='https://ai.bu.edu/M3SDA/',
                   index_sha256=hashes, records=records)
    plan = args.root / 'selection.json'
    if plan.exists() and json.loads(plan.read_text()) != planned:
        raise ValueError('Existing selection differs; use a new dataset root.')
    write_json(plan, planned)
    print('Selected', len(records), 'images before duplicate checks.', flush=True)
    if args.plan_only:
        return
    if not args.data:
        raise ValueError('--data is needed for duplicate checks.')
    for domain in ['clipart', 'painting', 'sketch']:
        selected = [r for r in records if r['domain'] == domain]
        pending = []
        for record in selected:
            path = args.root/record['path']
            try:
                check_image(path.read_bytes())
            except (OSError, ValueError):
                pending.append(record)
        if pending:
            # HTTP ranges avoid downloading gigabytes of unused categories.
            if not 1 <= args.workers <= 6:
                raise ValueError('Use between one and six download connections.')
            chunks = [pending[i::args.workers] for i in range(args.workers)]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                jobs = [pool.submit(download_group,args.root,domain,chunk) for chunk in chunks if chunk]
                for job in jobs:
                    job.result()
    competition_hashes = set()
    for split in ['train', 'validation']:
        with np.load(args.data/f'{split}.npz') as data:
            competition_hashes.update(digest(np.asarray(x, dtype=np.uint8).tobytes())
                                      for x in data['images'])
    kept, excluded, seen = [], [], set()
    # Give the untouched style priority over tuning/support in duplicate exclusion.
    for role in ['holdout', 'calibration', 'support']:
        for record in records:
            if record['role'] != role:
                continue
            payload = (args.root/record['path']).read_bytes()
            pixels = check_image(payload)
            record = dict(record, file_sha256=digest(payload), pixels32_sha256=pixels)
            reason = ('provided_train_or_validation_duplicate' if pixels in competition_hashes
                      else 'external_pixel_duplicate' if pixels in seen else None)
            if reason:
                excluded.append(dict(record, reason=reason))
            else:
                kept.append(record)
                seen.add(pixels)
    counts = {}
    for r in kept:
        key = f"{r['role']}/{r['domain']}/{r['category']}"
        counts[key] = counts.get(key, 0)+1
    expected = {(role, domain, label) for role, domain, _, _ in SOURCES for label in MAPPING.values()}
    assert {(r['role'],r['domain'],r['label']) for r in kept} == expected
    result = dict(planned, records=kept, excluded=excluded, counts=counts,
                  resize='PIL RGB bicubic 32x32 before encoder transform',
                  competition_test_images_or_labels_read=False,
                  all_paths_unique=len({r['id'] for r in kept}) == len(kept),
                  all_pixels32_unique=len({r['pixels32_sha256'] for r in kept}) == len(kept))
    write_json(args.root/'manifest.json', result)
    print('Ready', len(kept), 'images;', len(excluded), 'excluded.', flush=True)


if __name__ == '__main__':
    main()
