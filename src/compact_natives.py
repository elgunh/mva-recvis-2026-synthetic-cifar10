"""Replace the large generated-size PNG files of a generator by their 32px images.

For every record that has a native file and no saved 32px image, the 32px image
is computed with the frozen kernel choice (the same function as in the manifest
build) and saved under small/. The new records go to records_zz_small.jsonl,
which is read last and so replaces the old records. With --check MANIFEST every
image that is in that manifest must equal the manifest pixels. Native files are
deleted only with --delete and only if every check passed. The images can be
generated again from their seeds.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import prepare_synth_manifest as P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--kernel-file', type=Path, required=True)
    parser.add_argument('--check', type=Path, help='Manifest directory whose 32px files must be reproduced.')
    parser.add_argument('--delete', action='store_true')
    args = parser.parse_args()
    chosen = json.loads(args.kernel_file.read_text())['chosen']
    source, kernel = P.split_choice(chosen)
    assert source == 'native'
    expected = {}
    if args.check:
        manifest = json.loads((args.check/'manifest.json').read_text())
        assert manifest['kernel'] == kernel and manifest.get('kernel_source') == 'native'
        expected = {r['id']: args.check/r['path'] for r in manifest['records']}
    rows = [r for r in P.records_of(args.root) if 'native_path' in r and r.get('small_kernel') != chosen]
    print('records to convert', len(rows), 'kernel', chosen, flush=True)
    converted, checked, natives = [], 0, []
    for n, row in enumerate(rows, 1):
        native = args.root/row['native_path']
        pixels = P.downsample(Image.open(native).convert('RGB'), kernel)
        small = 'small'+row['path'][len('hires'):]
        target = args.root/small
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(target, format='PNG')
        with Image.open(target) as saved:
            assert np.array_equal(np.asarray(saved.convert('RGB')), pixels)
        if row['id'] in expected:
            with Image.open(expected[row['id']]) as reference:
                assert np.array_equal(np.asarray(reference.convert('RGB')), pixels), row['id']
            checked += 1
        new = dict(row, small_path=small, small_kernel=chosen)
        del new['native_path']
        converted.append(new)
        natives.append(native)
        if n % 2000 == 0 or n == len(rows):
            print(f'{n}/{len(rows)} converted, {checked} equal to the manifest', flush=True)
    if not converted:
        return
    assert not args.check or checked == len(expected), (checked, len(expected))
    with (args.root/'records_zz_small.jsonl').open('a') as log:
        for row in converted:
            log.write(json.dumps(row)+'\n')
    after = {r['id']: r for r in P.records_of(args.root)}
    assert all(after[r['id']].get('small_kernel') == chosen for r in converted)
    print('records rewritten', len(converted), flush=True)
    if args.delete:
        for native in natives:
            native.unlink()
        print('native files deleted', len(natives), flush=True)


if __name__ == '__main__':
    main()
