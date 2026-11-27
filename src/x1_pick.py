"""Rank the X1 pilots for the fallback generation queue of run_x1.sh.

Prints "<generator> <end seed index> <yes|no: keep native files>". The rank is
the number of passed checks, then the mixing rate. This only keeps the GPUs
busy; the decision on the recipe follows the rule written in EXPERIMENTS.md.
"""

import json
from pathlib import Path

OUT = Path('runs/x1_signature_20261116')
END = dict(sd_turbo=200, sd21_base=200, sd21_768=120)  # 768px images are about four times slower

ranked = []
for name in END:
    path = OUT/f'{name}.json'
    if path.exists():
        report = json.loads(path.read_text())
        ranked.append((sum(report['checks'].values()), report['mixing']['relative_to_ideal'], name))
if ranked:
    name = max(ranked)[2]
    kernel = json.loads(Path(f'external_data/x1_kernel_{name}.json').read_text())['chosen']
    print(name, END[name], 'yes' if kernel.startswith('native:') else 'no')
