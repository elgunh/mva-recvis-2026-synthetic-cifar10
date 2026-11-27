"""Figure 1 of the report: public accuracy against the number of labelled generated images.

All numbers are the public scores recorded in submission_ledger.json and the held-out error
counts recorded in EXPERIMENTS.md (student alone, 3,100 kept images of seeds 0-23).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

images = [12417, 20953, 48636, 95433, 161656, 194495]
public = [98.701, 98.897, 99.066, 99.223, 99.321, 99.406]
held_out_errors = [45, 41, 29, 29, 20, 17]

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'font.size': 7, 'axes.linewidth': .5, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(3.25, 1.05))
ax.axhline(98.048, color='0.45', lw=.7, ls='--')
ax.text(11200, 98.075, 'Stage 2 (no generated images): 98.05', color='0.3', fontsize=6, va='bottom')
ax.plot(images, public, '-o', color='#1f4e79', lw=1.1, ms=3.2)
for x, y, e in zip(images, public, held_out_errors):
    ax.annotate(str(e), (x, y), textcoords='offset points', xytext=(0, -9), ha='center', fontsize=6, color='#8a3b12')
ax.text(104000, 98.62, 'brown: held-out errors\nof the student (of 3,100)', fontsize=6, color='#8a3b12', ha='center', va='center')
ax.set_xscale('log')
ax.set_xticks([12500, 25000, 50000, 100000, 200000])
ax.set_xticklabels(['12.5k', '25k', '50k', '100k', '200k'])
ax.minorticks_off()
ax.set_xlim(10500, 235000)
ax.set_ylim(97.95, 99.52)
ax.set_yticks([98.0, 98.5, 99.0, 99.5])
ax.set_xlabel('labelled generated training images', labelpad=1.5)
ax.set_ylabel('public accuracy (%)', labelpad=2)
ax.grid(axis='y', lw=.3, color='0.85')
ax.tick_params(width=.5, length=2.5, pad=1.5)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
fig.subplots_adjust(left=.135, right=.985, top=.97, bottom=.3)
fig.savefig('fig_scaling.pdf')
fig.savefig('fig_scaling.png', dpi=300)
print('saved')
