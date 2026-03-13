import pandas as pd
import numpy as np
from ipdb import set_trace as bp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

FONT_SIZE = 12
BRACKET_COLOR = '#666666'
PLOT_TYPE = 'box'   # 'box' or 'violin'
USE_BEESWARM = False  # overlay beeswarm on box or violin when True

def get_significance_stars(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'ns'

df = pd.read_csv('../data/inference_v6emb_3920_all.csv')
df['date'] = pd.to_datetime(df['date'])
# Convert logits to probabilities
df['pred'] = 1 / (1 + np.exp(-df['pred']))

start1 = pd.Timestamp('2020-12-15')
start2 = pd.Timestamp('2020-12-22')
start3 = pd.Timestamp('2021-01-28')
start = start1 - pd.Timedelta(days=23)
end = start3 + pd.Timedelta(days=21)
section0 = df[(df['pid'] == '1007') & (df['date'] > start) &  (df['date'] <= start1)]  # Control
section1 = df[(df['pid'] == '1007') & (df['date'] > start1) & (df['date'] <= start2)]   # 37.5mg
section2 = df[(df['pid'] == '1007') & (df['date'] > start2) & (df['date'] <= start3)]  # 75mg
section3 = df[(df['pid'] == '1007') & (df['date'] > start3) & (df['date'] <= end)]     # 150mg

# Build long-format dataframe for boxplot
order = ['Before Start\n(0mg)', 'Week 1\n(37.5 mg)', 'Weeks 2-6\n(75 mg)', 'Post Week 6\n(150 mg)']
sections = [section0, section1, section2, section3]
data = []
labels = []
for sec, label in zip(sections, order):
    if len(sec) > 0:
        data.extend(sec['pred'].values)
        labels.extend([label] * len(sec))
df_plot = pd.DataFrame({'section': labels, 'pred': data})

fig, ax = plt.subplots(figsize=(6, 4))
if PLOT_TYPE == 'box':
    sns.boxplot(x='section', y='pred', data=df_plot, order=order, showfliers=False, palette='Greens', ax=ax)
elif PLOT_TYPE == 'violin':
    sns.violinplot(x='section', y='pred', data=df_plot, order=order, palette='Greens', ax=ax)
else:
    raise ValueError("PLOT_TYPE must be 'box' or 'violin'")
if USE_BEESWARM:
    sns.swarmplot(x='section', y='pred', data=df_plot, order=order, color='black', ax=ax, size=4, alpha=0.7)
ax.set_ylabel('Model Score', fontsize=FONT_SIZE)
ax.set_xlabel('Venlafaxine dose', fontsize=FONT_SIZE)
ax.tick_params(axis='x', labelsize=FONT_SIZE)
ax.tick_params(axis='y', labelsize=FONT_SIZE)

box_positions = ax.get_xticks()
control_values = df_plot[df_plot['section'] == 'Before Start\n(0mg)']['pred'].values
control_idx = 0

# N labels above medians
for i, sec_label in enumerate(order):
    subset = df_plot[df_plot['section'] == sec_label]
    if len(subset) > 0:
        median = subset['pred'].median()
        ax.text(i, median , f'N={len(subset)}', ha='center', va='bottom', fontsize=FONT_SIZE - 1)

# Significance brackets: each dose vs Control
bracket_offset = 0
for i, sec_label in enumerate(order):
    if sec_label == 'Before Start\n(0mg)':
        continue
    cohort_values = df_plot[df_plot['section'] == sec_label]['pred'].values
    if len(cohort_values) == 0 or len(control_values) == 0:
        continue
    ttest = ttest_ind(cohort_values, control_values)
    sig_stars = get_significance_stars(ttest.pvalue)
    max_y = max(control_values.max(), cohort_values.max())
    bracket_y = max_y + 0.02 +  (bracket_offset * 0.02)
    if sec_label == 'Post Week 6\n(150 mg)':
        bracket_y += 0.04
    x1 = box_positions[control_idx]
    x2 = box_positions[i]
    # Shorten bracket slightly so it doesn't touch boxes
    x1_off = x1 + 0.12
    x2_off = x2 - 0.12
    ax.plot([x1_off, x2_off], [bracket_y, bracket_y], color=BRACKET_COLOR, linewidth=1)
    ax.plot([x1_off, x1_off], [bracket_y - 0.01, bracket_y], color=BRACKET_COLOR, linewidth=1)
    ax.plot([x2_off, x2_off], [bracket_y - 0.01, bracket_y], color=BRACKET_COLOR, linewidth=1)
    ax.text((x1_off + x2_off) / 2, bracket_y - 0.02, sig_stars, ha='center', va='bottom', fontsize=FONT_SIZE, color='black')

ax.set_ylim(0, min(1.05, ax.get_ylim()[1] + 0.2))
plt.title('Patient Loading on Venlafaxine')
plt.tight_layout()
plt.savefig(f'venlafaxine_loading_visualization_{PLOT_TYPE}_{USE_BEESWARM}.png', dpi=300, bbox_inches='tight')
# plt.show()
