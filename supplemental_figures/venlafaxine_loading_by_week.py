import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FONT_SIZE = 12
N_WEEKS = 12
WEEK_LABEL_OFFSET = 3  # week 1 → "Week -2", week 2 → "Week -1", week 3 → "Week 0", etc.

# Explicit start of each week (Week -2 through Week 9). start1/start2/start3 are week boundaries.
start1 = pd.Timestamp('2020-12-15')   # start of Week 1 (37.5 mg)
start2 = pd.Timestamp('2020-12-22')   # start of Week 2 (75 mg)
start3 = pd.Timestamp('2021-01-28')   # start of Week 7 (150 mg)
# Week boundaries: [inclusive start, next week start). Last week needs an end.
WEEK_STARTS = [
    pd.Timestamp('2020-11-25'),   # Week -2 start (0 mg)
    pd.Timestamp('2020-12-02'),   # Week -1 start (0 mg)
    pd.Timestamp('2020-12-09'),   # Week  0 start (0 mg)
    start1,                       # Week  1 start (37.5 mg)
    start2,                       # Week  2 start (75 mg)
    pd.Timestamp('2020-12-29'),  # Week  3 start (75 mg)
    pd.Timestamp('2021-01-05'),   # Week  4 start (75 mg)
    pd.Timestamp('2021-01-12'),   # Week  5 start (75 mg)
    pd.Timestamp('2021-01-19'),   # Week  6 start (75 mg)
    start3,                       # Week  7 start (150 mg)
    pd.Timestamp('2021-02-04'),   # Week  8 start (150 mg)
    pd.Timestamp('2021-02-11'),   # Week  9 start (150 mg)
    pd.Timestamp('2021-02-18'),   # end of Week 9 (exclusive upper bound)
]
assert len(WEEK_STARTS) == N_WEEKS + 1, "N_WEEKS + 1 boundaries required"

df = pd.read_csv('../data/inference_v6emb_3920_all.csv')
df['date'] = pd.to_datetime(df['date'])
df['pred'] = 1 / (1 + np.exp(-df['pred']))

start = WEEK_STARTS[0]
end = WEEK_STARTS[-1]
df = df[(df['pid'] == '1007') & (df['date'] >= start) & (df['date'] < end)].copy()

# Assign week number (1 to N_WEEKS) by which [WEEK_STARTS[i], WEEK_STARTS[i+1]) interval date falls in
def assign_week(d):
    for w in range(1, N_WEEKS + 1):
        if WEEK_STARTS[w - 1] <= d < WEEK_STARTS[w]:
            return w
    return N_WEEKS

df['week'] = df['date'].apply(assign_week)

# Dosage per week: weeks 1-3 = 0mg, week 4 = 37.5mg, weeks 5-9 = 75mg, weeks 10-12 = 150mg
def dosage_for_week(w):
    if w <= 3:
        return '0 mg'
    if w == 4:
        return '37.5 mg'
    if w <= 9:
        return '75 mg'
    return '150 mg'

df['dosage'] = df['week'].map(dosage_for_week)

# Palette: one color per dosage (same color for all weeks with that dosage)
# 0mg = white, 37.5 = light green, 75 = medium green, 150 = dark green (more separation in hues)
PALETTE_DOSAGE = {'0 mg': 'white', '37.5 mg': '#a8e6a1', '75 mg': '#2e7d32', '150 mg': '#0d4d0d'}
# Build list of 12 colors for the 12 boxes (by week → dosage)
week_order = list(range(1, N_WEEKS + 1))
palette_weeks = [PALETTE_DOSAGE[dosage_for_week(w)] for w in week_order]

df_plot = df[['week', 'pred', 'dosage']].copy()
df_plot['week'] = df_plot['week'].astype(int)

fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(
    x='week', y='pred', data=df_plot, order=week_order,
    showfliers=False, palette=palette_weeks, ax=ax
)
ax.set_ylabel('Model Score', fontsize=FONT_SIZE)
ax.set_xlabel('Week from start of Venlafaxine', fontsize=FONT_SIZE)
ax.tick_params(axis='x', labelsize=FONT_SIZE - 1)
ax.tick_params(axis='y', labelsize=FONT_SIZE)
# Tick labels: Week -2, Week -1, Week 0, Week 1, ...
ax.set_xticklabels([f'Week {w - WEEK_LABEL_OFFSET - (1 if w < 4 else 0)}' for w in week_order])

# Vertical dashed line between Week 0 and Week 1 (between box index 2 and 3)
ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=1)

# Legend: dosage colors
legend_patches = [
    mpatches.Patch(color=PALETTE_DOSAGE['0 mg'], label='0 mg'),
    mpatches.Patch(color=PALETTE_DOSAGE['37.5 mg'], label='37.5 mg'),
    mpatches.Patch(color=PALETTE_DOSAGE['75 mg'], label='75 mg'),
    mpatches.Patch(color=PALETTE_DOSAGE['150 mg'], label='150 mg'),
]
leg = ax.legend(handles=legend_patches, fontsize=FONT_SIZE - 1, loc='lower right', title='Dosage')
leg.get_title().set_fontweight('bold')

# N labels above medians per week
# for i, w in enumerate(week_order):
#     subset = df_plot[df_plot['week'] == w]
#     if len(subset) > 0:
#         median = subset['pred'].median()
#         ax.text(i, median, f'N={len(subset)}', ha='center', va='bottom', fontsize=FONT_SIZE - 2)

ax.set_ylim(0, min(1.05, ax.get_ylim()[1] + 0.2))
ax.set_title('Patient Loading on Venlafaxine (by week)')
plt.tight_layout()
plt.savefig('venlafaxine_loading_by_week_v2.png', dpi=300, bbox_inches='tight')
