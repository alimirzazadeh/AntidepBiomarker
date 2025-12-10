import pandas as pd
import numpy as np 
import os 
from ipdb import set_trace as bp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def get_significance_stars(pval):
    if pval < 1e-10:
        return '***'
    elif pval < 0.001:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'ns'
CSV_DIR = '../data/'
INFERENCE_FILE = os.path.join(CSV_DIR,'inference_v6emb_3920_all.csv')
ahis = pd.read_csv(os.path.join(CSV_DIR,'shhs_mros_cfs_wsc_ahi.csv'))
df = pd.read_csv(INFERENCE_FILE)

df['filename'] = df['filepath'].apply(lambda x: x.split('/')[-1])
df = df.groupby('filename').agg(
    lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0]
)
df_tax = pd.read_csv(os.path.join(CSV_DIR,'antidep_taxonomy_all_datasets_v6.csv'))
df = pd.merge(df, df_tax, on='filename', how='inner')
df['is_tca'] = df['taxonomy'].apply(lambda x: 1 if x.startswith('T') or ',T' in x else 0)
df['is_ntca'] = df['taxonomy'].apply(lambda x: 1 if x.startswith('N') or ',N' in x else 0)
df['is_snri'] = df['taxonomy'].apply(lambda x: 1 if x.startswith('NN') or ',NN' in x else 0)
df['is_ssri'] = df['taxonomy'].apply(lambda x: 1 if x.startswith('NS') or ',NS' in x else 0)
# df = df[(df['is_snri'] == 1) | (df['label'] == 0)]
df = df[(df['is_snri'] == 1) | (df['is_ssri'] == 1) | (df['is_ntca'] == 1) | (df['is_tca'] == 1) | (df['label'] == 0)]
# Convert logits to probabilities using sigmoid
df['pred'] = 1 / (1 + np.exp(-df['pred']))
df['pid'] = df.apply(
    lambda x: x['pid'][1:] if x['dataset'] in ['shhs', 'mros', 'wsc'] else x['pid'], 
    axis=1
)

print(df.shape)
print(ahis.shape)

df = df.merge(ahis, on='filename', how='inner')

print(df.shape)




df['osa_group'] = pd.cut(df['ahi'], bins=[0, 5, 15, 30, np.inf], labels=['Normal', 'Mild OSA', 'Moderate OSA', 'Severe OSA'])
df['Group'] = df['label'].apply(lambda x: 'Control' if x == 0 else 'Antidepressant')

# Filter to only include rows with valid predictions
df_plot = df[df['pred'].notna()].copy()

# Define order for OSA groups
osa_labels = ['Normal', 'Mild OSA', 'Moderate OSA', 'Severe OSA']

fig, ax = plt.subplots(figsize=(12, 6))

# Create boxplots using seaborn with hue for paired boxes
sns.boxplot(data=df_plot, x='osa_group', y='pred', hue='Group', 
            order=osa_labels, palette='Greens', ax=ax, showfliers=False)
# Remove legend
ax.legend_.remove()

# Add N labels above each boxplot (above the median line)
# Get the positions of the boxes
box_positions = ax.get_xticks()

for i, osa_label in enumerate(osa_labels):
    # Control box (left side of pair)
    subset_control = df_plot[(df_plot['osa_group'] == osa_label) & 
                             (df_plot['Group'] == 'Control')]['pred'].dropna()
    if len(subset_control) > 0:
        n = len(subset_control)
        median_val = subset_control.median()
        q1 = subset_control.quantile(0.25)
        q3 = subset_control.quantile(0.75)
        print(f'{osa_label} Control: Median={median_val:.2f}, IQR: [{q1:.2f}-{q3:.2f}]')
        # Position is box_positions[i] - 0.2 (offset for paired boxes)
        ax.text(box_positions[i] - 0.2, median_val + 0.01, f'N={n}', 
               ha='center', va='bottom', fontsize=9)
    
    # Antidepressant box (right side of pair)
    subset_antidep = df_plot[(df_plot['osa_group'] == osa_label) & 
                            (df_plot['Group'] == 'Antidepressant')]['pred'].dropna()
    if len(subset_antidep) > 0:
        n = len(subset_antidep)
        median_val = subset_antidep.median()
        q1 = subset_antidep.quantile(0.25)
        q3 = subset_antidep.quantile(0.75)
        print(f'{osa_label} Antidepressant: Median={median_val:.2f}, IQR: [{q1:.2f}-{q3:.2f}]')
        # Position is box_positions[i] + 0.2 (offset for paired boxes)
        ax.text(box_positions[i] + 0.2, median_val + 0.01, f'N={n}', 
               ha='center', va='bottom', fontsize=9)
    
    # Compute t-test between Control and Antidepressant for this bin
    if len(subset_control) > 0 and len(subset_antidep) > 0:
        ttest = ttest_ind(subset_control.values, subset_antidep.values)
        print(f'T-test p value for {osa_label} (Control vs Antidepressant): {ttest.pvalue:.4e}')
        
        # Draw significance bracket and stars
        # Get max y-value from both boxes to position bracket above
        max_y_control = subset_control.max() if len(subset_control) > 0 else 0
        max_y_antidep = subset_antidep.max() if len(subset_antidep) > 0 else 0
        max_y = max(max_y_control, max_y_antidep)
        
        # Position bracket slightly above the boxes
        bracket_y = max_y + 0.05
        sig_stars = get_significance_stars(ttest.pvalue)
        
        # Draw bracket
        x1 = box_positions[i] - 0.2
        x2 = box_positions[i] + 0.2
        x_center = box_positions[i]
        
        # Draw horizontal line
        ax.plot([x1, x2], [bracket_y, bracket_y], 'k-', linewidth=1)
        # Draw vertical lines at ends
        ax.plot([x1, x1], [bracket_y - 0.01, bracket_y], 'k-', linewidth=1)
        ax.plot([x2, x2], [bracket_y - 0.01, bracket_y], 'k-', linewidth=1)
        # Add significance stars
        ax.text(x_center, bracket_y + 0.01, sig_stars, ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Model Score', fontsize=12)
ax.set_xlabel('OSA Group', fontsize=12)
ax.set_ylim(0, 1.1)  # Increased to accommodate significance brackets
ax.grid(axis='y', alpha=0.3, linestyle='--')

def cohen_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.var(x) + np.var(y)) / 2)

# Additional t-tests comparing different OSA groups within Antidepressant group
normal_antidep = df_plot[(df_plot['osa_group'] == 'Normal') & (df_plot['Group'] == 'Antidepressant')]['pred'].dropna()
severe_antidep = df_plot[(df_plot['osa_group'] == 'Severe OSA') & (df_plot['Group'] == 'Antidepressant')]['pred'].dropna()
mild_antidep = df_plot[(df_plot['osa_group'] == 'Mild OSA') & (df_plot['Group'] == 'Antidepressant')]['pred'].dropna()
moderate_antidep = df_plot[(df_plot['osa_group'] == 'Moderate OSA') & (df_plot['Group'] == 'Antidepressant')]['pred'].dropna()

if len(severe_antidep) > 0 and len(normal_antidep) > 0:
    t_stat, p_value = ttest_ind(severe_antidep, normal_antidep)
    print(f'Severe vs Normal (Antidepressant) t-statistic: {t_stat:.2f}, p-value: {p_value:.4e}')
if len(mild_antidep) > 0 and len(normal_antidep) > 0:
    t_stat, p_value = ttest_ind(mild_antidep, normal_antidep)
    print(f'Mild vs Normal (Antidepressant) t-statistic: {t_stat:.2f}, p-value: {p_value:.4e}')
if len(moderate_antidep) > 0 and len(normal_antidep) > 0:
    t_stat, p_value = ttest_ind(moderate_antidep, normal_antidep)
    print(f'Moderate vs Normal (Antidepressant) t-statistic: {t_stat:.2f}, p-value: {p_value:.4e}')

# Compare Antidepressant groups to Normal Control
normal_control = df_plot[(df_plot['osa_group'] == 'Normal') & (df_plot['Group'] == 'Control')]['pred'].dropna()
if len(severe_antidep) > 0 and len(normal_control) > 0:
    t_stat, p_value = ttest_ind(severe_antidep, normal_control)
    print(f'Severe OSA Antidepressant vs Normal Control t-statistic: {t_stat:.2f}, p-value: {p_value:.4e}')
if len(mild_antidep) > 0 and len(normal_control) > 0:
    t_stat, p_value = ttest_ind(mild_antidep, normal_control)
    print(f'Mild OSA Antidepressant vs Normal Control t-statistic: {t_stat:.2f}, p-value: {p_value:.4e}')
if len(moderate_antidep) > 0 and len(normal_control) > 0:
    t_stat, p_value = ttest_ind(moderate_antidep, normal_control)
    print(f'Moderate OSA Antidepressant vs Normal Control t-statistic: {t_stat:.2f}, p-value: {p_value:.4e}')

plt.tight_layout()
plt.savefig('osa_analysis_overall_v2.png', dpi=300, bbox_inches='tight')




print('done')