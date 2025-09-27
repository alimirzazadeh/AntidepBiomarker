import pandas as pd
import numpy as np 
import os 
from ipdb import set_trace as bp
import matplotlib.pyplot as plt
import seaborn as sns
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
df = df[(df['is_tca'] == 1) | (df['label'] == 0)]
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
osa_groups = {} 
osa_groups_control = {} 
fig, ax = plt.subplots(figsize=(12, 6))
for osa_group in ['Normal', 'Mild OSA', 'Moderate OSA', 'Severe OSA']:
    osa_group_df = df[(df['osa_group'] == osa_group) & (df['label'] == 1)]
    osa_group_df = osa_group_df[osa_group_df['pred'].notna()]
    osa_groups[osa_group] = osa_group_df['pred'].values
    osa_group_df = df[(df['osa_group'] == osa_group) & (df['label'] == 0)]
    osa_group_df = osa_group_df[osa_group_df['pred'].notna()]
    osa_groups_control[osa_group] = osa_group_df['pred'].values
# osa_groups['Control'] = df[df['label'] == 0]['pred'].values


cohort = len(osa_groups['Normal']) * ['Normal\nAntidepressant'] + len(osa_groups['Mild OSA']) * ['Mild OSA\nAntidepressant'] + len(osa_groups['Moderate OSA']) * ['Moderate OSA\nAntidepressant'] + len(osa_groups['Severe OSA']) * ['Severe OSA\nAntidepressant']
cohort_control = len(osa_groups_control['Normal']) * ['Normal\nControl'] + len(osa_groups_control['Mild OSA']) * ['Mild OSA\nControl'] + len(osa_groups_control['Moderate OSA']) * ['Moderate OSA\nControl'] + len(osa_groups_control['Severe OSA']) * ['Severe OSA\nControl']
df_plot = pd.DataFrame({
    'cohort':  cohort_control + cohort,
    'pred': np.concatenate([osa_groups_control['Normal'], osa_groups_control['Mild OSA'], osa_groups_control['Moderate OSA'], osa_groups_control['Severe OSA'], osa_groups['Normal'], osa_groups['Mild OSA'], osa_groups['Moderate OSA'], osa_groups['Severe OSA']])
})
sns.boxplot(x='cohort', y='pred', data=df_plot, showfliers=False, palette='Greens', order=['Normal\nControl', 'Mild OSA\nControl', 'Moderate OSA\nControl', 'Severe OSA\nControl', 'Normal\nAntidepressant', 'Mild OSA\nAntidepressant', 'Moderate OSA\nAntidepressant', 'Severe OSA\nAntidepressant'], ax=ax)
ax.set_ylabel('Model Score')
ax.set_xlabel('Cohort')


## N for each cohort
for osa_group in ['Normal\nAntidepressant', 'Mild OSA\nAntidepressant', 'Moderate OSA\nAntidepressant', 'Severe OSA\nAntidepressant', 'Normal\nControl', 'Mild OSA\nControl', 'Moderate OSA\nControl', 'Severe OSA\nControl']:
    subset = df_plot[df_plot['cohort'] == osa_group]
    n = subset.shape[0]
    median = subset['pred'].median()
    q1 = subset['pred'].quantile(0.25)
    q3 = subset['pred'].quantile(0.75)
    print(f'{osa_group}: Median={median:.2f}, IQR: [{q1:.2f}-{q3:.2f}]')
    ax.text(osa_group, median, f'N={n}', ha='center', va='bottom', fontsize=10)

from scipy.stats import ttest_ind
def cohen_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.var(x) + np.var(y)) / 2)
t_stat, p_value = ttest_ind(df_plot[df_plot['cohort'] == 'Severe OSA\nAntidepressant']['pred'], df_plot[df_plot['cohort'] == 'Normal\nAntidepressant']['pred'])
print(f'Severe t-statistic: {t_stat:.2f}, p-value: {p_value:.2f}')
t_stat, p_value = ttest_ind(df_plot[df_plot['cohort'] == 'Mild OSA\nAntidepressant']['pred'], df_plot[df_plot['cohort'] == 'Normal\nAntidepressant']['pred'])
print(f'Mild t-statistic: {t_stat:.2f}, p-value: {p_value:.2f}')
t_stat, p_value = ttest_ind(df_plot[df_plot['cohort'] == 'Moderate OSA\nAntidepressant']['pred'], df_plot[df_plot['cohort'] == 'Normal\nAntidepressant']['pred'])
print(f'Moderate t-statistic: {t_stat:.2f}, p-value: {p_value:.2f}')

t_stat, p_value = ttest_ind(df_plot[df_plot['cohort'] == 'Severe OSA\nAntidepressant']['pred'], df_plot[df_plot['cohort'] == 'Normal\nControl']['pred'])
print(f't-statistic: {t_stat:.2f}, p-value: {p_value:.2e}')
t_stat, p_value = ttest_ind(df_plot[df_plot['cohort'] == 'Mild OSA\nAntidepressant']['pred'], df_plot[df_plot['cohort'] == 'Normal\nControl']['pred'])
print(f'Mild t-statistic: {t_stat:.2f}, p-value: {p_value:.2e}')
t_stat, p_value = ttest_ind(df_plot[df_plot['cohort'] == 'Moderate OSA\nAntidepressant']['pred'], df_plot[df_plot['cohort'] == 'Normal\nControl']['pred'])
print(f'Moderate t-statistic: {t_stat:.2f}, p-value: {p_value:.2e}')

plt.savefig('osa_analysis_tca.png', dpi=300, bbox_inches='tight')




print('done')