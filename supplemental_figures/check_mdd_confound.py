"""
For MROS and WSC, we evaluate the model performance at different levels of MDD severity.

"""

import pandas as pd 
import numpy as np 
from ipdb import set_trace as bp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def bootstrap_auroc_ci(y_true, y_prob, n_bootstrap=1000, ci=0.95, random_state=42):
    """
    Calculate bootstrap confidence intervals for AUROC.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities for positive class
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence interval (default 0.95 for 95% CI)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (mean_auroc, lower_ci, upper_ci)
    """
    rng = np.random.RandomState(random_state)
    bootstrapped_scores = []
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap sampling"):
        # Sample with replacement
        indices = rng.choice(np.arange(len(y_true)), size=len(y_true), replace=True)
        
        # Skip if not both classes present in sample
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)
    
    # Calculate confidence intervals
    sorted_scores = np.sort(bootstrapped_scores)
    lower_idx = int((1.0 - ci) / 2.0 * len(sorted_scores))
    upper_idx = int((1.0 + ci) / 2.0 * len(sorted_scores))
    lower = sorted_scores[lower_idx]
    upper = sorted_scores[upper_idx]
    mean = np.mean(bootstrapped_scores)
    
    print(f"AUROC: {mean:.4f} (95% CI: {lower:.4f} - {upper:.4f})")
    return mean, lower, upper


def round_pandas_interval(interval, ndigits=0):
    """Rounds the left and right bounds of a pandas.Interval."""
    # Round the left and right bounds
    rounded_left = round(interval.left, ndigits)
    rounded_right = round(interval.right, ndigits)
    
    # Create a new Interval with the rounded bounds and original closed property
    return pd.Interval(
        left=rounded_left,
        right=rounded_right,
        closed=interval.closed
    )

df = pd.read_csv('../data/wsc-dataset-0.7.0.csv')
mros = pd.read_csv('../data/mros_gds_progression.csv')
mros_gds = {} 
for i, row in mros.iterrows():
    filename = f'mros-visit1-aa{row["nsrrid"][2:]}.npz'
    mros_gds[filename] = row['DPGS15_1']
    filename = f'mros-visit2-aa{row["nsrrid"][2:]}.npz'
    mros_gds[filename] = row['DPGS15_2']
mros = pd.DataFrame(list(mros_gds.items()), columns=['filename', 'gds'])
df['filename'] = df.apply(
        lambda x: f'wsc-visit{x["wsc_vst"]}-{x["wsc_id"]}-nsrr.npz', 
        axis=1
    )
labels = pd.read_csv('../data/inference_v6emb_3920_all.csv')

labels['filename'] = labels['filepath'].apply(lambda x: x.split('/')[-1])
labels = labels.groupby('filename').agg(
        lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0]
    )
labels['pred'] = 1 / (1 + np.exp(-labels['pred']))
df = df[['zung_index', 'filename']]
df = df.merge(labels, on='filename', how='inner')
mros = mros.merge(labels, on='filename', how='inner')
if True:
    df_pos = df.copy() #df[df['label'] == 1].copy()
    ## split up zung_index into custom bins: <30, 30-40, 40-50, 50-60, 60+
    # Create custom bins: below 30, then every 10 up to 60, then 60+
    bins = [0, 30, 40, 50, 60, 200]  # 200 is a reasonable upper bound
    # labels = ['<30', '30-40', '40-50', '50-60', '60+']
    df_pos['zung_index_bin'] = pd.cut(df_pos['zung_index'], bins=bins, include_lowest=True)
        # 3) Group and compute mean prediction
    # Pass observed=True to silence the future warning (useful when 'zung_index_bin' is categorical)
    ## plot boxplot of the pred by zung_index_bin
    ## do p-value test for each bin vs all others, put in a table 
    zung_index_results = {}
    for zung_index_bin in df_pos['zung_index_bin'].unique():
        df_zung_index_bin = df_pos[df_pos['zung_index_bin'] == zung_index_bin]
        if len(df_zung_index_bin) == 0:
            continue
        # auroc = roc_auc_score(df_zung_index_bin['label'].values, df_zung_index_bin['pred'].values)
        auroc, lower, upper = bootstrap_auroc_ci(df_zung_index_bin['label'].values, df_zung_index_bin['pred'].values)
        zung_index_results[zung_index_bin] = (auroc, lower, upper)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    zung_index_bins = sorted(list(zung_index_results.keys()), key=lambda x: x.left)
    sns.barplot(x=zung_index_bins, y=[zung_index_results[zung_index_bin][0] for zung_index_bin in zung_index_bins], ax=ax1, width=0.6, palette='Greens', order=zung_index_bins)
    for i, zung_index_bin in enumerate(zung_index_bins):
        mean_auroc = zung_index_results[zung_index_bin][0]
        lower_ci = zung_index_results[zung_index_bin][1]
        upper_ci = zung_index_results[zung_index_bin][2]
        yerr_lower = mean_auroc - lower_ci
        yerr_upper = upper_ci - mean_auroc
        ax1.errorbar(i, mean_auroc, yerr=[[yerr_lower], [yerr_upper]], fmt='none', ecolor='black', capsize=5)
        ## add N= on top of each bar
        n_samples = len(df_pos[df_pos['zung_index_bin'] == zung_index_bin])
        ax1.text(i, upper_ci + 0.01, f'N={n_samples}', ha='center', va='bottom')
    ax1.set_xlabel('Zung Index Bin')
    ax1.set_ylabel('AUROC')
    ax1.set_title('Model Performance by Zung Index Bin')
    ax1.set_ylim(0,1)
    ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax1.set_xticks(range(5))
    labels = ['<30', '30-40', '40-50', '50-60', '60+']
    ax1.set_xticklabels(labels)
    plt.tight_layout()
    plt.savefig('check_mdd_confound.png', dpi=300, bbox_inches='tight')
    bp() 
            
    if False:
        grouped = df_pos.groupby('zung_index_bin', observed=True)['pred'].agg(['mean', 'std']).sort_index()
        # Option A: plot with readable interval labels (strings)
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = grouped.index.astype(str)  # Already string labels, just convert to string for consistency
        values = grouped.values
        ax.bar(labels, grouped['mean'], yerr=grouped['std'], capsize=5)
        ax.set_xlabel('Zung Index Bin')
        ax.set_ylabel('Average Pred')
        ax.set_title('Average Pred vs Zung Index Bin (labels)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
if False:
    df_pos = mros[mros['label'] == 1].copy()
    ## split up gds into 8 bins, and plot the average pred 
    # df_pos['gds_bin'] = pd.cut(df_pos['gds'], bins=8)
    bins = [0, 2, 4, 6, 8, 100]
    labels = ['0-2', '2-4', '4-6', '6-8', '8+']
    df_pos['gds_bin'] = pd.cut(df_pos['gds'], bins=bins, labels=labels, include_lowest=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='gds_bin', y='pred', data=df_pos, ax=ax, palette='Greens', width=0.5)
    bin_counts = df_pos['gds_bin'].value_counts()
    bin_counts = bin_counts.sort_index()
    for i in range(len(bin_counts)):
        ax.text(i, df_pos['pred'].max(), f'N={bin_counts.iloc[i]}', ha='center', va='bottom')
    ax.set_xlabel('GDS Bin')
    ax.set_ylabel('Model Score')
    ax.set_title('Model Score vs GDS Bin')
    ax.set_ylim(0, 1.02)
    plt.show()
bp() 
print('done')