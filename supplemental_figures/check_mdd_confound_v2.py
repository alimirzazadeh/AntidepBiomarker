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
import scipy

def format_pvalue(pvalue):
    """Format p-value as a string."""
    if pvalue < 0.0001:
        return '<0.0001'
    elif pvalue < 0.001:
        return '<0.001'
    elif pvalue < 0.01:
        return '<0.01'
    else:
        return f'{pvalue:.4f}'
    
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

df.dropna(subset=['zung_index', 'pred'], inplace=True)
df_pos = df[df['label'] == 1].copy()
df_neg = df[df['label'] == 0].copy()


## repeat zung_bins 
bins = [0, 30, 40, 50, 60, 200]
labels = ['<30', '30-40', '40-50', '50-60', '60+']
df_pos['zung_index_bin'] = pd.cut(df_pos['zung_index'], bins=bins, labels=labels, include_lowest=True)
df_neg['zung_index_bin'] = pd.cut(df_neg['zung_index'], bins=bins, labels=labels, include_lowest=True)

# Calculate median and IQR for error bars
if True:
    fig,ax = plt.subplots(1,2,figsize=(16,8))
    pos_median = df_pos.groupby('zung_index_bin')['pred'].median()
    pos_mean = df_pos.groupby('zung_index_bin')['pred'].mean()
    pos_q1 = df_pos.groupby('zung_index_bin')['pred'].quantile(0.25)
    pos_q3 = df_pos.groupby('zung_index_bin')['pred'].quantile(0.75)
    pos_lower_err = pos_median - pos_q1
    pos_upper_err = pos_q3 - pos_median

    neg_median = df_neg.groupby('zung_index_bin')['pred'].median()
    neg_mean = df_neg.groupby('zung_index_bin')['pred'].mean()
    neg_q1 = df_neg.groupby('zung_index_bin')['pred'].quantile(0.25)
    neg_q3 = df_neg.groupby('zung_index_bin')['pred'].quantile(0.75)
    neg_lower_err = neg_median - neg_q1
    neg_upper_err = neg_q3 - neg_median
    green_color = sns.color_palette("Greens")[5] ## choose a dark green
    ax[0].bar(labels, pos_mean, capsize=5, color=green_color)
    ax[1].bar(labels, neg_mean, capsize=5, color=green_color)
    ## add the N
    for i, label in enumerate(labels):
        ax[0].text(i, pos_mean.iloc[i] + 0.02, f"N={df_pos.groupby('zung_index_bin')['pred'].count().iloc[i]}", ha='center', va='bottom')
        ax[1].text(i, neg_mean.iloc[i] + 0.02, f"N={df_neg.groupby('zung_index_bin')['pred'].count().iloc[i]}", ha='center', va='bottom')
    ## calculate pearson correlation and p-value
    corr, pval = scipy.stats.pearsonr(df_pos['zung_index'], df_pos['pred'])
    ## put in the top left corner of the plot
    ax[0].text(0.05, 0.93, f'Pearson correlation: {corr:.2f} (p{format_pvalue(pval)})', fontsize=12, transform=ax[0].transAxes)
    corr, pval = scipy.stats.pearsonr(df_neg['zung_index'], df_neg['pred'])
    ax[1].text(0.05, 0.93, f'Pearson correlation: {corr:.2f} (p{format_pvalue(pval)})', fontsize=12, transform=ax[1].transAxes)
    ax[0].set_title('Antidepressant')
    ax[1].set_title('Control')
    ax[0].set_xlabel('Zung Index')
    ax[0].set_ylabel('Model Prediction (Mean)')
    ax[1].set_xlabel('Zung Index')
    ax[1].set_ylabel('Model Prediction (Mean)')
    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('check_mdd_confound_v2.png', dpi=300, bbox_inches='tight')

## now we calculate with FN and FP rate in a similar box plot by zung_index_bin
if False:
    threshold = 0.25

    df_pos['fn'] = (df_pos['label'] == 1) & (df_pos['pred'] < threshold)
    df_pos['fp'] = (df_pos['label'] == 0) & (df_pos['pred'] >= threshold)
    df_neg['fn'] = (df_neg['label'] == 1) & (df_neg['pred'] < threshold)
    df_neg['fp'] = (df_neg['label'] == 0) & (df_neg['pred'] >= threshold)



    fig,ax = plt.subplots(1,2,figsize=(10, 6))
    ax[0].bar(labels, df_pos.groupby('zung_index_bin')['fn'].mean(), capsize=5)
    ax[1].bar(labels, df_neg.groupby('zung_index_bin')['fp'].mean(), capsize=5)
    for i, label in enumerate(labels):
        ax[0].text(i, df_pos.groupby('zung_index_bin')['fn'].mean().iloc[i] + 0.01, f"N={df_pos.groupby('zung_index_bin')['fn'].count().iloc[i]}", ha='center', va='bottom')
        ax[1].text(i, df_neg.groupby('zung_index_bin')['fp'].mean().iloc[i] + 0.01, f"N={df_neg.groupby('zung_index_bin')['fp'].count().iloc[i]}", ha='center', va='bottom')
        
    ## now print the ttest p-value for each zung index bin vs all others
    for i, label in enumerate(labels):
        ttest = ttest_ind(df_pos[df_pos['zung_index_bin'] == label]['pred'], df_pos[df_pos['zung_index_bin'] != label]['pred'])
        print(f'False Negative Rate for {label}: t-statistic: {ttest.statistic:.2f}, p-value: {ttest.pvalue:.4e}')
        ttest = ttest_ind(df_neg[df_neg['zung_index_bin'] == label]['pred'], df_neg[df_neg['zung_index_bin'] != label]['pred'])
        print(f'False Positive Rate for {label}: t-statistic: {ttest.statistic:.2f}, p-value: {ttest.pvalue:.4e}')

    ax[0].set_title('False Negative Rate')
    ax[1].set_title('False Positive Rate')
    ax[0].set_xlabel('Zung Index')
    ax[0].set_ylabel('False Negative Rate')
    ax[1].set_xlabel('Zung Index')
    ax[1].set_ylabel('False Positive Rate')
    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


print('done')