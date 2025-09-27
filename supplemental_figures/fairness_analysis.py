import pandas as pd 
from ipdb import set_trace as bp
import numpy as np 
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

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



INFERENCE_FILE = '../data/inference_v6emb_3920_all.csv'
df = pd.read_csv(INFERENCE_FILE)

# Group by filename and aggregate (mean for numeric, first for non-numeric)
df['filename'] = df['filepath'].apply(lambda x: x.split('/')[-1])
df = df.groupby('filename').agg(
    lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0]
)

# Convert logits to probabilities using sigmoid
df['pred'] = 1 / (1 + np.exp(-df['pred']))

# Clean patient IDs by removing dataset prefixes
df['pid'] = df.apply(
    lambda x: x['pid'][1:] if x['dataset'] in ['shhs', 'mros', 'wsc'] else x['pid'], 
    axis=1
)
df = df.groupby(['pid', 'label'], as_index=False).agg(
    lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0]
)

## Now we are going to bin the ages into size 10 years 
df['age_bin'] = pd.cut(df['mit_age'], bins=range(0, 100, 10), right=False)
## Calculat the AUROC for each age bin 
age_results = {}
gender_results = {}
df_age = df[df['mit_age'].notna()].copy()

## 27/1868 positive patients in the 20-30 age bin
for age_bin in df_age['age_bin'].unique():
    df_age_bin = df_age[df_age['age_bin'] == age_bin]
    if len(df_age_bin) == 0:
        continue
    # auroc = roc_auc_score(df_age_bin['label'].values, df_age_bin['pred'].values)
    auroc, lower, upper = bootstrap_auroc_ci(df_age_bin['label'].values, df_age_bin['pred'].values)
    age_results[age_bin] = (auroc, lower, upper)

df_gender = df[df['mit_gender'].notna()].copy()
for gender in df_gender['mit_gender'].unique():
    df_gender_bin = df_gender[df_gender['mit_gender'] == gender]
    if len(df_gender_bin) == 0:
        continue
    # auroc = roc_auc_score(df_gender['label'].values, df_gender['pred'].values)
    auroc, lower, upper = bootstrap_auroc_ci(df_gender_bin['label'].values, df_gender_bin['pred'].values)
    gender_results[gender] = (auroc, lower, upper)


keys = sorted(list(age_results.keys()))
for age_bin in keys:
    print(age_bin.left, '- ' + str(age_bin.right) +': '+ str(np.round(age_results[age_bin][0], 2)), '[', np.round(age_results[age_bin][1], 2), '-', np.round(age_results[age_bin][2], 2), ']')

for gender in gender_results.keys():
    print(gender, np.round(gender_results[gender][0], 2), '[', np.round(gender_results[gender][1], 2), '-', np.round(gender_results[gender][2], 2), ']')


## now plot each as an sns barplot , make the x axis labels sorted by age and print as X-X


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
age_bins = sorted(list(age_results.keys()), key=lambda x: x.left)
# Convert intervals to string labels for plotting
age_labels = [f"{int(age_bin.left)}-{int(age_bin.right)}" for age_bin in age_bins]

# Create the bar plot with string labels
## add an error bar for each bar 
sns.barplot(x=age_labels, y=[age_results[age_bin][0] for age_bin in age_bins], ax=ax1)
for i, age_bin in enumerate(age_bins):
    # Calculate error bar length (distance from mean to CI bounds)
    mean_auroc = age_results[age_bin][0]
    lower_ci = age_results[age_bin][1]
    upper_ci = age_results[age_bin][2]

    # Error bar goes from mean to upper CI and mean to lower CI
    yerr_lower = mean_auroc - lower_ci
    yerr_upper = upper_ci - mean_auroc

    ax1.errorbar(i, mean_auroc, yerr=[[yerr_lower], [yerr_upper]], fmt='none', ecolor='black', capsize=5)

    # Add n= on top of each bar
    n_samples = len(df_age[df_age["age_bin"] == age_bin])
    # ax1.text(i, mean_auroc + 0.01, f'n={n_samples}', ha='center', va='bottom')

ax1.set_xlabel('Age Range (years)')
ax1.set_ylabel('AUROC')
ax1.set_title('Model Performance by Age Group')
# ax1.tick_params(axis='x', rotation=45)

sns.barplot(x=list(gender_results.keys()), y=[gender_results[gender][0] for gender in gender_results.keys()], ax=ax2, order=[1.0,2.0], width=0.3)
for i, gender in enumerate([1.0,2.0]):
    mean_auroc = gender_results[gender][0]
    lower_ci = gender_results[gender][1]
    upper_ci = gender_results[gender][2]

    # Error bar goes from mean to upper CI and mean to lower CI
    yerr_lower = mean_auroc - lower_ci
    yerr_upper = upper_ci - mean_auroc

    ax2.errorbar(i, mean_auroc, yerr=[[yerr_lower], [yerr_upper]], fmt='none', ecolor='black', capsize=5)

    # Add n= on top of each bar
    n_samples = len(df_gender_bin)
    # ax2.text(i, mean_auroc + 0.01, f'n={n_samples}', ha='center', va='bottom')

ax2.set_xlabel('Sex')
ax2.set_ylabel('AUROC')
ax2.set_title('Model Performance by Sex')
ax2.set_ylim(0,1)
ax2.set_xticks([0,1])
ax2.set_xticklabels(['Male', 'Female'])
ax2.set_xlim(-0.5, 1.5)
# plt.show()
plt.savefig('fairness_analysis.png', dpi=300, bbox_inches='tight')

print('done')