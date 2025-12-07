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
bmi_bins = [0, 18.5, 25, 30, 35, 40, 100]
df['bmi_bin'] = pd.cut(df['mit_bmi'], bins=bmi_bins, right=False)
## Calculat the AUROC for each age bin 
age_results = {}
gender_results = {}
bmi_results = {}
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

df_bmi = df[df['mit_bmi'].notna()].copy()
for bmi_bin in df_bmi['bmi_bin'].unique():
    df_bmi_bin = df_bmi[df_bmi['bmi_bin'] == bmi_bin]
    if len(df_bmi_bin) == 0:
        continue
    # auroc = roc_auc_score(df_bmi_bin['label'].values, df_bmi_bin['pred'].values)
    auroc, lower, upper = bootstrap_auroc_ci(df_bmi_bin['label'].values, df_bmi_bin['pred'].values)
    bmi_results[bmi_bin] = (auroc, lower, upper)

keys = sorted(list(age_results.keys()))
for age_bin in keys:
    print(age_bin.left, '- ' + str(age_bin.right) +': '+ str(np.round(age_results[age_bin][0], 2)), '[', np.round(age_results[age_bin][1], 2), '-', np.round(age_results[age_bin][2], 2), ']')

for gender in gender_results.keys():
    print(gender, np.round(gender_results[gender][0], 2), '[', np.round(gender_results[gender][1], 2), '-', np.round(gender_results[gender][2], 2), ']')

for bmi_bin in bmi_results.keys():
    print(bmi_bin.left, '- ' + str(bmi_bin.right) +': '+ str(np.round(bmi_results[bmi_bin][0], 2)), '[', np.round(bmi_results[bmi_bin][1], 2), '-', np.round(bmi_results[bmi_bin][2], 2), ']')
## now plot each as an sns barplot , make the x axis labels sorted by age and print as X-X




# Create the bar plot with string labels
## add an error bar for each bar 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
age_bins = sorted(list(age_results.keys()), key=lambda x: x.left)
# Convert intervals to string labels for plotting
age_labels = [f"{int(age_bin.left)}-{int(age_bin.right)}" for age_bin in age_bins]
sns.barplot(x=age_labels, y=[age_results[age_bin][0] for age_bin in age_bins], ax=ax1, palette='Greens')
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
ax1.set_ylim(0,1)
ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax1.set_xlabel('Age Range (years)')
ax1.set_ylabel('AUROC')
ax1.set_title('Model Performance by Age Group')
# ax1.tick_params(axis='x', rotation=45)

sns.barplot(x=list(gender_results.keys()), y=[gender_results[gender][0] for gender in gender_results.keys()], ax=ax2, width=0.3, palette='Greens')
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

## order the bmi bins by the bmi values 
bmi_bins = sorted(list(bmi_results.keys()), key=lambda x: x.left)
sns.barplot(x=bmi_bins, y=[bmi_results[bmi_bin][0] for bmi_bin in bmi_bins], ax=ax3, width=0.6, palette='Greens')
for i, bmi_bin in enumerate(bmi_bins):
    mean_auroc = bmi_results[bmi_bin][0]
    lower_ci = bmi_results[bmi_bin][1]
    upper_ci = bmi_results[bmi_bin][2]

    # Error bar goes from mean to upper CI and mean to lower CI
    yerr_lower = mean_auroc - lower_ci
    yerr_upper = upper_ci - mean_auroc

    ax3.errorbar(i, mean_auroc, yerr=[[yerr_lower], [yerr_upper]], fmt='none', ecolor='black', capsize=5)
    ## add N= on top of each bar
    n_samples = len(df_bmi[df_bmi["bmi_bin"] == bmi_bin])
    ax3.text(i, upper_ci + 0.01, f'N={n_samples}', ha='center', va='bottom')
    print(bmi_bin.left, '- ' + str(bmi_bin.right) +': '+ str(np.round(mean_auroc, 2)), '[', np.round(lower_ci, 2), '-', np.round(upper_ci, 2), ']')
    # Add n= on top of each bar
    n_samples = len(df_bmi[df_bmi["bmi_bin"] == bmi_bin])
    # ax3.text(i, mean_auroc + 0.01, f'n={n_samples}', ha='center', va='bottom')

ax3.set_xlabel('BMI Range (kg/m^2)')
ax3.set_ylabel('AUROC')
ax3.set_title('Model Performance by BMI Group')
ax3.set_ylim(0,1)
ax3.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax2.set_xlabel('Sex')
ax2.set_ylabel('AUROC')
ax2.set_title('Model Performance by Sex')
ax2.set_ylim(0,1)
ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax2.set_xticks([0,1])
ax2.set_xticklabels(['Male', 'Female'])
ax2.set_xlim(-0.5, 1.5)
plt.tight_layout()
plt.savefig('fairness_analysis.png', dpi=300, bbox_inches='tight')


if True:
    from scipy.stats import ttest_ind
    ## repeat zung_bins 
    bmi_bins = df_bmi['bmi_bin'].unique()
    bins = [0, 18.5, 25, 30, 35, 40, 100]
    labels = ['<18.5', '18.5-25', '25-30', '30-35', '35-40', '>40']
    df_bmi['bmi_bin'] = pd.cut(df_bmi['mit_bmi'], bins=bins, labels=labels, include_lowest=True)
    df_bmi['Group'] = df_bmi['label'].apply(lambda x: 'Control' if x == 0 else 'Antidep')
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Prepare data for boxplots: controls first, then antidepressants
    # Add group labels

    
    # Create a column for x-axis ordering: controls first, then antidepressants
    # We'll create a combined label that includes both group and bin
    df_bmi['BMI Bin'] = df_bmi.apply(
        lambda x: f"{x['Group']}\n{x['bmi_bin']}", axis=1
    )
    
    # Create order list: controls first (ordered by zung_index_bin), then antidepressants
    order_list = []
    for label in labels:
        order_list.append(f'Control\n{label}')
    for label in labels:
        order_list.append(f'Antidep\n{label}')
    
    # Create boxplots using seaborn
    sns.boxplot(data=df_bmi, x='BMI Bin', y='pred', order=order_list,
                palette='Greens', ax=ax, showfliers=False)
    
    # Add N labels above each boxplot (above the median line)
    for i, x_label in enumerate(order_list):
        subset = df_bmi[df_bmi['BMI Bin'] == x_label]['pred'].dropna()
        if i < len(labels):
            ## compute t test p value against positives 
            ttest = ttest_ind(subset.values, df_bmi[df_bmi['Group'] == 'Antidepressant']['pred'].dropna().values)
            print(f'T-test p value for {x_label} vs positives: {ttest.pvalue:.4e}')
        else:
            ## compute t test p value against negatives
            ttest = ttest_ind(subset.values, df_bmi[df_bmi['Group'] == 'Control']['pred'].dropna().values)
            print(f'T-test p value for {x_label} vs negatives: {ttest.pvalue:.4e}')
        if len(subset) > 0:
            n = len(subset)
            median_val = subset.median()
            ax.text(i, median_val + 0.01, f'N={n}', ha='center', va='bottom', fontsize=9)
    p_value_grid = np.zeros((len(labels), len(labels)))
    for i, x_label in enumerate(order_list):
        if not x_label.startswith('Control'):
            continue
        subset = df_bmi[df_bmi['BMI Bin'] == x_label]['pred'].dropna()
        for j, x_label2 in enumerate(order_list):
            if not x_label2.startswith('Antidep'):
                continue
            subset2 = df_bmi[df_bmi['BMI Bin'] == x_label2]['pred'].dropna()
            ttest = ttest_ind(subset.values, subset2.values)
            p_value_grid[i % len(labels), j % len(labels)] = ttest.pvalue
    p_value_grid = pd.DataFrame(p_value_grid, index=labels, columns=labels).T
    print(p_value_grid.max())
    ax.set_ylabel('Model Prediction')
    ax.set_xlabel('BMI Group')
    plt.tight_layout()
    plt.savefig('fairness_analysis_bmi.png', dpi=300, bbox_inches='tight')

print('done')