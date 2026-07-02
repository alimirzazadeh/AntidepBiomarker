import pandas as pd
from ipdb import set_trace as bp
import numpy as np
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ttest_ind as _ttest_ind_base
def ttest_ind(a, b, equal_var=False, **kwargs):  # Welch's by default
    return _ttest_ind_base(a, b, equal_var=equal_var, **kwargs)
import os
font_size = 13
def get_significance_stars(pval):
    if pval < .001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'ns'
def bootstrap_auroc_ci(y_true, y_prob, n_bootstrap=1000, ci=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    bootstrapped_scores = []

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap sampling"):
        indices = rng.choice(np.arange(len(y_true)), size=len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    lower_idx = int((1.0 - ci) / 2.0 * len(sorted_scores))
    upper_idx = int((1.0 + ci) / 2.0 * len(sorted_scores))
    lower = sorted_scores[lower_idx]
    upper = sorted_scores[upper_idx]
    mean = np.mean(bootstrapped_scores)

    print(f"AUROC: {mean:.4f} (95% CI: {lower:.4f} - {upper:.4f})")
    return mean, lower, upper


def load_data():
    CSV_DIR = 'data/'
    INFERENCE_FILE = os.path.join(CSV_DIR, 'inference_v6emb_3920_all.csv')
    df = pd.read_csv(INFERENCE_FILE)
    df['filename'] = df['filepath'].apply(lambda x: x.split('/')[-1])
    df = df.groupby('filename').agg(
        lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0]
    )
    df['pred'] = 1 / (1 + np.exp(-df['pred']))
    df['pid'] = df.apply(
        lambda x: x['pid'][1:] if x['dataset'] in ['shhs', 'mros', 'wsc'] else x['pid'],
        axis=1
    )
    df = df.groupby(['pid', 'label'], as_index=False).agg(
        lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0]
    )
    return df


def generate_fairness_analysis_figure(save=True, ax=None):
    df = load_data()

    bmi_bins_edges = [0, 18.5, 25, 30, 35, 40, 100]
    df['bmi_bin'] = pd.cut(df['mit_bmi'], bins=bmi_bins_edges, right=False)

    bmi_results = {}
    df_bmi = df[df['mit_bmi'].notna()].copy()
    for bmi_bin in df_bmi['bmi_bin'].unique():
        df_bmi_bin = df_bmi[df_bmi['bmi_bin'] == bmi_bin]
        if len(df_bmi_bin) == 0:
            continue
        auroc, lower, upper = bootstrap_auroc_ci(df_bmi_bin['label'].values, df_bmi_bin['pred'].values)
        bmi_results[bmi_bin] = (auroc, lower, upper)

    for bmi_bin in bmi_results.keys():
        print(bmi_bin.left, '- ' + str(bmi_bin.right) +': '+ str(np.round(bmi_results[bmi_bin][0], 2)), '[', np.round(bmi_results[bmi_bin][1], 2), '-', np.round(bmi_results[bmi_bin][2], 2), ']')

    if True:
        ## repeat zung_bins 
        print('Unique Datasets used in BMI Analysis: ', df_bmi['dataset'].unique())
        bmi_bins = df_bmi['bmi_bin'].unique()
        bins = [0, 18.5, 25, 30, 35, 40, 100]
        labels = ['<18.5', '18.5-25', '25-30', '30-35', '35-40', '>40']
        df_bmi['bmi_bin'] = pd.cut(df_bmi['mit_bmi'], bins=bins, labels=labels, include_lowest=True)
        df_bmi['Group'] = df_bmi['label'].apply(lambda x: 'Control' if x == 0 else 'Antidepressant')
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        
        # Create boxplots using seaborn with hue for paired boxes
        sns.boxplot(data=df_bmi, x='bmi_bin', y='pred', hue='Group', 
                    order=labels, palette='Greens', ax=ax, showfliers=False)
        # Remove legend
        ax.legend_.remove()
        
        # Add N labels above each boxplot (above the median line)
        # Get the positions of the boxes
        box_positions = ax.get_xticks()
        n_bins = len(labels)
        
        for i, label in enumerate(labels):
            # Control box (left side of pair)
            subset_control = df_bmi[(df_bmi['bmi_bin'] == label) & 
                                    (df_bmi['Group'] == 'Control')]['pred'].dropna()
            if len(subset_control) > 0:
                n = len(subset_control)
                median_val = subset_control.median()
                # Position is box_positions[i] - 0.2 (offset for paired boxes)
                ax.text(box_positions[i] - 0.2, median_val + 0.01, f'N={n}', 
                    ha='center', va='bottom', fontsize=font_size)
            
            # Antidepressant box (right side of pair)
            subset_antidep = df_bmi[(df_bmi['bmi_bin'] == label) & 
                                (df_bmi['Group'] == 'Antidepressant')]['pred'].dropna()
            if len(subset_antidep) > 0:
                n = len(subset_antidep)
                median_val = subset_antidep.median()
                # Position is box_positions[i] + 0.2 (offset for paired boxes)
                ax.text(box_positions[i] + 0.2, median_val + 0.01, f'N={n}', 
                    ha='center', va='bottom', fontsize=font_size)
            
            # Compute t-test between Control and Antidepressant for this bin
            if len(subset_control) > 0 and len(subset_antidep) > 0:
                ttest = ttest_ind(subset_control.values, subset_antidep.values)
                print(f'T-test p value for {label} (Control vs Antidepressant): {ttest.pvalue:.4e}')
                
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
                ax.text(x_center, bracket_y + 0.01, sig_stars, ha='center', va='bottom', fontsize=font_size)
            
            # Compute t-test against all positives/negatives
            if len(subset_control) > 0:
                ttest = ttest_ind(subset_control.values, df_bmi[df_bmi['Group'] == 'Antidepressant']['pred'].dropna().values)
                print(f'T-test p value for Control {label} vs all positives: {ttest.pvalue:.4e}')
            if len(subset_antidep) > 0:
                ttest = ttest_ind(subset_antidep.values, df_bmi[df_bmi['Group'] == 'Control']['pred'].dropna().values)
                print(f'T-test p value for Antidepressant {label} vs all negatives: {ttest.pvalue:.4e}')
        
        ax.set_ylabel('Model Score', fontsize=font_size)
        ax.set_xlabel('BMI', fontsize=font_size)
        ax.set_ylim(0, 1.1)  # Increased to accommodate significance brackets
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
        if save:
            plt.tight_layout()
            plt.savefig('fairness_analysis_bmi.png', dpi=300, bbox_inches='tight')
        return ax

if __name__ == '__main__':
    generate_fairness_analysis_figure(save=True)