"""
Baseline Random Forest Analysis for Sleep Study Antidepressant Prediction

This script performs cross-validation analysis using Random Forest classifiers on:
1. Sleep stage features (baseline model)
2. EEG features (baseline model)
3. Our proposed model (pre-computed predictions)

The analysis includes bootstrap confidence intervals and per-dataset performance evaluation.
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from tqdm import tqdm
from ipdb import set_trace as bp
import dataframe_image as dfi
CSV_DIR = '../data/'

label_noise = 0

def check_single_sample_auroc():
    labels = pd.read_csv(os.path.join(CSV_DIR,'inference_v6emb_3920_all.csv'))
    labels = labels[labels['dataset'] == 'rf']
    ci = 0.95
    bootstrapped_scores = []
    for _ in tqdm(range(1000), desc="Bootstrap sampling"):
        # Sample with replacement
        sample = labels.groupby('pid').apply(lambda x: x.sample(n=1, replace=False))
        
        score = roc_auc_score(sample['label'], sample['pred'])
        bootstrapped_scores.append(score)
    
    # Calculate confidence intervals
    sorted_scores = np.sort(bootstrapped_scores)
    lower_idx = int((1.0 - ci) / 2.0 * len(sorted_scores))
    upper_idx = int((1.0 + ci) / 2.0 * len(sorted_scores))
    lower = sorted_scores[lower_idx]
    upper = sorted_scores[upper_idx]
    mean = np.mean(bootstrapped_scores)
    
    print(f"AUROC: {mean:.4f} (95% CI: {lower:.4f} - {upper:.4f})")
    ## group by pid, but as part of bootstrapping randomly sample a single per pid 
    
    auroc = roc_auc_score(labels['label'], labels['pred'])
    return auroc

def flip_label(label, prop_flip=0.1, only_if_label_is_0=False):
    """
    Flip the label with a given probability.
    """
    if only_if_label_is_0 and label == 1:
        return label
    if only_if_label_is_0 and label == 0 and np.random.rand() < prop_flip:
        return 1
    if np.random.rand() < prop_flip:
        return 1 - label
    return label

def load_and_prepare_data(labels):
    """
    Load and prepare all datasets for analysis.
    
    Returns:
    --------
    tuple : (df, df_eeg, labels_model_baseline, model1_cols, model2_cols)
    """
    # Load datasets
    # df = pd.read_csv(os.path.join(CSV_DIR,'df_baseline.csv'))
    # df_eeg = pd.read_csv(os.path.join(CSV_DIR,'df_baseline_eeg.csv'))
    df_taxonomy = pd.read_csv(os.path.join(CSV_DIR,'antidep_taxonomy_all_datasets_v6.csv'))
    
    # Prepare sleep stage features dataset
    # df = df.drop(columns=['dataset'])
    # model1_cols = [col for col in df.columns if col not in ['filename', 'fold', 'dataset', 'label']]
    
    # # Prepare EEG features dataset
    # df_eeg = df_eeg.merge(df, on='filename', how='inner')
    # df_eeg = df_eeg.drop(columns=['dataset'])
    # model2_cols = [col for col in df_eeg.columns if col not in ['filename', 'fold', 'dataset', 'label']]
    
    # Process our model predictions
    labels['filename'] = labels['filename'].apply(lambda x: x.split('/')[-1])
    labels = labels.groupby('filename', as_index=False).agg(
        lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0]
    )
    
    # Clean participant IDs across datasets
    for dataset in ['wsc', 'mros', 'shhs']:
        mask = labels['dataset'] == dataset
        labels.loc[mask, 'pid'] = labels.loc[mask, 'pid'].apply(lambda x: x[1:] if isinstance(x, str) and x else x)
    
    # Merge with taxonomy data
    labels_model_baseline = pd.merge(labels, df_taxonomy, on='filename', how='inner')
    labels_model_baseline = labels_model_baseline.groupby(['pid', 'taxonomy']).agg({
        'pred': 'mean', 
        'dataset': 'first', 
        'label': 'first', 
        'fold': 'first'
    }).reset_index()
    
    # Convert logits to probabilities
    labels_model_baseline['prob'] = 1 / (1 + np.exp(-labels_model_baseline['pred']))
    
    
    # Merge labels with feature datasets
    # labels_subset = labels[['filename', 'label', 'fold', 'dataset', 'mit_gender', 'mit_age']]
    # df = df.merge(labels_subset, on='filename', how='inner')
    # df_eeg = df_eeg.merge(labels_subset, on='filename', how='inner')
    
    return labels_model_baseline #df, df_eeg, labels_model_baseline, model1_cols, model2_cols


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


def evaluate_per_dataset_performance(labels_model_baseline):
    """
    Evaluate and print per-dataset performance for all models.
    
    Parameters:
    -----------
    result_sleep : DataFrame
        Sleep stage model results
    result_eeg : DataFrame
        EEG model results
    labels_model_baseline : DataFrame
        Our model results
    """
    print('\n' + '='*50)
    print('PER-DATASET PERFORMANCE EVALUATION')
    print('='*50)
    
    output = []
    
    for dataset in labels_model_baseline['dataset'].unique():
        print(f'\n--- {dataset.upper()} Dataset ---')
        
        # Our Model
        print(f'\n{dataset} Our Model:')
        dataset_mask_our = labels_model_baseline['dataset'] == dataset
        if dataset_mask_our.sum() > 0:
            auroc_our = roc_auc_score(
                labels_model_baseline[dataset_mask_our]['label'], 
                labels_model_baseline[dataset_mask_our]['prob']
            )
            mean, lower, upper = bootstrap_auroc_ci(
                labels_model_baseline[dataset_mask_our]['label'], 
                labels_model_baseline[dataset_mask_our]['prob']
            )
            output.append({
                'Dataset': dataset.replace('rf','MIT').upper(),
                'auroc': auroc_our,
                'lower': lower,
                'upper': upper
            })
        
        print()
    
    return output


def main():
    """
    Main analysis pipeline.
    """
    # check_single_sample_auroc()
    # bp() 
    results = []
    # df, df_eeg, labels_model_baseline, model1_cols, model2_cols = load_and_prepare_data()
    df_0 = load_and_prepare_data(pd.read_csv(os.path.join(CSV_DIR,'inference_v6emb_3920_all.csv')))
    df_01noise = load_and_prepare_data(pd.read_csv(os.path.join(CSV_DIR,'inference_v6emb_3920_all_noise01.csv')))
    df_05noise = load_and_prepare_data(pd.read_csv(os.path.join(CSV_DIR,'inference_v6emb_3920_all_noise05.csv')))
    df_10noise = load_and_prepare_data(pd.read_csv(os.path.join(CSV_DIR,'inference_v6emb_3920_all_noise10.csv')))
    
    for i, labels_model_baseline in enumerate([df_0, df_01noise, df_05noise, df_10noise]):
        label_noise = [0, 0.01, 0.05, 0.1][i]
        label_noise_pre = label_noise
        ## represents percentage of positive class, first calcualte the proportion of positives to negatives 
        prop_positive = labels_model_baseline['label'].mean()
        prop_ratio = (1 - prop_positive) / prop_positive
        label_noise = label_noise / prop_ratio
        print(f'Label noise: {label_noise}')
        # labels_model_baseline_noisy = labels_model_baseline.copy()
        # labels_model_baseline_noisy['label'] = labels_model_baseline_noisy['label'].apply(lambda x: flip_label(x, label_noise, only_if_label_is_0=True))
        # Evaluate per-dataset performance
        print("\n3. Evaluating per-dataset performance...")
        output = evaluate_per_dataset_performance(labels_model_baseline)
        print(output)
        output = pd.DataFrame(output)
        output['label_noise'] = label_noise_pre * 100
        results.append(output)
    df = pd.concat(results)
    df['formatted'] = df.apply(
        lambda row: f"{row['auroc']:.3f} [{row['lower']:.3f} - {row['upper']:.3f}]", 
        axis=1
    )
    dfp = df.pivot(index='label_noise', columns='Dataset', values='auroc')
    df2p = dfp.copy()
    
    og_auroc = dfp.iloc[0]


    for col in dfp.columns:
        df2p[col] = np.round(dfp[col] - og_auroc[col], 3)

    # df2p.iloc[0] = dfp.iloc[0].round(3)
    df2p.index.name = 'Label Noise %'
    dfi.export(df2p, 'label_noise_sensitivity_v3.png')
    print('done')
    
if __name__ == "__main__":
    main()