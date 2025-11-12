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
from sklearn.metrics import average_precision_score ## auprc
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from tqdm import tqdm
from ipdb import set_trace as bp

CSV_DIR = '../data/'
def load_and_prepare_data():
    """
    Load and prepare all datasets for analysis.
    
    Returns:
    --------
    tuple : (df, df_eeg, labels_model_baseline, model1_cols, model2_cols)
    """
    # Load datasets
    df = pd.read_csv(os.path.join(CSV_DIR,'df_baseline.csv'))
    df_eeg = pd.read_csv(os.path.join(CSV_DIR,'df_baseline_eeg.csv'))
    labels = pd.read_csv(os.path.join(CSV_DIR,'inference_v6emb_3920_all.csv'))
    df_taxonomy = pd.read_csv(os.path.join(CSV_DIR,'antidep_taxonomy_all_datasets_v6.csv'))
    
    # Prepare sleep stage features dataset
    df = df.drop(columns=['dataset'])
    model1_cols = [col for col in df.columns if col not in ['filename', 'fold', 'dataset', 'label']]
    
    # Prepare EEG features dataset
    df_eeg = df_eeg.merge(df, on='filename', how='inner')
    df_eeg = df_eeg.drop(columns=['dataset'])
    model2_cols = [col for col in df_eeg.columns if col not in ['filename', 'fold', 'dataset', 'label']]
    
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
    labels_subset = labels[['filename', 'label', 'fold', 'dataset', 'mit_gender', 'mit_age']]
    df = df.merge(labels_subset, on='filename', how='inner')
    df_eeg = df_eeg.merge(labels_subset, on='filename', how='inner')
    
    return df, df_eeg, labels_model_baseline, model1_cols, model2_cols


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
            
        score = average_precision_score(y_true[indices], y_prob[indices])
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


def run_rf_auroc_pruned(train_set, train_y, test_set, test_y, cols=None):
    """
    Train Random Forest with feature pruning based on permutation importance.
    
    Parameters:
    -----------
    train_set : array-like
        Training features
    train_y : array-like
        Training labels
    test_set : array-like
        Test features
    test_y : array-like
        Test labels
    cols : list, optional
        Column names for feature importance reporting
        
    Returns:
    --------
    array : Predicted probabilities for test set
    """
    # Clean NaNs and infs
    train_set = np.nan_to_num(train_set, nan=0.0, posinf=0.0, neginf=0.0)
    test_set = np.nan_to_num(test_set, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train original model on all features
    original_model = RandomForestClassifier(
        n_estimators=1000,
        bootstrap=True,
        class_weight={0: 1, 1: 2},
        random_state=42,
        max_depth=10
    )
    original_model.fit(train_set, train_y)
    y_prob_orig = original_model.predict_proba(test_set)[:, 1]
    original_auroc = average_precision_score(test_y, y_prob_orig)
    print(f"Original Test AUROC (all features): {original_auroc:.4f}")
    
    # Split training data for pruning validation
    train_sub, val_sub, y_sub, y_val = train_test_split(
        train_set, train_y, test_size=0.2, stratify=train_y, random_state=42
    )
    
    # Train base model for permutation importance
    base_model = RandomForestClassifier(
        n_estimators=500,
        bootstrap=True,
        class_weight={0: 1, 1: 2},
        random_state=42,
        max_depth=10
    )
    base_model.fit(train_sub, y_sub)
    
    # Calculate permutation importance on validation set
    result = permutation_importance(
        base_model, val_sub, y_val, 
        n_repeats=5, scoring='roc_auc', random_state=42
    )
    importances = result.importances_mean
    
    # Select important features
    threshold = 0.0005
    important_indices = np.where(importances > threshold)[0]
    num_dropped = train_set.shape[1] - len(important_indices)
    
    if len(important_indices) == 0:
        print("⚠️ No important features found above threshold; using all features.")
        important_indices = np.arange(train_set.shape[1])
        num_dropped = 0
    
    print(f"Dropped {num_dropped} feature(s) below importance threshold {threshold}")
    
    # Filter feature sets
    train_reduced = train_set[:, important_indices]
    test_reduced = test_set[:, important_indices]
    selected_cols = [cols[i] for i in important_indices] if cols is not None else None
    
    # Retrain final model on reduced features
    model = RandomForestClassifier(
        n_estimators=1000,
        bootstrap=True,
        class_weight={0: 1, 1: 2},
        random_state=42,
        max_depth=10
    )
    model.fit(train_reduced, train_y)
    
    # Print top feature importances
    # if selected_cols is not None:
    #     feature_importances = model.feature_importances_
    #     sorted_indices = np.argsort(feature_importances)[::-1]
    #     print("Retained Feature importances:")
    #     for idx in sorted_indices[:10]:
    #         print(f"{selected_cols[idx]}: {feature_importances[idx]:.4f}")
    
    # Calculate predictions and AUROC
    y_prob = model.predict_proba(test_reduced)[:, 1]
    train_y_prob = model.predict_proba(train_reduced)[:, 1]
    auroc = average_precision_score(test_y, y_prob)
    train_auroc = average_precision_score(train_y, train_y_prob)
    
    print(f"\nNew Test AUROC (pruned): {auroc:.4f}", f"Train AUROC (pruned): {train_auroc:.4f} \n")
    
    return y_prob


def run_rf_auroc(train_set, train_y, test_set, test_y, cols=None):
    """
    Train Random Forest classifier and evaluate performance.
    
    Parameters:
    -----------
    train_set : array-like
        Training features
    train_y : array-like
        Training labels
    test_set : array-like
        Test features
    test_y : array-like
        Test labels
    cols : list, optional
        Column names for feature importance reporting
        
    Returns:
    --------
    array : Predicted probabilities for test set
    """
    # Clean NaNs and infs
    train_set = np.nan_to_num(train_set, nan=0.0, posinf=0.0, neginf=0.0)
    test_set = np.nan_to_num(test_set, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Configure and train Random Forest
    model = RandomForestClassifier(
        n_estimators=1000,
        bootstrap=True,
        class_weight={0: 1, 1: 2},  # Handle class imbalance
        random_state=42,
        max_depth=10
    )
    model.fit(train_set, train_y)
    
    # Print top feature importances
    # if cols is not None:
    #     feature_importances = model.feature_importances_
    #     sorted_indices = np.argsort(feature_importances)[::-1]
    #     print("Feature importances:")
    #     for idx in sorted_indices[:10]:
    #         print(f"{cols[idx]}: {feature_importances[idx]:.4f}")
    
    # Get predicted probabilities
    y_prob = model.predict_proba(test_set)[:, 1]
    train_y_prob = model.predict_proba(train_set)[:, 1]
    
    # Compute AUROC for both test and training sets
    auroc = average_precision_score(test_y, y_prob)
    train_auroc = average_precision_score(train_y, train_y_prob)
    
    print(f"\n Test AUROC: {auroc:.4f}", f"Train AUROC: {train_auroc:.4f} \n")
    
    return y_prob





def run_cross_validation(df, df_eeg, model1_cols, model2_cols):
    """
    Perform 4-fold cross-validation for both baseline models.
    
    Parameters:
    -----------
    df : DataFrame
        Sleep stage features dataset
    df_eeg : DataFrame
        EEG features dataset
    model1_cols : list
        Column names for sleep stage features
    model2_cols : list
        Column names for EEG features
        
    Returns:
    --------
    tuple : (result_sleep_stage, result_eeg)
    """
    result_sleep_stage = pd.DataFrame()
    result_eeg = pd.DataFrame()
    
    print(f"Dataset shape: {df.shape}")
    
    for fold in range(4):
        print(f'\n=== Fold {fold} ===')
        
        # Split data: train on folds != current_fold and dataset != 'wsc', test on current_fold + 'wsc'
        train_mask = (df['fold'] != fold) & (df['dataset'] != 'wsc')
        test_mask = (df['fold'] == fold) | (df['dataset'] == 'wsc')
        
        # Sleep stage model data
        train_set = df[train_mask].copy()
        test_set = df[test_mask].copy()
        train_y = train_set['label'].values
        test_y = test_set['label'].values
        
        # EEG model data
        train_set_eeg = df_eeg[train_mask].copy()
        test_set_eeg = df_eeg[test_mask].copy()
        
        # Store metadata
        test_set_datasets = test_set['dataset'].values
        test_set_labels = test_set['label'].values
        
        # Prepare feature matrices
        train_features = train_set.drop(columns=['filename', 'fold', 'dataset', 'label']).values
        test_features = test_set.drop(columns=['filename', 'fold', 'dataset', 'label']).values
        
        train_features_eeg = train_set_eeg.drop(columns=['filename', 'fold', 'dataset', 'label']).values
        test_features_eeg = test_set_eeg.drop(columns=['filename', 'fold', 'dataset', 'label']).values
        
        # Train and evaluate sleep stage model
        print('Sleep Stage Model:')
        probs_sleep = run_rf_auroc(train_features, train_y, test_features, test_y, cols=model1_cols)
        bootstrap_auroc_ci(test_y, probs_sleep)
        
        # Store results
        fold_result_sleep = pd.DataFrame({
            'prob': probs_sleep,
            'label': test_set_labels,
            'dataset': test_set_datasets,
            'fold': fold
        })
        result_sleep_stage = pd.concat([result_sleep_stage, fold_result_sleep], ignore_index=True)
        
        # Train and evaluate EEG model
        print('EEG Model:')
        probs_eeg = run_rf_auroc(train_features_eeg, train_y, test_features_eeg, test_y, cols=model2_cols)
        bootstrap_auroc_ci(test_y, probs_eeg)
        
        # Store results
        fold_result_eeg = pd.DataFrame({
            'prob': probs_eeg,
            'label': test_set_labels,
            'dataset': test_set_datasets,
            'fold': fold
        })
        result_eeg = pd.concat([result_eeg, fold_result_eeg], ignore_index=True)
    
    return result_sleep_stage, result_eeg


def evaluate_per_dataset_performance(result_sleep, result_eeg, labels_model_baseline):
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
    
    for dataset in result_sleep['dataset'].unique():
        print(f'\n--- {dataset.upper()} Dataset ---')
        
        # Sleep Stage Model
        dataset_mask = result_sleep['dataset'] == dataset
        print('Baseline AUPRC: ', result_sleep[dataset_mask]['label'].mean())
        print(f'{dataset} Sleep Stage Model:')
        auroc = average_precision_score(
            result_sleep[dataset_mask]['label'], 
            result_sleep[dataset_mask]['prob']
        )
        bootstrap_auroc_ci(
            result_sleep[dataset_mask]['label'], 
            result_sleep[dataset_mask]['prob']
        )
        
        # Per-fold AUROC
        fold_aurocs = []
        for fold in range(4):
            fold_mask = dataset_mask & (result_sleep['fold'] == fold)
            if fold_mask.sum() > 0:
                fold_aurocs.append(average_precision_score(
                    result_sleep[fold_mask]['label'], 
                    result_sleep[fold_mask]['prob']
                ))
        
        print(f'AUROC: {auroc:.4f}')
        print(f'Fold AUROC: {fold_aurocs}')
        
        # EEG Model
        print(f'\n{dataset} EEG Model:')
        dataset_mask_eeg = result_eeg['dataset'] == dataset
        auroc_eeg = average_precision_score(
            result_eeg[dataset_mask_eeg]['label'], 
            result_eeg[dataset_mask_eeg]['prob']
        )
        bootstrap_auroc_ci(
            result_eeg[dataset_mask_eeg]['label'], 
            result_eeg[dataset_mask_eeg]['prob']
        )
        
        # Per-fold AUROC for EEG
        fold_aurocs_eeg = []
        for fold in range(4):
            fold_mask_eeg = dataset_mask_eeg & (result_eeg['fold'] == fold)
            if fold_mask_eeg.sum() > 0:
                fold_aurocs_eeg.append(average_precision_score(
                    result_eeg[fold_mask_eeg]['label'], 
                    result_eeg[fold_mask_eeg]['prob']
                ))
        
        print(f'AUROC: {auroc_eeg:.4f}')
        print(f'Fold AUROC: {fold_aurocs_eeg}')
        
        # Our Model
        print(f'\n{dataset} Our Model:')
        dataset_mask_our = labels_model_baseline['dataset'] == dataset
        if dataset_mask_our.sum() > 0:
            auroc_our = average_precision_score(
                labels_model_baseline[dataset_mask_our]['label'], 
                labels_model_baseline[dataset_mask_our]['prob']
            )
            bootstrap_auroc_ci(
                labels_model_baseline[dataset_mask_our]['label'], 
                labels_model_baseline[dataset_mask_our]['prob']
            )
            
            # Per-fold AUROC for Our Model (skip WSC as it doesn't have folds)
            fold_aurocs_our = []
            if dataset != 'wsc':
                for fold in range(4):
                    fold_mask_our = dataset_mask_our & (labels_model_baseline['fold'] == fold)
                    if fold_mask_our.sum() > 0:
                        fold_aurocs_our.append(average_precision_score(
                            labels_model_baseline[fold_mask_our]['label'], 
                            labels_model_baseline[fold_mask_our]['prob']
                        ))
            
            print(f'AUROC: {auroc_our:.4f}')
            print(f'Fold AUROC: {fold_aurocs_our}')
        
        print()
    
    # Additional datasets for our model only
    for dataset in ['hchs', 'rf']:
        if dataset in labels_model_baseline['dataset'].values:
            print(f'{dataset} Our Model:')
            dataset_mask = labels_model_baseline['dataset'] == dataset
            print('Baseline AUPRC: ', labels_model_baseline[dataset_mask]['label'].mean())
            auroc = average_precision_score(
                labels_model_baseline[dataset_mask]['label'], 
                labels_model_baseline[dataset_mask]['prob']
            )
            bootstrap_auroc_ci(
                labels_model_baseline[dataset_mask]['label'], 
                labels_model_baseline[dataset_mask]['prob']
            )
            
            # Per-fold AUROC
            fold_aurocs = []
            for fold in range(4):
                fold_mask = dataset_mask & (labels_model_baseline['fold'] == fold)
                if fold_mask.sum() > 0:
                    fold_aurocs.append(average_precision_score(
                        labels_model_baseline[fold_mask]['label'], 
                        labels_model_baseline[fold_mask]['prob']
                    ))
            
            print(f'AUROC: {auroc:.4f}')
            print(f'Fold AUROC: {fold_aurocs}')
            print()


def evaluate_overall_performance(result_sleep, result_eeg, labels_model_baseline):
    """
    Evaluate and print overall performance across all datasets.
    
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
    print('OVERALL PERFORMANCE EVALUATION')
    print('='*50)
    
    # Sleep Stage Model Overall
    print('Overall Sleep Stage Model:')
    auroc_sleep = average_precision_score(result_sleep['label'], result_sleep['prob'])
    print(f'AUROC: {auroc_sleep:.4f}')
    bootstrap_auroc_ci(result_sleep['label'], result_sleep['prob'])
    
    # EEG Model Overall
    print('\nOverall EEG Model:')
    auroc_eeg = average_precision_score(result_eeg['label'], result_eeg['prob'])
    print(f'AUROC: {auroc_eeg:.4f}')
    bootstrap_auroc_ci(result_eeg['label'], result_eeg['prob'])
    
    # Our Model Overall
    print('\nOverall Our Model:')
    auroc_our = average_precision_score(labels_model_baseline['label'], labels_model_baseline['prob'])
    print(f'AUROC: {auroc_our:.4f}')
    bootstrap_auroc_ci(labels_model_baseline['label'], labels_model_baseline['prob'])


def main():
    """
    Main analysis pipeline.
    """
    print("="*60)
    print("BASELINE RANDOM FOREST ANALYSIS")
    print("Antidepressant Prediction from Sleep Studies")
    print("="*60)
    
    # Load and prepare data
    print("\n1. Loading and preparing datasets...")
    df, df_eeg, labels_model_baseline, model1_cols, model2_cols = load_and_prepare_data()
    
    # Run cross-validation
    print("\n2. Running 4-fold cross-validation...")
    result_sleep_stage, result_eeg = run_cross_validation(df, df_eeg, model1_cols, model2_cols)
    
    # Evaluate per-dataset performance
    print("\n3. Evaluating per-dataset performance...")
    evaluate_per_dataset_performance(result_sleep_stage, result_eeg, labels_model_baseline)
    
    # Evaluate overall performance 
    # print("\n4. Evaluating overall performance...")
    # evaluate_overall_performance(result_sleep_stage, result_eeg, labels_model_baseline)
    
    print('\n' + '='*60)
    print('ANALYSIS COMPLETE')
    print('='*60)


if __name__ == "__main__":
    main()