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
import dataframe_image as dfi
import scipy
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
    labels_model_baseline = pd.merge(labels, df_taxonomy, on='filename', how='inner').reset_index()
    labels_model_baseline = labels_model_baseline.groupby(['pid', 'taxonomy']).agg({
        'pred': 'mean', 
        'dataset': 'first', 
        'label': 'first', 
        'fold': 'first',
        'filename': 'first'
    }).reset_index()
    # Convert logits to probabilities
    labels_model_baseline['prob'] = 1 / (1 + np.exp(-labels_model_baseline['pred']))
    
    # Merge labels with feature datasets
    labels_subset = labels[['filename', 'label', 'fold', 'dataset', 'mit_gender', 'mit_age']]
    df = df.merge(labels_subset, on='filename', how='inner')
    df_eeg = df_eeg.merge(labels_subset, on='filename', how='inner')
    
    return df, df_eeg, labels_model_baseline, model1_cols, model2_cols



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
        test_set_filenames = test_set['filename'].values
        
        # Prepare feature matrices
        train_features = train_set.drop(columns=['filename', 'fold', 'dataset', 'label']).values
        test_features = test_set.drop(columns=['filename', 'fold', 'dataset', 'label']).values
        
        train_features_eeg = train_set_eeg.drop(columns=['filename', 'fold', 'dataset', 'label']).values
        test_features_eeg = test_set_eeg.drop(columns=['filename', 'fold', 'dataset', 'label']).values
        test_features_eeg_filenames = test_set_eeg['filename'].values
        # Train and evaluate sleep stage model
        print('Sleep Stage Model:')
        probs_sleep = run_rf_auroc(train_features, train_y, test_features, test_y, cols=model1_cols)

        
        # Store results
        fold_result_sleep = pd.DataFrame({
            'prob': probs_sleep,
            'label': test_set_labels,
            'dataset': test_set_datasets,
            'fold': fold,
            'filename': test_set_filenames
        })
        result_sleep_stage = pd.concat([result_sleep_stage, fold_result_sleep], ignore_index=True)
        
        # Train and evaluate EEG model
        print('EEG Model:')
        probs_eeg = run_rf_auroc(train_features_eeg, train_y, test_features_eeg, test_y, cols=model2_cols)
        
        # Store results
        fold_result_eeg = pd.DataFrame({
            'prob': probs_eeg,
            'label': test_set_labels,
            'dataset': test_set_datasets,
            'fold': fold,
            'filename': test_features_eeg_filenames
        })
        result_eeg = pd.concat([result_eeg, fold_result_eeg], ignore_index=True)
    
    return result_sleep_stage, result_eeg


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
    
    result_sleep_stage.rename(columns={'prob': 'sleep_stage_prob'}, inplace=True)
    result_eeg.rename(columns={'prob': 'eeg_prob'}, inplace=True)
    result_sleep_stage = result_sleep_stage.merge(result_eeg[['filename', 'fold', 'eeg_prob']], on=['filename', 'fold'], how='inner')

    # Evaluate per-dataset performance
    df_master = pd.read_csv('../data/master_dataset.csv')

    ## merge by pid with same dosage 
    df_master = df_master[['pred', 'rem_latency_gt', 'filename','fold']]
    df_master2 = df_master.merge(result_sleep_stage[['filename', 'fold', 'sleep_stage_prob','eeg_prob']], on=['filename', 'fold'], how='inner')
    df_master2.dropna(inplace=True)
    print('Pearson correlation between sleep stage probability and rem latency: ', scipy.stats.pearsonr(df_master2['sleep_stage_prob'], df_master2['rem_latency_gt']).statistic)
    print('Pearson correlation between eeg probability and rem latency: ', scipy.stats.pearsonr(df_master2['eeg_prob'], df_master2['rem_latency_gt']).statistic)
    print('Pearson correlation between model probability and rem latency: ', scipy.stats.pearsonr(df_master2['pred'], df_master2['rem_latency_gt']).statistic)
    ## now repeat with logs of rem latency 
    df_master2['rem_latency_gt_log'] = np.log(df_master2['rem_latency_gt'] + 1)
    corr, pval = scipy.stats.pearsonr(df_master2['sleep_stage_prob'], df_master2['rem_latency_gt_log'])
    print('Pearson correlation between sleep stage probability and log rem latency: ', corr, 'with p-value: ', pval)
    corr, pval = scipy.stats.pearsonr(df_master2['eeg_prob'], df_master2['rem_latency_gt_log'])
    print('Pearson correlation between eeg probability and log rem latency: ', corr, 'with p-value: ', pval)
    corr, pval = scipy.stats.pearsonr(df_master2['pred'], df_master2['rem_latency_gt_log'])
    print('Pearson correlation between model probability and log rem latency: ', corr, 'with p-value: ', pval)
    # df = df.groupby(['pid','dosage']).agg({'mit_age': 'mean'}).reset_index()
        
    # evaluate_per_dataset_performance(result_sleep_stage, result_eeg, labels_model_baseline)
    
    # Evaluate overall performance 
    # print("\n4. Evaluating overall performance...")
    # evaluate_overall_performance(result_sleep_stage, result_eeg, labels_model_baseline)
    
    print('\n' + '='*60)
    print('ANALYSIS COMPLETE')
    print('='*60)


if __name__ == "__main__":
    main()