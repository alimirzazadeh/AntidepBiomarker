"""
Antidepressant Types Ablation Analysis for Sleep Study Model
===========================================================

This script analyzes model performance across different antidepressant medications
and medication classes, generating results for Figure 4 of the paper submission.
The analysis computes AUC, sensitivity, specificity, PPV, and NPV for each 
medication type versus controls.

"""

import os
import numpy as np
import pandas as pd
from ipdb import set_trace as bp
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# Configuration
# INFERENCE_FILE = '../../data/inference_v6emb_3920_all.csv'
# INFERENCE_FILE = '../../data/nodulox_inference_v6emb_3920_all.csv'
INFERENCE_FILE = '../../data/inference_v6emb_3920_all_novenlafaxine.csv'

TAXONOMY_FILE = '../../data/antidep_taxonomy_all_datasets_v6.csv'

# Analysis configuration
EVALUATE_BY_DATASET = True 
ONLY_SINGLE_MEDICATION = True
THRESHOLD = 0.25
N_BOOTSTRAP = 1000
RANDOM_STATE = 42

# Medication mappings
MEDICATION_MAPPING = {
    'NSEE': 'Escitalopram',
    'NSEC': 'Citalopram', 
    'NSFx': 'Fluoxetine',
    'NSSx': 'Sertraline',
    'NSPx': 'Paroxetine',
    'NNVx': 'Venlafaxine',
    'NNDx': 'Desvenlafaxine',
    'NNUx': 'Duloxetine',
    'NVxx': 'Vortioxetine',
    'NMxx': 'Mirtazapine',
    'NBxx': 'Bupropion',
    'TAxx': 'Amitryptiline',
    'TIxx': 'Imipramine',
    'TNxx': 'Nortriptyline',
    'TDxx': 'Doxepin'
}

MEDICATION_TYPE_MAPPING = {
    'NS': 'SSRI',
    'NN': 'SNRI', 
    'NV': 'Vortioxetine',
    'NM': 'Mirtazapine',
    'NB': 'Bupropion',
    'T': 'TCA'
}

def load_and_preprocess_data():
    """
    Load inference results and taxonomy data, then merge and preprocess.
    
    Returns:
        pd.DataFrame: Preprocessed dataframe ready for analysis
    """
    # Load inference results
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
    
    # Load and merge taxonomy data
    df_taxonomy = pd.read_csv(TAXONOMY_FILE)
    df = pd.merge(df, df_taxonomy, on='filename', how='inner')
    
    # Group by patient and taxonomy, taking mean of predictions
    df = df.groupby(['pid', 'taxonomy'], as_index=False).agg(
        lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0]
    )
    
    print(f'Total Patient-Taxonomy Combinations: {df.shape[0]}')
    return df

def calculate_ppv_npv_bootstrap(labels, y_pred, n_boot=N_BOOTSTRAP, 
                               threshold=THRESHOLD, random_state=RANDOM_STATE):
    """
    Calculate PPV and NPV with bootstrap confidence intervals.
    
    Args:
        labels (array): True binary labels
        y_pred (array): Predicted probabilities or binary predictions
        n_boot (int): Number of bootstrap iterations
        threshold (float): Threshold for binary classification
        random_state (int): Random seed for reproducibility
    """
    rng = np.random.default_rng(random_state)
    
    labels = np.asarray(labels)
    y_pred = np.asarray(y_pred)
    
    # Get positive and negative indices
    positives = np.where(labels == 1)[0]
    negatives = np.where(labels == 0)[0]
    n_pos = len(positives)
    
    ppv_list = []
    npv_list = []
    
    # Bootstrap sampling
    for _ in range(n_boot):
        pos_idx = rng.choice(positives, size=n_pos, replace=True)
        neg_idx = rng.choice(negatives, size=n_pos, replace=True)  # Balanced sampling
        
        idx = np.concatenate([pos_idx, neg_idx])
        y_true = labels[idx]
        y_prob = y_pred[idx]
        y_bin = (y_prob >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel()
        
        # Calculate PPV and NPV
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        
        ppv_list.append(ppv)
        npv_list.append(npv)
    
    # Calculate statistics
    ppv_mean = np.round(np.nanmean(ppv_list), 2)
    npv_mean = np.round(np.nanmean(npv_list), 2)
    ppv_95ci = tuple(np.round(np.nanpercentile(ppv_list, [2.5, 97.5]), 2))
    npv_95ci = tuple(np.round(np.nanpercentile(npv_list, [2.5, 97.5]), 2))
    
    print(f'    PPV: {ppv_mean} ({ppv_95ci[0]} - {ppv_95ci[1]})')
    print(f'    NPV: {npv_mean} ({npv_95ci[0]} - {npv_95ci[1]})')

def calculate_sensitivity_specificity(y_true, y_pred):
    """
    Calculate sensitivity and specificity from binary predictions.
    
    Args:
        y_true (array): True binary labels
        y_pred (array): Predicted binary labels
        
    Returns:
        tuple: (sensitivity, specificity)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity, specificity

def analyze_single_medications(df):
    """
    Analyze performance for individual medications (single medication only).
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
    """
    print("Individual Medication Analysis")
    print("=" * 50)
    
    # Get available single medications (no combinations)
    available_meds = df['taxonomy'].unique()
    available_meds = [item for item in available_meds if ',' not in item and item != 'C']
    print(f"Available medications: {available_meds}")
    
    controls = df[df['taxonomy'] == 'C']['pred'].values
    
    for med_code in MEDICATION_MAPPING:
        if med_code not in available_meds:
            print(f'{med_code} not available in dataset')
            continue
        
        # Get medication predictions
        med_vals = df[df['taxonomy'] == med_code]['pred'].values
        
        # Prepare labels and predictions for analysis
        labels = np.concatenate([np.zeros_like(controls), np.ones_like(med_vals)])
        preds = np.concatenate([controls, med_vals])
        
        # Calculate AUC
        auc = np.round(roc_auc_score(labels, preds), 3)
        med_name = MEDICATION_MAPPING[med_code]
        
        print(f'{med_name}: AUC = {auc:.3f}, N = {len(med_vals)}')
        
        # Calculate performance metrics at fixed threshold
        y_pred = (preds >= THRESHOLD).astype(int)
        sens, spec = calculate_sensitivity_specificity(labels, y_pred)
        
        print(f'  - Threshold {THRESHOLD}: Sensitivity = {sens:.3f}, Specificity = {spec:.3f}')
        calculate_ppv_npv_bootstrap(labels, y_pred)
        print()

def analyze_medication_types(df):
    """
    Analyze performance for medication types/classes.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
    """
    print("\nMedication Type Analysis")
    print("=" * 50)
    
    controls = df[df['taxonomy'] == 'C']['pred'].values
    
    for type_code, type_name in MEDICATION_TYPE_MAPPING.items():
        # Create boolean mask for medication type
        df[type_code] = df['taxonomy'].apply(
            lambda x: x.startswith(type_code) and ',' not in x
        )
        
        med_type_vals = df[df[type_code] == True]['pred'].values
        
        if len(med_type_vals) == 0:
            print(f'{type_name}: No data available')
            continue
        
        # Prepare labels and predictions
        labels = np.concatenate([np.zeros_like(controls), np.ones_like(med_type_vals)])
        preds = np.concatenate([controls, med_type_vals])
        
        # Calculate AUC
        auc = np.round(roc_auc_score(labels, preds), 3)
        print(f'{type_name}: AUC = {auc:.3f}, N = {len(med_type_vals)}')
        
        # Calculate performance metrics at fixed threshold
        y_pred = (preds >= THRESHOLD).astype(int)
        sens, spec = calculate_sensitivity_specificity(labels, y_pred)
        
        print(f'  - Threshold {THRESHOLD}: Sensitivity = {sens:.3f}, Specificity = {spec:.3f}')
        calculate_ppv_npv_bootstrap(labels, y_pred)
        
        # Dataset-specific analysis
        print(f'  - Dataset-specific results:')
        for dataset in df['dataset'].unique():
            dataset_controls = df[
                (df['taxonomy'] == 'C') & (df['dataset'] == dataset)
            ]['pred'].values
            
            dataset_med_vals = df[
                (df[type_code] == True) & (df['dataset'] == dataset)
            ]['pred'].values
            
            if len(dataset_med_vals) == 0:
                continue
            
            dataset_labels = np.concatenate([
                np.zeros_like(dataset_controls), 
                np.ones_like(dataset_med_vals)
            ])
            dataset_preds = np.concatenate([dataset_controls, dataset_med_vals])
            
            dataset_auc = np.round(roc_auc_score(dataset_labels, dataset_preds), 3)
            print(f'    {dataset}: AUC = {dataset_auc:.3f}, N = {len(dataset_med_vals)}')
        
        print()

def analyze_combination_medications(df):
    """
    Analyze performance including combination medications.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
    """
    print("\nCombination Medication Analysis")
    print("=" * 50)
    
    controls = df[df['taxonomy'] == 'C']['pred'].values
    
    # Individual medications (including combinations)
    for med_code, med_name in MEDICATION_MAPPING.items():
        # Create mask for medications (including combinations)
        med_mask = []
        for taxonomy in df['taxonomy'].values:
            med_list = taxonomy.split(',')
            med_mask.append(med_code in med_list)
        
        if not any(med_mask):
            print(f'{med_name}: Not found in combinations')
            continue
        
        med_vals = df[med_mask]['pred'].values
        labels = np.concatenate([np.zeros_like(controls), np.ones_like(med_vals)])
        preds = np.concatenate([controls, med_vals])
        
        auc = np.round(roc_auc_score(labels, preds), 3)
        print(f'{med_name}: AUC = {auc:.3f}, N = {len(med_vals)}')
    
    print()
    
    # Medication types (including combinations)
    for type_code, type_name in MEDICATION_TYPE_MAPPING.items():
        med_mask = []
        for taxonomy in df['taxonomy'].values:
            med_list = taxonomy.split(',')
            type_match = any(med.startswith(type_code) for med in med_list)
            med_mask.append(type_match)
        
        if not any(med_mask):
            print(f'{type_name}: Not found in combinations')
            continue
        
        med_vals = df[med_mask]['pred'].values
        labels = np.concatenate([np.zeros_like(controls), np.ones_like(med_vals)])
        preds = np.concatenate([controls, med_vals])
        
        auc = np.round(roc_auc_score(labels, preds), 3)
        print(f'{type_name}: AUC = {auc:.3f}, N = {len(med_vals)}')

def main():
    """
    Main analysis pipeline.
    """
    print("Starting Antidepressant Types Ablation Analysis...")
    print("=" * 60)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    if EVALUATE_BY_DATASET:
        for dataset in df['dataset'].unique():
            print(f"\nAnalyzing {dataset} dataset")
            dataset_df = df[df['dataset'] == dataset]
            try:
                analyze_single_medications(dataset_df)
                # analyze_medication_types(dataset_df)
            except Exception as e:
                print(f"Error analyzing {dataset} dataset: {e}")
                continue
    bp() 
    
    if ONLY_SINGLE_MEDICATION:
        print(f"\nAnalyzing single medications only (threshold = {THRESHOLD})")
        analyze_single_medications(df)
        analyze_medication_types(df)
    else:
        print(f"\nAnalyzing all medications including combinations")
        analyze_combination_medications(df)
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()