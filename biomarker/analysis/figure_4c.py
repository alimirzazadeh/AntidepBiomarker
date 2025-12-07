"""
Dosage Ablation Analysis for Antidepressant Prediction Model
===========================================================

This script analyzes the relationship between antidepressant dosage and model predictions,
generating Figure 4c for the paper submission. The analysis examines both per-night and 
per-patient predictions across different dosage ranges.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# Configuration
INFERENCE_FILE = '../../data/inference_v6emb_3920_all_dosagev2.csv'
TAXONOMY_FILE = '../../data/antidep_taxonomy_all_datasets_v6.csv'
OUTPUT_FILE = 'figure_4c.png'
DOSAGE_COL = 'dosage_v2' if 'dosagev2' in INFERENCE_FILE else 'dosage'

# Dosage binning configuration
DOSAGE_BREAKS = [0, 0.25, 0.75, 1.75, 2.75, np.inf]
DOSAGE_LABELS = ['0', '0.25-0.75', '0.75-1.75', '1.75-2.75', '2.75+']

def load_and_preprocess_data():
    """
    Load inference results and taxonomy data, then merge and preprocess.
    
    Returns:
        pd.DataFrame: Merged and preprocessed dataframe
    """
    # Load inference results
    df = pd.read_csv(INFERENCE_FILE)
    
    # Filter for RF dataset only
    df_rf = df[df['dataset'] == 'rf'].copy()
    
    # Convert logits to probabilities using sigmoid
    df_rf['pred'] = 1 / (1 + np.exp(-df_rf['pred']))
    
    # Extract filename from full path
    df_rf['filename'] = df_rf['filename'].apply(lambda x: x.split('/')[-1])
    
    # Load taxonomy data
    df_taxonomy = pd.read_csv(TAXONOMY_FILE)
    
    # Merge datasets
    df_merged = pd.merge(df_rf, df_taxonomy, on='filename', how='inner')
    
    return df_merged

def create_dosage_bins(df):
    """
    Create dosage bins for analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe with dosage column
        
    Returns:
        pd.DataFrame: Dataframe with dosage bins added
    """
    df = df.copy()
    df['Dosage (Proportion of Standard Dose)'] = pd.cut(
        df[DOSAGE_COL], 
        bins=DOSAGE_BREAKS, 
        labels=DOSAGE_LABELS, 
        right=False
    )
    return df

def analyze_dosage_distribution(df):
    """
    Print dosage distribution statistics for each unique dosage value.
    
    Args:
        df (pd.DataFrame): Per-patient aggregated dataframe
    """
    print("Dosage Distribution Analysis:")
    print("=" * 50)
    
    for dose in np.sort(df[DOSAGE_COL].unique()):
        subset = df[df[DOSAGE_COL] == dose]
        print(f'Dose: {dose:>6.2f} | '
              f'Count: {subset.shape[0]:>3d} | '
              f'Mean: {subset["pred"].mean():.3f} | '
              f'Std: {subset["pred"].std():.3f}')

def create_visualization(df_per_night, df_per_patient):
    """
    Create the dosage ablation visualization.
    
    Args:
        df_per_night (pd.DataFrame): Per-night data with dosage bins
        df_per_patient (pd.DataFrame): Per-patient data with dosage bins
    """
    # Calculate summary statistics
    total_nights = df_per_night.shape[0]
    total_patients = df_per_patient.shape[0]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    
    # Create color palette
    n_categories = len(df_per_patient['Dosage (Proportion of Standard Dose)'].unique())
    palette = sns.color_palette("Greys", n_colors=n_categories)[::-1]
    
    # Create boxplot for per-patient data
    sns.boxplot(
        x='Dosage (Proportion of Standard Dose)', 
        y='pred', 
        data=df_per_patient, 
        palette=palette, 
        ax=ax, 
        showfliers=False
    )
    
    # Set labels and title
    ax.set_title(f'Model Score Per Patient (N={total_patients} patients, {total_nights} nights)')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Model Score')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    
    print(f"\nFigure saved to: {OUTPUT_FILE}")

def main():
    """
    Main analysis pipeline.
    """
    print("Starting Dosage Ablation Analysis...")
    print("=" * 50)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    print(f"Loaded {df.shape[0]} records from RF dataset")
    
    # Create per-night analysis (with dosage bins)
    df_per_night = create_dosage_bins(df)
    
    # Create per-patient analysis (median aggregation)
    df_per_patient = df.groupby(['pid', DOSAGE_COL, 'taxonomy']).agg({
        'pred': 'median'
    }).reset_index()
    
    # Add dosage bins to per-patient data
    df_per_patient = create_dosage_bins(df_per_patient)
    
    # Print dosage distribution statistics
    analyze_dosage_distribution(df_per_patient)
    
    # Create visualization
    create_visualization(df_per_night, df_per_patient)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()