import os
import pandas as pd
import numpy as np
from ipdb import set_trace as bp
import sys 
sys.path.append('./')
from biomarker.analysis.figure_4d import process_mros_medications, process_wsc_medications
CSV_DIR = 'data/'
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def get_significance_stars(pval):
    if pval < 1e-10:
        return '***'
    elif pval < 0.001:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'ns'
INFERENCE_FILE = os.path.join(CSV_DIR,'inference_v6emb_3920_all.csv')
TAXONOMY_FILE = os.path.join(CSV_DIR,'antidep_taxonomy_all_datasets_v6.csv')

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
    df['is_tca'] = df['taxonomy'].apply(lambda x: 1 if x.startswith('T') or ',T' in x else 0)
    df['is_ntca'] = df['taxonomy'].apply(lambda x: 1 if x.startswith('N') or ',N' in x else 0)
    df['is_snri'] = df['taxonomy'].apply(lambda x: 1 if x.startswith('NN') or ',NN' in x else 0)
    df['is_ssri'] = df['taxonomy'].apply(lambda x: 1 if x.startswith('NS') or ',NS' in x else 0)
    # df = df[(df['is_snri'] == 1) | (df['label'] == 0)]
    df = df[(df['is_snri'] == 1) | (df['is_ssri'] == 1) | (df['is_ntca'] == 1) | (df['is_tca'] == 1) | (df['label'] == 0)]
    # Group by patient and taxonomy, taking mean of predictions
    # df = df.groupby(['pid', 'taxonomy'], as_index=False).agg(
    #     lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0]
    # )
    
    print(f'Total Patient-Taxonomy Combinations: {df.shape[0]}')
    return df

def generate_cotherapy_analysis_figure(save=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    df = load_and_preprocess_data()
    controls = df[df['label'] == 0].copy()
    all_antidep = df[df['label'] == 1].copy()
    multi_therapy = df[df['taxonomy'].str.contains(',')].copy()
    single_therapy = df[(~df['taxonomy'].str.contains(',')) & (df['label'] != 0)].copy()

    print('Multi-therapy: ', multi_therapy.shape[0])
    print('Single-therapy: ', single_therapy.shape[0])

    df_mros_meds = process_mros_medications(EXP_FOLDER = CSV_DIR)
    df_wsc_meds = process_wsc_medications(EXP_FOLDER = CSV_DIR)

    # Combine medication data
    df_other = pd.concat([df_mros_meds, df_wsc_meds])
    print('Before merge data: ', df.shape[0])
    df = df.merge(df_other, on='filename', how='inner')
    print('Total merged data: ', df.shape[0])


    antidep_plus_benzo = df[(df['label'] == 1) & (df['benzos'] == True)].copy()
    antidep_plus_anticonvulsant = df[(df['label'] == 1) & (df['convuls'] == True)].copy()
    antidep_plus_antipsychotic = df[(df['label'] == 1) & (df['antipsycho'] == True)].copy()

    g0 = controls['pred'].values 
    g1 = single_therapy['pred'].values 
    g2 = multi_therapy['pred'].values 
    g3 = antidep_plus_benzo['pred'].values 
    g4 = antidep_plus_anticonvulsant['pred'].values 
    g5 = antidep_plus_antipsychotic['pred'].values 
    cohort = len(g0) * ['Controls'] + len(g1) * ['Single\nAntidep'] + len(g2) * ['Multi-\nAntidep'] + len(g3) * ['Antidep+\nBenzodiazepine'] + len(g4) * ['Antidep+\nAnticonvulsant'] + len(g5) * ['Antidep+\nAntipsychotic']
    df_plot = pd.DataFrame({
        'cohort': cohort,
        'pred': np.concatenate([g0, g1, g2, g3, g4, g5])
    })
    order = ['Controls', 'Single\nAntidep', 'Antidep+\nAnticonvulsant', 'Antidep+\nAntipsychotic', 'Antidep+\nBenzodiazepine', 'Multi-\nAntidep']
    sns.boxplot(x='cohort', y='pred', data=df_plot, showfliers=False, palette='Greens', order=order, ax=ax)
        # sns.boxplot(
        #     x="medication", y="pred", data=df_viz, 
        #     palette='Greens', ax=ax, order=order, showfliers=False
        # )
    ax.set_ylabel('Model Score')
    ax.set_xlabel('Cohort')


    # Get box positions for bracket drawing
    box_positions = ax.get_xticks()
    
    # Get control values (first group)
    control_values = df_plot[df_plot['cohort'] == 'Controls']['pred'].values
    control_idx = order.index('Controls')
    
    for i, cohort in enumerate(order):
        subset = df_plot[df_plot['cohort'] == cohort]
        ## print the median and the Q1 and Q3
        median = subset['pred'].median()
        q1 = subset['pred'].quantile(0.25)
        q3 = subset['pred'].quantile(0.75)
        print(f'{cohort}: Median={median:.2f}, IQR: [{q1:.2f}-{q3:.2f}]')
        n = subset.shape[0]
        ax.text(i, median + 0.01, f'N={n}', ha='center', va='bottom', fontsize=10)
    
    # Add significance brackets comparing each group to Controls
    bracket_offset = 0
    for i, cohort in enumerate(order):
        if cohort == 'Controls':
            continue  # Skip Controls itself
        
        cohort_values = df_plot[df_plot['cohort'] == cohort]['pred'].values
        
        if len(cohort_values) > 0 and len(control_values) > 0:
            # Perform t-test
            ttest = ttest_ind(cohort_values, control_values)
            pval = ttest.pvalue
            sig_stars = get_significance_stars(pval)
            
            # Get max y-value from both boxes to position bracket above
            max_y_control = control_values.max() if len(control_values) > 0 else 0
            max_y_cohort = cohort_values.max() if len(cohort_values) > 0 else 0
            max_y = max(max_y_control, max_y_cohort)
            
            # Position bracket slightly above the boxes with vertical spacing to avoid overlap
            bracket_y = max_y + 0.05 + (bracket_offset * 0.015)
            bracket_offset += 1
            
            # Draw bracket between Controls position and current cohort position
            x1 = box_positions[control_idx]  # Controls position
            x2 = box_positions[i]  # Cohort position
            x_center = (x1 + x2) / 2
            
            # Draw horizontal line
            ax.plot([x1, x2], [bracket_y, bracket_y], 'k-', linewidth=1)
            # Draw vertical lines at ends
            ax.plot([x1, x1], [bracket_y - 0.01, bracket_y], 'k-', linewidth=1)
            ax.plot([x2, x2], [bracket_y - 0.01, bracket_y], 'k-', linewidth=1)
            # Add significance stars
            ax.text(x2, bracket_y - 0.05, sig_stars, ha='center', va='bottom', fontsize=10)
    
    # Adjust y-axis limits to accommodate brackets
    ax.set_ylim(0, 1.1)
    
    if save:
        plt.tight_layout()
        plt.savefig('supplemental_figures/cotherapy_analysis_overall_v2.png', dpi=300, bbox_inches='tight')
    return ax


if __name__ == '__main__':
    generate_cotherapy_analysis_figure(save=True)