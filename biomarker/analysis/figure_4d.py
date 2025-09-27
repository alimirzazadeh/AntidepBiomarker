"""
Analysis of Model Predictions Across Different Medication Types
==============================================================

This script analyzes model predictions for various medication types including
benzodiazepines, antipsychotics, anticonvulsants, hypnotics, and antidepressants
using data from MROS and WSC datasets.

Author: [Author Name]
Date: [Date]
"""

"""
Anticholinergics: 
- Antihistamines (first-generation, strong anticholinergic effects):
Diphenhydramine — Benadryl
Chlorpheniramine — Chlor-Trimeton
Hydroxyzine — Vistaril, Atarax
Clemastine — Tavist
Meclizine — Antivert, Bonine
Promethazine — Phenergan

- Non-antihistamines:
Benztropine — Cogentin
Trihexyphenidyl — (no widely known U.S. brand; formerly Artane)
Oxybutynin — Ditropan
Tolterodine — Detrol
Solifenacin — Vesicare
Dicyclomine — Bentyl
Hyoscyamine — Levsin, Anaspaz
Scopolamine — Transderm Scop

"""

import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.__config__ import show
from ipdb import set_trace as bp
import scipy 
# Configuration
EXP_FOLDER = '../../data/'


def calculate_pval_effect_size(x,y,equal_var=False):
    t,p=scipy.stats.ttest_ind(x,y,equal_var=equal_var)
    nx,ny=len(x),len(y)
    md=np.mean(x)-np.mean(y)
    if equal_var:
        sp=np.sqrt(((nx-1)*np.var(x,ddof=1)+(ny-1)*np.var(y,ddof=1))/(nx+ny-2))
    else:
        sp=np.sqrt((np.var(x,ddof=1)+np.var(y,ddof=1))/2)
    return p,md/sp
def get_significance_stars(pval):
    if pval < 1e-10:
        return '***'
    elif pval < 0.001:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'ns'

# Function to get effect size crosses
def get_effect_size_crosses(effect_size):
    abs_es = abs(effect_size)
    if abs_es > 0.5:
        return '†††'
    elif abs_es > 0.3:
        return '††'
    elif abs_es > 0.1:
        return '†'
    else:
        return 'ne'

def load_and_preprocess_data():
    """Load and preprocess the main inference data."""
    # Load main inference data
    df = pd.read_csv(os.path.join(EXP_FOLDER, 'inference_v6emb_3920_all.csv'))
    
    # Apply sigmoid transformation to predictions
    df['pred'] = 1 / (1 + np.exp(-df['pred'])) 
    
    # Extract filename and clean patient IDs
    df['filename'] = df['filepath'].apply(lambda x: x.split('/')[-1])
    df['pid'] = df.apply(lambda x: x['pid'] if x['dataset'] != 'wsc' else x['pid'][1:], axis=1)
    df['pid'] = df.apply(lambda x: x['pid'] if x['dataset'] != 'mros' else x['pid'][1:], axis=1)
    
    # Aggregate by filename (take mean for numeric, first value for non-numeric)
    df = df.groupby('filename').agg(lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0])
    
    # Filter to only include MROS and WSC datasets
    df = df[~((df['dataset'] != 'mros') * (df['dataset'] != 'wsc'))].copy()
    
    return df

def add_taxonomy_data(df):
    """Add taxonomy information to the dataframe."""
    df_taxonomy = pd.read_csv(os.path.join(EXP_FOLDER,'antidep_taxonomy_all_datasets_v6.csv'))
    df_taxonomy = df_taxonomy[['filename', 'taxonomy']]
    df = pd.merge(df, df_taxonomy, on='filename', how='inner')
    return df

def process_mros_medications(EXP_FOLDER = '../../data/'):
    """Process MROS medication data and create medication flags."""
    # Load MROS medication data
    df_mros_meds = pd.read_csv(os.path.join(EXP_FOLDER,'mros2-dataset-augmented-live.csv'))
    df_mros1_meds = pd.read_csv(os.path.join(EXP_FOLDER,'mros1-dataset-augmented-live.csv'))
    df_mros_meds = pd.concat([df_mros_meds, df_mros1_meds])
    
    # Define medication categories with their corresponding variable names
    medication_categories = {
        'benzos': ['M1ALPRAZ', 'M1DIAZEP', 'M1LORAZE', 'M1CLONAZ', 'M1TEMAZE', 'M1MIDAZO'],
        'antipsycho': ['M1HALOPE', 'M1CHLORM', 'M1RISPER', 'M1OLANZA', 'M1QUETIA', 'M1ARIPIP', 'M1CLOZAP'],
        'convuls': ['M1CARBAZ', 'M1LAMOTR', 'M1GABAPE', 'M1PREGAB', 'M1TOPIRA', 'M1LEVETI'],
        'hypnotics': ['M1ZOLPID', 'M1ESZOPI', 'M1ZALEPL', 'M1RAMELT'],
        'stimulants': ['M1METHPH', 'M1AMPHET', 'M1MODAFI'],
        'anticholinergics': ['M1DIPHHY','M1CHLORR','M1HYDROY','M1CLEMAS','M1MECLIZ','M1PROMET','M1BENZTR','M1TRIHEX','M1OXYBUT','M1TOLTER','M1SOLIFE','M1DICYCL','M1HYOSCY','M1SCOPOL',]
    }
    
    # Create binary flags for each medication category
    for category, medications in medication_categories.items():
        if category != 'stimulants':  # Skip stimulants as they're not used in the final analysis
            df_mros_meds[category.replace('convuls', 'convuls')] = df_mros_meds.apply(
                lambda x: any([x[med] == True for med in medications if med in df_mros_meds.columns]), 
                axis=1
            )
    
    return df_mros_meds[['filename', 'benzos', 'antipsycho', 'convuls', 'hypnotics', 'anticholinergics']]

def process_wsc_medications(EXP_FOLDER = '../../data/'):
    """Process WSC medication data and create medication flags."""
    df_wsc_meds = pd.read_csv(os.path.join(EXP_FOLDER,'wsc-dataset-0.7.0.csv'))
    
    # Define medication categories with their corresponding WSC variable names
    medication_categories = {
        'benzos': ['dr412', 'dr420', 'dr421', 'dr431', 'dr863', 'dr440'],
        'antipsycho': ['dr423', 'dr429', 'dr430'],
        'convuls': ['dr418', 'dr404', 'dr443', 'dr433'],
        'hypnotics': ['dr858'],
        'stimulants': ['dr706', 'dr776', 'dr861', 'dr842', 'dr859'],
        'anticholinergics': ['dr219','dr221','dr235','dr239','dr207','dr229','dr783','dr854','dr759','dr717','dr256']
    }
    
    # Create filename for WSC data
    df_wsc_meds['filename'] = df_wsc_meds.apply(
        lambda x: f'wsc-visit{x["wsc_vst"]}-{x["wsc_id"]}-nsrr.npz', 
        axis=1
    )
    
    # Create binary flags for each medication category
    for category, medications in medication_categories.items():
        if category != 'stimulants':  # Skip stimulants as they're not used in the final analysis
            df_wsc_meds[category.replace('convuls', 'convuls')] = df_wsc_meds.apply(
                lambda x: any([x[med] == True for med in medications if med in df_wsc_meds.columns]), 
                axis=1
            )
    
    return df_wsc_meds[['filename', 'benzos', 'antipsycho', 'convuls', 'hypnotics', 'anticholinergics']]

def extract_medication_groups(df):
    """Extract prediction values for different medication groups."""
    # Define medication groups
    groups = {}
    
    # Controls: no antidepressants and no other medications
    groups['controls'] = df[
        (df['label'] == 0) & 
        (df['benzos'] == False) & 
        (df['antipsycho'] == False) & 
        (df['convuls'] == False) & 
        (df['hypnotics'] == False) & 
        (df['anticholinergics'] == False)
    ]['pred'].values
    
    # Specific medication groups (non-antidepressant users)
    groups['benzos'] = df[(df['label'] == 0) & (df['benzos'] == True)]['pred'].values
    groups['antipsycho'] = df[(df['label'] == 0) & (df['antipsycho'] == True)]['pred'].values
    groups['convuls'] = df[(df['label'] == 0) & (df['convuls'] == True)]['pred'].values
    groups['hypnotics'] = df[(df['label'] == 0) & (df['hypnotics'] == True)]['pred'].values
    groups['anticholinergics'] = df[(df['label'] == 0) & (df['anticholinergics'] == True)]['pred'].values
    # Antidepressant users without other medications
    groups['antidep'] = df[
        (df['label'] == 1) & 
        (df['benzos'] == False) & 
        (df['antipsycho'] == False) & 
        (df['convuls'] == False) & 
        (df['hypnotics'] == False) & 
        (df['anticholinergics'] == False)
    ]['pred'].values
    
    return groups

def create_visualization(groups, save_path):
    """Create and save the medication comparison visualization."""
    # Print group sizes
    print('Group sizes:')
    for group_name, values in groups.items():
        print(f'  {group_name.capitalize()}: {len(values)}')
    
    # Prepare data for visualization
    df_viz = pd.DataFrame({
        'medication': (
            ['No Psycho\ntropic'] * len(groups['controls']) +
            ['Benzo-\ndiazepines'] * len(groups['benzos']) +
            ['Anti-\npsychotics'] * len(groups['antipsycho']) +
            ['Anti-\nconvulsants'] * len(groups['convuls']) +
            ['Hypnotics'] * len(groups['hypnotics']) +
            ['Anti-\ncholinergics'] * len(groups['anticholinergics']) +
            ['Anti-\ndepressants'] * len(groups['antidep'])
        ),
        'pred': np.concatenate([
            groups['controls'], groups['benzos'], groups['antipsycho'], 
            groups['convuls'], groups['hypnotics'], groups['anticholinergics'], groups['antidep']
        ])
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define order for consistent presentation
    order = ['No Psycho\ntropic', 'Anti-\ncholinergics', 'Hypnotics', 'Anti-\nconvulsants', 'Benzo-\ndiazepines',
             'Anti-\npsychotics', 'Anti-\ndepressants']
    
    # Create boxplot
    sns.boxplot(
        x="medication", y="pred", data=df_viz, 
        palette='Greens', ax=ax, order=order, showfliers=False
    )
    
    

    ### Add significances 
    for i, medication in enumerate(order[:-1]):
        medication_values = df_viz[df_viz['medication'] == medication]['pred'].values
        antidep_values = df_viz[df_viz['medication'] == 'Anti-\ndepressants']['pred'].values
        pval_pred, effect_size_pred = calculate_pval_effect_size(antidep_values, medication_values)

        
        # Get current y limits for positioning
        y_min, y_max = ax.get_ylim()
        y_pos = y_max - (y_max - y_min) * 0.15  # Position above the boxes
        
        # Add annotations for GT comparison (between positions 0 and 1)
        stars_gt = get_significance_stars(pval_pred)
        crosses_gt = get_effect_size_crosses(effect_size_pred)
        annotation_gt = stars_gt + '\n' + crosses_gt if stars_gt or crosses_gt else ''
        if annotation_gt:
            # Position between GT Control (0) and GT Antidep (1)
            ax.text(i, y_pos, annotation_gt, 
                      ha='center', va='bottom', fontsize=10, color='black')
    
    
    
    
    
    
    # Add sample size annotations
    for i, medication in enumerate(order):
        n = df_viz[df_viz['medication'] == medication].shape[0]
        ypos = np.percentile(df_viz[df_viz['medication'] == medication]['pred'], 88) + 0.03
        ax.text(i, ypos, f'N={n}', horizontalalignment='center', 
                size='medium', color='black')
    
    # Customize plot
    ax.set_ylim(0, 1)
    ax.set_ylabel('Model Prediction')
    ax.set_xlabel('Medication Type')
    plt.tight_layout()
    
    # Save figure
    fig.savefig(save_path, dpi=300, bbox_inches='tight')

def main():
    """Main analysis pipeline."""
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    print("Adding taxonomy data...")
    df = add_taxonomy_data(df)
    
    print("Processing medication data...")
    df_mros_meds = process_mros_medications()
    df_wsc_meds = process_wsc_medications()
    
    # Combine medication data
    df_other = pd.concat([df_mros_meds, df_wsc_meds])
    df = df.merge(df_other, on='filename', how='inner')
    
    print("Extracting medication groups...")
    groups = extract_medication_groups(df)
    
    print("Creating visualization...")
    save_path = 'figure_4d.png'
    create_visualization(groups, save_path)
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()