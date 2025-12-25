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
EXP_FOLDER = 'data/'
font_size = 12

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
    if pval < .001: #1e-10:
        return '***'
    elif pval < 0.01: #0.001:
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
    # Filter to only include MROS and WSC datasets
    df = df[df['dataset'].isin(['mros', 'wsc','rf'])].copy() 
    # Apply sigmoid transformation to predictions
    df['pred'] = 1 / (1 + np.exp(-df['pred'])) 
    
    # Extract filename and clean patient IDs
    df['filename'] = df['filepath'].apply(lambda x: x.split('/')[-1])
    df['pid'] = df.apply(lambda x: x['pid'] if x['dataset'] != 'wsc' else x['pid'][1:], axis=1)
    df['pid'] = df.apply(lambda x: x['pid'] if x['dataset'] != 'mros' else x['pid'][1:], axis=1)
    
    # Aggregate by filename (take mean for numeric, first value for non-numeric)
    df = df.groupby('filename').agg(lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0])
    

    return df

def process_mit_medications(EXP_FOLDER = 'data/'):
    """Add taxonomy information to the dataframe."""
    df_taxonomy = pd.read_csv(os.path.join(EXP_FOLDER,'master_dataset.csv'))
    df_taxonomy = df_taxonomy[df_taxonomy['dataset'] == 'rf'].copy()
    df_taxonomy = df_taxonomy[['filename', 'taxonomy','date','pid']]

    df_taxonomy['date'] = pd.to_datetime(df_taxonomy['date'])
    df_rf = pd.read_csv(os.path.join(EXP_FOLDER, 'mit_psychotropics.csv'))
    df_rf.rename(columns={'Unnamed: 0': 'pid'}, inplace=True)
    def get_benzos(pid, date):
        if pid in ['NIHYA889LELYV','NIHFW795KLATW']:
            return True if (pid == 'NIHYA889LELYV' and date < pd.to_datetime('2019-09-15')) or (pid == 'NIHFW795KLATW' and date > pd.to_datetime('2021-01-01')) else False
        else:
            if  pid not in df_rf['pid'].values:
                return False
            else:
                return df_rf[df_rf['pid'] == pid]['Benzodiazepines'].item()
    def get_antipsycho(pid, date):
        # print(type(pid), type(df_rf['pid'].values))
        if  pid not in df_rf['pid'].values:
                return False
        else:
                return df_rf[df_rf['pid'] == pid]['Antipsychotics'].item()
    def get_anticonvuls(pid, date):
        if pid in ['NIHKH638RXUVN']:
            return True if (pid == 'NIHKH638RXUVN' and date < pd.to_datetime('2021-03-20')) and (date > pd.to_datetime('2021-02-20')) else False
        else:
            if  pid not in df_rf['pid'].values:
                return False
            else:
                return df_rf[df_rf['pid'] == pid]['Anticonvulsants'].item()
    def get_hypnotics(pid, date):
            if  pid not in df_rf['pid'].values:
                return False
            else:
                return df_rf[df_rf['pid'] == pid]['Hypnotics'].item()
    def get_anticholinergics(pid, date):
        if pid in ['NIHPX213JXJZC','NIHYA889LELYV']:
            return True if (pid == 'NIHPX213JXJZC' and date > pd.to_datetime('2020-07-14')) or (pid == 'NIHYA889LELYV' and date < pd.to_datetime('2020-10-01')) else False
        else:
            if  pid not in df_rf['pid'].values:
                return False
            else:
                return df_rf[df_rf['pid'] == pid]['Anticholinergics'].item()
    def get_stimulants(pid, date):
        if  pid not in df_rf['pid'].values:
                return False
        else:
                return df_rf[df_rf['pid'] == pid]['Stimulants'].item()
    df_taxonomy['pid'] = df_taxonomy['pid'].astype(str)
    df_taxonomy['benzos'] = df_taxonomy.apply(lambda x: get_benzos(x['pid'],x['date']), axis=1)
    df_taxonomy['antipsycho'] = df_taxonomy.apply(lambda x: get_antipsycho(x['pid'],x['date']), axis=1)
    df_taxonomy['convuls'] = df_taxonomy.apply(lambda x: get_anticonvuls(x['pid'],x['date']), axis=1)
    df_taxonomy['hypnotics'] = df_taxonomy.apply(lambda x: get_hypnotics(x['pid'],x['date']), axis=1)
    df_taxonomy['anticholinergics'] = df_taxonomy.apply(lambda x: get_anticholinergics(x['pid'],x['date']), axis=1)
    df_taxonomy['stimulants'] = df_taxonomy.apply(lambda x: get_stimulants(x['pid'],x['date']), axis=1)
    df_taxonomy.fillna(False, inplace=True)
    print('Stimulants: ', df_taxonomy[df_taxonomy['stimulants'] == True]['pid'].unique())
    print('Anticholinergics: ', df_taxonomy[df_taxonomy['anticholinergics'] == True]['pid'].unique())
    print('Hypnotics: ', df_taxonomy[df_taxonomy['hypnotics'] == True]['pid'].unique())
    print('Convuls: ', df_taxonomy[df_taxonomy['convuls'] == True]['pid'].unique())
    print('Antipsycho: ', df_taxonomy[df_taxonomy['antipsycho'] == True]['pid'].unique())
    print('Benzos: ', df_taxonomy[df_taxonomy['benzos'] == True]['pid'].unique())
    return df_taxonomy[['filename', 'benzos', 'antipsycho', 'convuls', 'hypnotics', 'anticholinergics', 'stimulants']]
    

def add_taxonomy_data(df):
    """Add taxonomy information to the dataframe."""
    df_taxonomy = pd.read_csv(os.path.join(EXP_FOLDER,'antidep_taxonomy_all_datasets_v6.csv'))
    df_taxonomy = df_taxonomy[['filename', 'taxonomy']]
    df = pd.merge(df, df_taxonomy, on='filename', how='inner')
    return df

def process_mros_medications(EXP_FOLDER = 'data/'):
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
        if True: #category != 'stimulants':  # Skip stimulants as they're not used in the final analysis
            df_mros_meds[category.replace('convuls', 'convuls')] = df_mros_meds.apply(
                lambda x: any([x[med] == True for med in medications if med in df_mros_meds.columns]), 
                axis=1
            )
    
    return df_mros_meds[['filename', 'benzos', 'antipsycho', 'convuls', 'hypnotics', 'anticholinergics','stimulants']]

def process_wsc_medications(EXP_FOLDER = 'data/'):
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
        # if category != 'stimulants':  # Skip stimulants as they're not used in the final analysis
        df_wsc_meds[category.replace('convuls', 'convuls')] = df_wsc_meds.apply(
            lambda x: any([x[med] == True for med in medications if med in df_wsc_meds.columns]), 
            axis=1
        )
    
    return df_wsc_meds[['filename', 'benzos', 'antipsycho', 'convuls', 'hypnotics', 'anticholinergics', 'stimulants']]

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
    # groups['antidep'] = df[
    #     (df['label'] == 1) & 
    #     (df['benzos'] == False) & 
    #     (df['antipsycho'] == False) & 
    #     (df['convuls'] == False) & 
    #     (df['hypnotics'] == False) & 
    #     (df['anticholinergics'] == False)
    # ]['pred'].values
    # All Antidepressant users 
    groups['antidep'] = df[(df['label'] == 1)]['pred'].values
    
    return groups

def create_visualization(groups, save_path=None, ax=None, save=True):
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
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    
    # Define order for consistent presentation
    order = ['No Psycho\ntropic', 'Anti-\ncholinergics', 'Hypnotics', 'Anti-\nconvulsants', 'Benzo-\ndiazepines',
             'Anti-\npsychotics', 'Anti-\ndepressants']
    
    # Create boxplot
    sns.boxplot(
        x="medication", y="pred", data=df_viz, 
        palette='Greens', ax=ax, order=order, showfliers=False
    )
    ax.tick_params(axis='x', labelsize=font_size-1)
    ax.tick_params(axis='y', labelsize=font_size)
    
    

    ### Add significances - compare each medication to Antidepressants
    # Get Antidepressants values (last column)
    antidep_values = df_viz[df_viz['medication'] == 'Anti-\ndepressants']['pred'].values
    box_positions = ax.get_xticks()
    
    # Find the index of Antidepressants in the order
    antidep_idx = order.index('Anti-\ndepressants')
    
    # Compare each medication (except Antidepressants) to Antidepressants
    bracket_offset = 0  # Track vertical offset for each bracket to avoid overlap
    for i, medication in enumerate(order):
        if medication == 'Anti-\ndepressants':
            continue  # Skip Antidepressants itself
        
        medication_values = df_viz[df_viz['medication'] == medication]['pred'].values
        
        if len(medication_values) > 0 and len(antidep_values) > 0:
            # Calculate p-value and effect size
            pval, effect_size = calculate_pval_effect_size(medication_values, antidep_values)
            sig_stars = get_significance_stars(pval)
            crosses = get_effect_size_crosses(effect_size)
            
            # Get max y-value from both boxes to position bracket above
            max_y_antidep = antidep_values.max() if len(antidep_values) > 0 else 0
            max_y_medication = medication_values.max() if len(medication_values) > 0 else 0
            max_y = max(max_y_antidep, max_y_medication) + 0.08
            
            # Position bracket slightly above the boxes with vertical spacing to avoid overlap
            bracket_y = max_y + 0.05 + (bracket_offset * 0.02)  # 0.02 spacing between brackets
            bracket_offset -= 1
            
            # Draw bracket between Antidepressants position and current medication position
            x1 = box_positions[antidep_idx]  # Antidepressants position
            x2 = box_positions[i]  # Medication position
            x_center = (x1 + x2) / 2
            
            # Draw horizontal line
            ax.plot([x1, x2], [bracket_y, bracket_y], 'k-', linewidth=1)
            # Draw vertical lines at ends
            ax.plot([x1, x1], [bracket_y - 0.01, bracket_y], 'k-', linewidth=1)
            ax.plot([x2, x2], [bracket_y - 0.01, bracket_y], 'k-', linewidth=1)
            # Add significance stars
            ax.text(x2, bracket_y - 0.01 - 0.06, sig_stars, ha='center', va='bottom', fontsize=font_size)
            # Add effect size crosses right above the stars
            # ax.text(x2, bracket_y - 0.01 - 0.05 -  0.06, crosses, ha='center', va='bottom', fontsize=font_size)
    
    
    
    
    # Add sample size annotations
    for i, medication in enumerate(order):
        n = df_viz[df_viz['medication'] == medication].shape[0]
        # ypos = np.percentile(df_viz[df_viz['medication'] == medication]['pred'], 88) + 0.03
        ypos = df_viz[df_viz['medication'] == medication]['pred'].median() + 0.01
        ax.text(i, ypos, f'N={n}', horizontalalignment='center', 
                size=font_size, color='black')
    
    # Customize plot
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel('Model Score', fontsize=font_size)
    ax.set_xlabel('Medication Type', fontsize=font_size)
    if save:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return ax

def generate_other_medications_figure(save=True, ax=None):
    """Main analysis pipeline."""
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    print("Adding taxonomy data...")
    df = add_taxonomy_data(df)
    
    print("Processing medication data...")
    df_mros_meds = process_mros_medications()
    df_wsc_meds = process_wsc_medications()
    
    print("Processing MIT medication data...")
    df_mit_meds = process_mit_medications()
    
    # Combine medication data
    df_other = pd.concat([df_mros_meds, df_wsc_meds, df_mit_meds])

    df = df.merge(df_other, on='filename', how='inner')
    df = df[['filename', 'taxonomy', 'pred', 'pid', 'label', 'benzos', 'antipsycho', 'convuls', 'hypnotics', 'anticholinergics', 'stimulants','dataset']].copy()
    
    df = df.groupby(['pid',  'label','benzos', 'antipsycho', 'convuls', 'hypnotics', 'anticholinergics', 'stimulants','dataset']).agg({'pred': 'mean'}).reset_index()
    print(df[df['dataset'] == 'mros'].shape[0])
    print(df[df['dataset'] == 'wsc'].shape[0])
    print(df[df['dataset'] == 'rf'].shape[0])
    print('Total merged data: ', df.shape[0])
    
    print("Extracting medication groups...")
    groups = extract_medication_groups(df)
    
    print("Creating visualization...")
    save_path = 'biomarker/analysis/figure_4a.png'
    if save:
        create_visualization(groups, save_path=save_path, save=True)
    else:
        create_visualization(groups, save_path=None, ax=ax, save=False)

    print("Analysis completed successfully!")

if __name__ == "__main__":
    generate_other_medications_figure(save=True)