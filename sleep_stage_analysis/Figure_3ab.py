import numpy as np 
import pandas as pd 
import os 
from ipdb import set_trace as bp 
import copy 
from tqdm import tqdm 
import seaborn as sns
from scipy.stats import pearsonr
from scipy import stats
import dataframe_image as dfi

def calculate_rem_latency(sleep_stage):
    if 4 not in sleep_stage:
        return np.nan 
    return np.argwhere(sleep_stage == 4)[0][0] - np.argwhere(sleep_stage > 0)[0][0]

def calculate_sws_duration(sleep_stage):
    return np.sum(sleep_stage == 3) 

def calculate_rem_duration(sleep_stage):
    return np.sum(sleep_stage == 4) 

## this is sleep efficiency 
def calculate_sleep_continuity(sleep_stage):
    return np.sum(sleep_stage > 0) / (np.sum(sleep_stage > 0) + np.sum(sleep_stage == 0))


def process_stages(stages):
    stages = stages['data'][::int(30*stages['fs'])]
    stages[stages < 0] = 0
    stages[stages > 5] = 0
    stages = stages.astype(int)
    # mapping = np.array([0, 2, 2, 3, 3, 1, 0, 0, 0, 0], int)
    mapping = np.array([0, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0], np.int64)
    return mapping[stages]

df = pd.read_csv('../data/antidep_taxonomy_all_datasets_v6.csv')
df_loc = pd.read_csv('../data/inference_v6emb_3920_all.csv')
df_loc['filename'] = df_loc['filename'].apply(lambda x: x.split('/')[-1])
control_df = copy.deepcopy(df[df['taxonomy'] == 'C'])
antidep_df = copy.deepcopy(df[df['taxonomy'] != 'C'])

EEG_STAGE_DIR = "/Users/alimirz/mnt_all/data/netmit/wifall/ADetect/data"
STAGE_PREDICTION_DIR = "/Users/alimirz/mnt_all/data/netmit/wifall/chaoli/best_stage_predictions"
for idx, row in tqdm(control_df.iterrows()):
    # if idx % 10 != 0:
    #     continue 
    try:
        dataset = df_loc[df_loc['filename'] == row['filename']]['dataset'].values[0]
        if 'mros-visit1' in row['filename']:
            dataset = 'mros1_new'
        elif 'mros-visit2' in row['filename']:
            dataset = 'mros2_new'
        elif 'shhs1' in row['filename']:
            dataset = 'shhs1_new'
        elif 'shhs2' in row['filename']:
            dataset = 'shhs2_new'
        elif 'wsc' in row['filename']:
            dataset = 'wsc_new'
    except:
        pass
    if dataset in ['hchs', 'rf']:
        continue 
    filepath = f"{EEG_STAGE_DIR}/{dataset}/stage/{row['filename']}"
    try:
        sleep_stage = np.load(filepath)
    except:
        print('skipping', row['filename'])
        continue 
    sleep_stage = process_stages(sleep_stage)
    if np.sum(sleep_stage > 0) < 4 * 60 * 2:
        print('2skipping', row['filename'])
        continue 
    control_df.loc[idx, 'rem_latency_gt'] = calculate_rem_latency(sleep_stage)
    control_df.loc[idx, 'sws_duration_gt'] = calculate_sws_duration(sleep_stage)
    control_df.loc[idx, 'rem_duration_gt'] = calculate_rem_duration(sleep_stage)
    control_df.loc[idx, 'sleep_efficiency_gt'] = calculate_sleep_continuity(sleep_stage)
    
    pred_filepath = f"{STAGE_PREDICTION_DIR}/{dataset}/thorax/{row['filename']}"
    try:
        pred_sleep_stage = np.load(pred_filepath)['data'].flatten()
    except:
        print('3skipping', row['filename'])
        continue 
    control_df.loc[idx, 'dataset'] = dataset 
    control_df.loc[idx, 'rem_latency_pred'] = calculate_rem_latency(pred_sleep_stage)
    control_df.loc[idx, 'sws_duration_pred'] = calculate_sws_duration(pred_sleep_stage)
    control_df.loc[idx, 'rem_duration_pred'] = calculate_rem_duration(pred_sleep_stage)
    control_df.loc[idx, 'sleep_efficiency_pred'] = calculate_sleep_continuity(pred_sleep_stage)

for idx, row in antidep_df.iterrows():
    # if idx % 10 != 0:
    #     continue 
    try:
        dataset = df_loc[df_loc['filename'] == row['filename']]['dataset'].values[0]
        if 'mros-visit1' in row['filename']:
            dataset = 'mros1_new'
        elif 'mros-visit2' in row['filename']:
            dataset = 'mros2_new'
        elif 'shhs1' in row['filename']:
            dataset = 'shhs1_new'
        elif 'shhs2' in row['filename']:
            dataset = 'shhs2_new'
        elif 'wsc' in row['filename']:
            dataset = 'wsc_new'
    except:
        pass
    if dataset in ['hchs', 'rf']:
        continue 

    filepath = f"{EEG_STAGE_DIR}/{dataset}/stage/{row['filename']}"
    try:
        sleep_stage = np.load(filepath)
    except:
        print('skipping', row['filename'])
        continue 
    sleep_stage = process_stages(sleep_stage)
    if np.sum(sleep_stage > 0) < 4 * 60 * 2:
        print('skipping', row['filename'])
        continue 
    antidep_df.loc[idx, 'rem_latency_gt'] = calculate_rem_latency(sleep_stage)
    antidep_df.loc[idx, 'sws_duration_gt'] = calculate_sws_duration(sleep_stage)
    antidep_df.loc[idx, 'rem_duration_gt'] = calculate_rem_duration(sleep_stage)
    antidep_df.loc[idx, 'sleep_efficiency_gt'] = calculate_sleep_continuity(sleep_stage)
    
    pred_filepath = f"{STAGE_PREDICTION_DIR}/{dataset}/thorax/{row['filename']}"
    try:
        pred_sleep_stage = np.load(pred_filepath)['data'].flatten()
    except:
        print('skipping', row['filename'])
        continue 
    antidep_df.loc[idx, 'dataset'] = dataset 
    antidep_df.loc[idx, 'rem_latency_pred'] = calculate_rem_latency(pred_sleep_stage)
    antidep_df.loc[idx, 'sws_duration_pred'] = calculate_sws_duration(pred_sleep_stage)
    antidep_df.loc[idx, 'rem_duration_pred'] = calculate_rem_duration(pred_sleep_stage)
    antidep_df.loc[idx, 'sleep_efficiency_pred'] = calculate_sleep_continuity(pred_sleep_stage)


def calculate_pval_effect_size(x,y,equal_var=False):
    t,p=stats.ttest_ind(x,y,equal_var=equal_var)
    nx,ny=len(x),len(y)
    md=np.mean(x)-np.mean(y)
    if equal_var:
        sp=np.sqrt(((nx-1)*np.var(x,ddof=1)+(ny-1)*np.var(y,ddof=1))/(nx+ny-2))
    else:
        sp=np.sqrt((np.var(x,ddof=1)+np.var(y,ddof=1))/2)
    return p,md/sp

import matplotlib.pyplot as plt 
if True:
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    for i, name in enumerate(['rem_latency', 'sws_duration', 'rem_duration', 'sleep_efficiency']):
        x1 = control_df[name + '_gt'] / 2
        x2 = control_df[name + '_pred'] / 2
        y1 = antidep_df[name + '_gt'] / 2
        y2 = antidep_df[name + '_pred'] / 2
        if name == 'sleep_efficiency':
            x1 = 2 * x1 
            y1 = 2 * y1 
            x2 = 2 * x2 
            y2 = 2 * y2 
        maskx = ~np.isnan(x1) & ~np.isnan(x2) 
        masky = ~np.isnan(y1) & ~np.isnan(y2) 
        x1 = x1[maskx]
        x2 = x2[maskx]
        y1 = y1[masky]
        y2 = y2[masky]
        
        # Calculate p-values and effect sizes for annotations
        pval_gt, effect_size_gt = calculate_pval_effect_size(x1, y1)
        pval_pred, effect_size_pred = calculate_pval_effect_size(x2, y2)
        print(name, 'GT', pval_gt, effect_size_gt)
        print(name, 'Pred', pval_pred, effect_size_pred)
        
        # Create data for seaborn boxplot
        data = []
        labels = []
        data.extend(x1)
        labels.extend(['GT\nControl'] * len(x1))
        data.extend(y1)
        labels.extend(['GT\nAntidep'] * len(y1))
        data.extend(x2)
        labels.extend(['Pred\nControl'] * len(x2))
        data.extend(y2)
        labels.extend(['Pred\nAntidep'] * len(y2))
        
        # Create DataFrame for seaborn
        plot_df = pd.DataFrame({'value': data, 'group': labels})
        
        # Create seaborn boxplot
        sns.boxplot(data=plot_df, x='group', y='value', ax=ax[i], showfliers=False, 
                   palette=['royalblue', 'coral', 'royalblue', 'coral'])
        
        # Add text annotations for significance and effect size
        # Function to get significance stars
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
        
        # Get current y limits for positioning
        y_min, y_max = ax[i].get_ylim()
        y_pos = y_max - (y_max - y_min) * 0.05  # Position above the boxes
        
        # Add annotations for GT comparison (between positions 0 and 1)
        stars_gt = get_significance_stars(pval_gt)
        crosses_gt = get_effect_size_crosses(effect_size_gt)
        annotation_gt = stars_gt + '\n' + crosses_gt if stars_gt or crosses_gt else ''
        
        if annotation_gt:
            # Position between GT Control (0) and GT Antidep (1)
            ax[i].text(0.5, y_pos, annotation_gt, 
                      ha='center', va='bottom', fontsize=10, color='black')
        
        # Add annotations for Pred comparison (between positions 2 and 3)
        stars_pred = get_significance_stars(pval_pred)
        crosses_pred = get_effect_size_crosses(effect_size_pred)
        annotation_pred = stars_pred + '\n' + crosses_pred if stars_pred or crosses_pred else ''
        
        if annotation_pred:
            # Position between Pred Control (2) and Pred Antidep (3)
            ax[i].text(2.5, y_pos, annotation_pred, 
                      ha='center', va='bottom', fontsize=10, color='black')
        
        # Adjust y limits to accommodate annotations
        ax[i].set_ylim(y_min, y_max + (y_max - y_min) * 0.15)
        
        ax[i].set_title(name.replace('_', ' ').title())
        ax[i].set_ylabel('')
        ax[i].set_xlabel('')
    ax[0].set_ylabel('Minutes')
    plt.tight_layout()
    plt.savefig('Figure_3a.png', dpi=300)

## now get the correlation between the ground truth and predicted features 

if True: 
    # Create lists to store results
    features = []
    control_correlations = []
    antidep_correlations = []
    control_pvalues = []
    antidep_pvalues = []
    for name in ['rem_latency', 'sws_duration', 'rem_duration', 'sleep_efficiency']:
        x1 = control_df[name + '_gt']
        x2 = control_df[name + '_pred']
        maskx = ~np.isnan(x1) & ~np.isnan(x2) 
        x1 = x1[maskx]
        x2 = x2[maskx]
        y1 = antidep_df[name + '_gt']
        y2 = antidep_df[name + '_pred']
        masky = ~np.isnan(y1) & ~np.isnan(y2) 
        y1 = y1[masky]
        y2 = y2[masky]
        
        # Calculate correlations
        control_corr = pearsonr(x1, x2)[0]
        antidep_corr = pearsonr(y1, y2)[0]
        control_pvalue = pearsonr(x1, x2)[1]
        antidep_pvalue = pearsonr(y1, y2)[1]
        # Store results
        features.append(name.replace('_', ' ').title())
        control_correlations.append(control_corr)
        antidep_correlations.append(antidep_corr)
        control_pvalues.append(control_pvalue)
        antidep_pvalues.append(antidep_pvalue)
    # Create DataFrame for nice table display
    correlation_df = pd.DataFrame({
        'Feature': features,
        'Pearsons r': [f'{corr:.3f}' for corr in control_correlations],
        'p': [f'{p:.2e}' for p in control_pvalues],
        'Pearsons r ': [f'{corr:.3f}' for corr in antidep_correlations], 
        'p ': [f'{p:.2e}' for p in antidep_pvalues],
    })

    # Display the table
    print("\n" + "="*60)
    print("CORRELATION BETWEEN GROUND TRUTH AND PREDICTED FEATURES")
    print("="*60)
    print("="*15 + "Control" + "="*15 + "Antidep" + "="*15)
    print(correlation_df.to_string(index=False))
    print("="*60)
    
    # Style and export the table as PNG
    styled_df = correlation_df.style.background_gradient(cmap="Blues") \
                         .set_table_styles([
                            {'selector': 'th', 'props': [
                                ('font-size', '12pt'),
                                ('text-align', 'center'),
                                ('min-width', '100px')  # ⬅ widen header cells
                            ]},
                            {'selector': 'td', 'props': [
                                ('font-size', '10pt'),
                                ('text-align', 'center'),
                                ('min-width', '100px')  # ⬅ widen body cells
                            ]},
                            {'selector': 'table', 'props': [
                                ('width', '100%')  # ⬅ make table take full width
                            ]}
                         ]) \
                         .hide(axis="index")  # hide index if not needed

    # Save as PNG
    dfi.export(styled_df, "Figure_3b.png")
    

pd.concat([control_df[['filename', 'rem_latency_gt', 'rem_latency_pred']], antidep_df[['filename', 'rem_latency_gt', 'rem_latency_pred']]]).to_csv('figure_draft_v15_rem_latency.csv', index=False)
bp() 

print('done ')