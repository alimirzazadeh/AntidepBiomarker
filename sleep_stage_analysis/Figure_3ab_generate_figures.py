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

SIMULATED = False
# if SIMULATED:
#     df = pd.read_csv('SIMULATED_figure_draft_v16_rem_latency.csv')
# else:

df = pd.read_csv('../data/figure_draft_v16_rem_latency.csv')
# bp() 
# df = pd.read_csv('../data/master_dataset.csv')
# df = df[['filename', 'rem_latency_gt', 'rem_latency_pred', 'sws_duration_gt', 'sws_duration_pred', 'rem_duration_gt', 'rem_duration_pred', 'sleep_efficiency_gt', 'sleep_efficiency_pred', 'label']]
bp() 
control_df = df[df['label'] == 0]
antidep_df = df[df['label'] == 1]


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
    

