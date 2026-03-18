"""
Longitudinal Analysis of Antidepressant Loading and Tapering Effects
===================================================================

This script analyzes individual patient trajectories during antidepressant
medication changes, including stopping sertraline, starting venlafaxine,
and fluoxetine initiation patterns.

"""

import pandas as pd 
import numpy as np 
import os
import datetime 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
from ipdb import set_trace as bp 
from tqdm import tqdm
import matplotlib.dates as mdates

# Configuration
EXP_FOLDER = '../../data/'
FILTERED = True
FONT_SIZE = 14
QUANTILE = 0.8


def sigmoid(x):
    """Apply sigmoid transformation to convert logits to probabilities."""
    return 1 / (1 + np.exp(-x))

def load_and_preprocess_data():
    """Load inference data and apply preprocessing."""
    df = pd.read_csv(os.path.join(EXP_FOLDER, 'inference_v6emb_3920_all.csv'))
    if FILTERED == True:
        bad_sig = pd.read_csv('../../data/bad_signal_time_4_patients.csv')
        bad_sig['filename'] = bad_sig['filename'].apply(lambda x: x.split('/')[-1])
        print('df shape before merge', df.shape)
        df['filename'] = df['filename'].apply(lambda x: x.split('/')[-1])
        df = pd.merge(df, bad_sig, on='filename', how='inner')
        print('df shape after merge', df.shape)
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Apply sigmoid transformation to predictions
    df['pred'] = df['pred'].apply(sigmoid)
    
    return df

def extract_patient_cohorts(df, key=None, FILTERED_THRESHOLD=None):
    """
    Extract specific patient cohorts for longitudinal analysis.
    
    Returns:
        list: List of DataFrames for each patient cohort
        list: List of corresponding titles for each cohort
    """
    # Patient 1033: Stopping Sertraline


    
    # '0h_3h_TRIM_left', '0h_3h_TRIM_right', '0h_3h_TRIM_both', '0h_3h_TRIM_none', '0.25h_3h_TRIM_left', '0.25h_3h_TRIM_right', '0.25h_3h_TRIM_both', '0.25h_3h_TRIM_none', '0.5h_3h_TRIM_left', '0.5h_3h_TRIM_right', '0.5h_3h_TRIM_both', '0.5h_3h_TRIM_none', '1h_3h_TRIM_left', '1h_3h_TRIM_right', '1h_3h_TRIM_both', '1h_3h_TRIM_none', 
    # '0h_9h_TRIM_both', '0h_9h_TRIM_none', '0.25h_9h_TRIM_left', '0.25h_9h_TRIM_right', '0.25h_9h_TRIM_both', '0.25h_9h_TRIM_none', '0.5h_9h_TRIM_left', '0.5h_9h_TRIM_right', '0.5h_9h_TRIM_both', '0.5h_9h_TRIM_none', '1h_9h_TRIM_left', '1h_9h_TRIM_right', '1h_9h_TRIM_both', '1h_9h_TRIM_none', '0h_9h_TRIM_left', '0h_9h_TRIM_right',
    # '0h_6h_TRIM_left', '0h_6h_TRIM_right', '0h_6h_TRIM_both', '0h_6h_TRIM_none', '0.25h_6h_TRIM_left', '0.25h_6h_TRIM_right', '0.25h_6h_TRIM_both', '0.25h_6h_TRIM_none', '0.5h_6h_TRIM_left', '0.5h_6h_TRIM_right', '0.5h_6h_TRIM_both', '0.5h_6h_TRIM_none', '1h_6h_TRIM_left', '1h_6h_TRIM_right', '1h_6h_TRIM_both', '1h_6h_TRIM_none', 

    # '0h_4h_TRIM_left', '0h_4h_TRIM_right', '0h_4h_TRIM_both', '1h_4h_TRIM_none', '0.25h_4h_TRIM_left', '0.25h_4h_TRIM_right', '0.25h_4h_TRIM_both', '0.5h_4h_TRIM_left', '0.5h_4h_TRIM_right', '0.5h_4h_TRIM_both', '0.5h_4h_TRIM_none', '1h_4h_TRIM_left', '1h_4h_TRIM_right', '1h_4h_TRIM_both', 
    ## find threshold where 15% of the data is bad
    if FILTERED_THRESHOLD is None:
        FILTERED_THRESHOLD = df[key].quantile(QUANTILE)
    print(f"Key: {key}, FILTERED_THRESHOLD: {FILTERED_THRESHOLD}")
    df_1033 = df[df['pid'] == '1033'].copy()
    df_1033 = df_1033[df_1033['date'] < datetime.datetime(2022, 5, 28)]

    if FILTERED == True:
        df_1033 = df_1033[df_1033[key] < FILTERED_THRESHOLD]
    stable_period_pos_1033 = df_1033[df_1033['date'] < datetime.datetime(2022, 2, 1)]['pred']
    print(f"Stable period positive 1033: {stable_period_pos_1033.std()}, total days: {len(stable_period_pos_1033)}")
    
    # Patient 1007: Starting Venlafaxine (filtered to before Sept 2021)
    df_1007 = df[df['pid'] == '1007'].copy()
    df_1007 = df_1007[df_1007['date'] < datetime.datetime(2021, 8, 20)]
    if FILTERED == True:
        df_1007 = df_1007[df_1007[key] < FILTERED_THRESHOLD]
    
    # stable_period_neg_1007 = df_1007[df_1007['date'] <= datetime.datetime(2020, 12, 15)]['pred']
    stable_period_pos_1007 = df_1007[df_1007['date'] > datetime.datetime(2021, 1, 28)]['pred']
    print(f"Stable period positive 1007: {stable_period_pos_1007.std()}, total days: {len(stable_period_pos_1007)}")
    
    # Patient 1022: Starting Fluoxetine with known adherence issues
    df_1022 = df[df['pid'] == '1022'].copy()
    if FILTERED == True:
        df_1022 = df_1022[df_1022[key] < FILTERED_THRESHOLD]
    # stable_period_neg_1022 = df_1022[df_1022['date'] <= datetime.datetime(2020, 12, 15)]['pred']
    stable_period_pos_1022 = df_1022['pred']
    print(f"Stable period positive 1022: {stable_period_pos_1022.std()}, total days: {len(stable_period_pos_1022)}")
    
    # Patient NIHYM875FLXFF: Starting Fluoxetine (filtered time window)
    df_xff = df[df['pid'] == 'NIHYM875FLXFF'].copy()
    df_xff = df_xff[df_xff['date'] >= datetime.datetime(2020, 8, 1)]
    df_xff = df_xff[df_xff['date'] < datetime.datetime(2022, 4, 1)]
    if FILTERED == True:
        df_xff = df_xff[df_xff[key] < FILTERED_THRESHOLD]
    # stable_period_neg_xff = df_xff[df_xff['date'] <= datetime.datetime(2021, 6, 10)]['pred']
    stable_period_pos_xff = df_xff[df_xff['date'] > datetime.datetime(2021, 7, 10)]['pred']
    print(f"Stable period negative xff: {stable_period_pos_xff.std()}, total days: {len(stable_period_pos_xff)}")
    
    # Apply additional filtering to remove edge effects (90 days from start/end)
    min_xff = df_xff['date'].min() + datetime.timedelta(days=90)
    max_xff = df_xff['date'].max() - datetime.timedelta(days=90)
    df_xff = df_xff[(df_xff['date'] > min_xff) & (df_xff['date'] < max_xff)]
    
    # Define cohort information
    cohorts = [df_1033, df_1007, df_1022, df_xff]
    titles = [
        'Stopping Sertraline', 
        'Starting Venlafaxine', 
        'Starting Citalopram, Non-Adherent', 
        'Starting Fluoxetine'
    ]
    
    return cohorts, titles, FILTERED_THRESHOLD

def apply_smoothing_filter(signal, median_kernel_size=7, gaussian_sigma=1):
    """
    Apply median and Gaussian filtering to smooth the signal.
    
    Args:
        signal (array): Input signal to be smoothed
        median_kernel_size (int): Kernel size for median filter
        gaussian_sigma (float): Sigma parameter for Gaussian filter
    
    Returns:
        array: Smoothed signal
    """
    # Apply median filter to remove outliers
    filtered_signal = medfilt(signal, kernel_size=median_kernel_size)
    
    # Apply Gaussian filter for additional smoothing
    filtered_signal = gaussian_filter1d(filtered_signal, sigma=gaussian_sigma)
    
    return filtered_signal

def create_patient_trajectory_plot(cohorts, titles, save_path, key=None, FILTERED_THRESHOLD=None, disable_title=False):
    """
    Create a 2x2 subplot showing individual patient trajectories.
    
    Args:
        cohorts (list): List of patient DataFrames
        titles (list): List of subplot titles
        save_path (str): Path to save the figure
    """
    # Configuration for subplot arrangement
    FILTERED_THRESHOLD = round(FILTERED_THRESHOLD, 2)
    n_rows, n_cols = 2, 2
    plot_order = [1, 0, 3, 2]  # Custom ordering for presentation
    
    # Create figure and subplots
    fig, axs = plt.subplots(
        n_rows, n_cols, 
        figsize=(13, 8), 
        gridspec_kw={'wspace': 0.25}
    )
    
    # Plot each patient cohort
    for plot_idx, cohort_idx in enumerate(plot_order):
        # Get current cohort data
        df_patient = cohorts[cohort_idx].copy()
        title = titles[cohort_idx]
        
        # Sort by date for proper time series visualization
        df_patient = df_patient.sort_values(by='date')
        
        # Calculate subplot position
        row_idx = plot_idx % n_rows
        col_idx = plot_idx // n_rows
        ax = axs[row_idx, col_idx]
        
        # Plot raw data points
        ax.scatter(
            df_patient['date'], 
            df_patient['pred'], 
            color='#7a9973', 
            alpha=0.2,
            label='Raw predictions'
        )
        
        # Apply smoothing and plot trend line
        smoothed_signal = apply_smoothing_filter(df_patient['pred'].values)
        ax.plot(
            df_patient['date'].values, 
            smoothed_signal, 
            alpha=0.6, 
            color='#7a9973',
            linewidth=2,
            label='Smoothed trend'
        )
        ## add a b c d in the top left corner 
        # text(-0.05, 1.05, , fontsize=14, fontweight='bold', transform=ax.transAxes)
        # ax.text(-0.1, 1.1, ['a)', 'c)', 'b)', 'd)'][plot_idx], transform=ax.transAxes, fontsize=14, fontfamily='Calibri', fontweight='bold', 
        #   verticalalignment='top', horizontalalignment='left')
        # Customize subplot
        ax.set_ylim(0, 1)
        ax.set_ylabel('Model Score', fontsize=FONT_SIZE-2)
        ax.set_xlabel('Time (Month)', fontsize=FONT_SIZE-2)
        ax.set_title(title, fontsize=FONT_SIZE, pad=10)
        
        # Format x-axis with monthly ticks
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        ## set the font size of the x-axis tick labels to FONT_SIZE
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_SIZE-2)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_SIZE-2)
        
        ## write the month number as the x-axis tick labels
        
        # ax.set_xticklabels([f'{int(tick)}' for tick in ax.get_xticks()])
    
    # Add overall figure labels
    # fig.supxlabel('Date (Point=Night, Tick=Month)', y=0.05, fontsize=12)
    # fig.supxlabel('Time (Month)', y=0.05, fontsize=FONT_SIZE)
    # fig.supylabel('Model Score', x=0.09, fontsize=12)
    
    # Adjust layout and save
    if not disable_title:
        fig.suptitle(key + f" FILTERED_THRESHOLD: {FILTERED_THRESHOLD}")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.22)
    save_path = save_path.replace('.png', f'_{key}_{FILTERED_THRESHOLD}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Figure saved to: {save_path}")

def plot_loading_curve(df, key, save_path, FILTERED_THRESHOLD=None):
    df1007 = df[df['pid'] == '1007'].copy()
    if FILTERED_THRESHOLD is None:
        FILTERED_THRESHOLD = df[key].quantile(QUANTILE)
    
    start1 = pd.Timestamp('2020-12-15')
    start2 = pd.Timestamp('2020-12-22')
    start3 = pd.Timestamp('2021-01-28')
    start = start1 - pd.Timedelta(days=23)
    end = start3 + pd.Timedelta(days=21)
    fig, ax = plt.subplots(figsize=(12, 4))
    section0 = df1007[(df1007['pid'] == '1007') & (df1007['date'] > start) &  (df1007['date'] <= start1)]  # Control
    section1 = df1007[(df1007['pid'] == '1007') & (df1007['date'] > start1) & (df1007['date'] <= start2)]   # 37.5mg
    section2 = df1007[(df1007['pid'] == '1007') & (df1007['date'] > start2) & (df1007['date'] <= start3)]  # 75mg
    section3 = df1007[(df1007['pid'] == '1007') & (df1007['date'] > start3) & (df1007['date'] < end)]     # 150mg
    ax.scatter(section0[section0[key] < FILTERED_THRESHOLD]['date'], section0[section0[key] < FILTERED_THRESHOLD]['pred'], color='green', label='0mg')
    ax.scatter(section1[section1[key] < FILTERED_THRESHOLD]['date'], section1[section1[key] < FILTERED_THRESHOLD]['pred'], color='orange', label='37.5mg')
    ax.scatter(section2[section2[key] < FILTERED_THRESHOLD]['date'], section2[section2[key] < FILTERED_THRESHOLD]['pred'], color='red', label='75mg')
    ax.scatter(section3[section3[key] < FILTERED_THRESHOLD]['date'], section3[section3[key] < FILTERED_THRESHOLD]['pred'], color='maroon', label='150mg')
    # ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))  # every Monday

    # 2. Set major ticks at the start of each month (for labels)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))  # e.g. "Jan 2024"
    ax.set_xlabel('Month')
    ax.set_ylabel('Model Score')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title('Patient Loading Venlafaxine (Not Thresholded)')
    save_path = save_path.replace('.png', f'_{key}_{FILTERED_THRESHOLD}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    ## put an X over the dots where bad_signal_time > 50


def main():
    """Main analysis pipeline for patient trajectory visualization."""
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    os.makedirs('filtered_figure', exist_ok=True)
    os.makedirs('filtered_loading_curve', exist_ok=True)
    ## BEST: ['0h_4h_TRIM_none', '0h_6h_TRIM_none', '0h_3h_TRIM_none', '0.25h_3h_TRIM_none', '0.5h_3h_TRIM_left', '0.25h_9h_TRIM_none', '1h_6h_TRIM_none','0.25h_6h_TRIM_both','0.25h_6h_TRIM_left','0h_6h_TRIM_left', '1h_4h_TRIM_left', '1h_4h_TRIM_none',]
    ## OR: 
    # '0h_4h_TRIM_none', '0.25h_4h_TRIM_none', @ 60 , 0h_6h_TRIM_none @ 90, 80, 0h_3h_TRIM_none @ 30 
    
    # for key in tqdm(['0h_4h_TRIM_left', '0h_4h_TRIM_right', '0h_4h_TRIM_both', '1h_4h_TRIM_none', '0.25h_4h_TRIM_left', '0.25h_4h_TRIM_right', '0.25h_4h_TRIM_both', '0.5h_4h_TRIM_left', '0.5h_4h_TRIM_right', '0.5h_4h_TRIM_both', '0.5h_4h_TRIM_none', '1h_4h_TRIM_left', '1h_4h_TRIM_right', '1h_4h_TRIM_both', ]):
    #     print("Extracting patient cohorts...")
    #     cohorts, titles, FILTERED_THRESHOLD = extract_patient_cohorts(df, key=key)
        
    #     # Print cohort information
    #     print("\nPatient cohort summary:")
    #     for i, (cohort, title) in enumerate(zip(cohorts, titles)):
    #         date_range = f"{cohort['date'].min().strftime('%Y-%m-%d')} to {cohort['date'].max().strftime('%Y-%m-%d')}"
    #         print(f"  {i+1}. {title}: {len(cohort)} observations ({date_range})")
        
    #     print("\nCreating patient trajectory visualization...")

    #     save_path = os.path.join('filtered_figure/figure_5_v2_.png')
    #     create_patient_trajectory_plot(cohorts, titles, save_path, key=key, FILTERED_THRESHOLD=FILTERED_THRESHOLD)
    #     save_path = 'filtered_loading_curve/figure_5_v2_.png'
    #     plot_loading_curve(df, key, save_path)
    #     print("Analysis completed successfully!")
    for key, FILTERED_THRESHOLD in tqdm([['0h_4h_TRIM_right',60]]): #, ['0h_4h_TRIM_both',60], ['0h_4h_TRIM_none',60], ['0.25h_4h_TRIM_none', 60], ['0h_6h_TRIM_none', 90],['0h_6h_TRIM_none', 80], ['0h_3h_TRIM_none', 30]]):
        save_path = os.path.join('filtered_figure/figure_5_v2_.png')
        cohorts, titles, _ = extract_patient_cohorts(df, key=key, FILTERED_THRESHOLD=FILTERED_THRESHOLD)
        create_patient_trajectory_plot(cohorts, titles, save_path, key=key, FILTERED_THRESHOLD=FILTERED_THRESHOLD, disable_title=True)
        save_path = 'filtered_loading_curve/figure_5_v2_.png'
        plot_loading_curve(df, key, save_path, FILTERED_THRESHOLD=FILTERED_THRESHOLD)
        print("Analysis completed successfully!")

    print('done')

if __name__ == "__main__":
    main()