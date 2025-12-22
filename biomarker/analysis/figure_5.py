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

# Configuration
EXP_FOLDER = '../../data/'

FONT_SIZE = 14

def sigmoid(x):
    """Apply sigmoid transformation to convert logits to probabilities."""
    return 1 / (1 + np.exp(-x))

def load_and_preprocess_data():
    """Load inference data and apply preprocessing."""
    df = pd.read_csv(os.path.join(EXP_FOLDER, 'inference_v6emb_3920_all.csv'))
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Apply sigmoid transformation to predictions
    df['pred'] = df['pred'].apply(sigmoid)
    
    return df

def extract_patient_cohorts(df):
    """
    Extract specific patient cohorts for longitudinal analysis.
    
    Returns:
        list: List of DataFrames for each patient cohort
        list: List of corresponding titles for each cohort
    """
    # Patient 1033: Stopping Sertraline
    df_1033 = df[df['pid'] == '1033'].copy()
    
    # Patient 1007: Starting Venlafaxine (filtered to before Sept 2021)
    df_1007 = df[df['pid'] == '1007'].copy()
    df_1007 = df_1007[df_1007['date'] < datetime.datetime(2021, 9, 1)]
    
    # Patient 1022: Starting Fluoxetine with known adherence issues
    df_1022 = df[df['pid'] == '1022'].copy()
    
    # Patient NIHYM875FLXFF: Starting Fluoxetine (filtered time window)
    df_xff = df[df['pid'] == 'NIHYM875FLXFF'].copy()
    df_xff = df_xff[df_xff['date'] >= datetime.datetime(2020, 8, 1)]
    
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
    
    return cohorts, titles

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

def create_patient_trajectory_plot(cohorts, titles, save_path):
    """
    Create a 2x2 subplot showing individual patient trajectories.
    
    Args:
        cohorts (list): List of patient DataFrames
        titles (list): List of subplot titles
        save_path (str): Path to save the figure
    """
    # Configuration for subplot arrangement
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
        ax.set_ylabel('Model Score', fontsize=FONT_SIZE)
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
    fig.supxlabel('Time (Month)', y=0.05, fontsize=FONT_SIZE)
    # fig.supylabel('Model Score', x=0.09, fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.22)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Figure saved to: {save_path}")

def main():
    """Main analysis pipeline for patient trajectory visualization."""
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    print("Extracting patient cohorts...")
    cohorts, titles = extract_patient_cohorts(df)
    
    # Print cohort information
    print("\nPatient cohort summary:")
    for i, (cohort, title) in enumerate(zip(cohorts, titles)):
        date_range = f"{cohort['date'].min().strftime('%Y-%m-%d')} to {cohort['date'].max().strftime('%Y-%m-%d')}"
        print(f"  {i+1}. {title}: {len(cohort)} observations ({date_range})")
    
    print("\nCreating patient trajectory visualization...")
    save_path = os.path.join('figure_5.png')
    create_patient_trajectory_plot(cohorts, titles, save_path)
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()