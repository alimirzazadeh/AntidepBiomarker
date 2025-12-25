# AntidepBiomarker

This repository contains the code and analysis pipeline for the paper on antidepressant response biomarkers derived from sleep cycle and EEG data.

## Overview

The pipeline consists of several analysis components:
- Sleep cycle feature extraction and analysis
- Ground truth EEG analysis
- Baseline model comparisons
- Biomarker analysis and visualization
- Model training and inference

## Prerequisites

- Python 3.10
- R (for sleep cycle processing)
- Required Python packages (see requirements.txt if available)
- Access to sleep stage data, EEG data, and associated labels


## System requirements: 
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2        # drop if not used
pytorch-lightning==2.0.4
torchmetrics==0.11.4
lightning-utilities==0.9.0
numpy==1.24.4
scipy==1.10.1
pandas==1.5.3
scikit-learn==1.2.1
mne==1.4.2
umap-learn==0.5.3
numba==0.57.0
llvmlite==0.40.0
matplotlib==3.7.0

- Analysis Scripts will work in Python >3.9
- Install Time: 1-2hrs
- Installation Instructions: 
1. Install the necessary pip libraries listed above on a valid python installation. 
2. Acquire and download data, place in data folder 
3. Run all analyses scripts, as needed. Support for out-of-box compiled AI Biomarker model in development. 
- Excepted Output: 
PNG files with the same file name as the python file are created. If data creation, appropriate csv is created in the data folder 
Expected Run Time for Analyses: < 30 minutes
- How to Run: 
Choose appropriate file. Synthetic data csv is provided in the data subfolder. Access to real data must be requested and follow appropriate Data Use Agreement Protocols. 

Scripts: 

- Sleep Stage Analysis (Figure 3):
  - Figure_3ab_calculate_metrics.py: 
  - REM_ALIGNMENT_PROD.py: 
  - Figure_3ab_generate_figures.py: 
  - Figure_3c.py: 

- Biomarker Analysis: 
  - Figure 4a



Anonymized Datasets: 
- anonymized_antidep_taxonomy_all_datasets_v6.csv
- anonymized_control_pwr_sleep.npy
- anonymized_df_baseline_eeg.csv
- anonymized_df_baseline.csv
- anonymized_figure_draft_v16_rem_latency.csv
- anonymized_inference_v6emb_3920_all.csv
- anonymized_mros1-dataset-augmented-live.csv
- anonymized_mros2-dataset-augmented-live.csv
- anonymized_negative_example.npz
- anonymized_positive_example.npz
- anonymized_shhs_mros_cfs_wsc_ahi.csv
- anonymized_so_beta_powers_sleep_only.csv
- anonymized_wsc-dataset-0.7.0.csv