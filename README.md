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
- Required Python packages (see requirements.txt if available)


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

- Excepted Outputs: 
Analysis Scripts: 
- PNG files with the same file name as the python file are created. If data creation, appropriate csv is created in the data folder 
Model Inference:
- The script takes in a data file and outputs an antidepressant score from 0 - 1, indicating likelihood of antidepressant use. 

Expected Run Time for Analyses: < 30 minutes
- How to Run: 

Analysis Scripts: 
Choose appropriate file. Anonymized data csv is provided in the data subfolder. Run the script from the main directory, i.e. python biomarker/analysis/figure_4a.py. Output is automatically saved/printed.

Model Inference: 
To run the model binary, simply input a nocturnal breathing signal (examples provided in data folder for positive and negative example), and run the python script as follows:


Scripts: 

- Data Processing:
  - data/anonymize_data.py
  - data/create_master_dataset.py
  - data/create_synthetic_dataset.py

- Sleep Stage Analysis (Figure 2):
  - sleep_stage_analysis/Figure_2ab_calculate_metrics.py
  - sleep_stage_analysis/REM_ALIGNMENT_PROD.py
  - sleep_stage_analysis/Figure_2ab_generate_figures.py
  - sleep_stage_analysis/Figure_2c.py

- Biomarker Analysis (Figures 3-6):
  - biomarker/analysis/figure_3a.py
  - biomarker/analysis/figure_3b.py
  - biomarker/analysis/figure_3c.py
  - biomarker/analysis/figure_3d.py
  - biomarker/analysis/figure_4.py
  - biomarker/analysis/figure_4a.py
  - biomarker/analysis/figure_4b.py
  - biomarker/analysis/figure_4c.py
  - biomarker/analysis/figure_4d.py
  - biomarker/analysis/figure_4e.py
  - biomarker/analysis/figure_5.py
  - biomarker/analysis/figure_6ab.py
  - biomarker/analysis/figure_6c.py

- Supplemental Figures:
  - supplemental_figures/check_age_dose_correlation.py
  - supplemental_figures/check_auprc.py
  - supplemental_figures/check_dataset_durations.py
  - supplemental_figures/check_label_noise_sensitivity.py
  - supplemental_figures/check_other_tsne_lda.py
  - supplemental_figures/check_other_tsne.py
  - supplemental_figures/per_patient_variability.py
  - supplemental_figures/rem_latency_correlation_supplemental.py 



Anonymized Datasets: 

- Biomarker predictions:
   - anonymized_inference_v6emb_3920_all.csv
- Results from Baselines: 
   - anonymized_df_baseline.csv - baseline: sleep stage only
   - anonymized_df_baseline_eeg.csv - baseline: sleep stage + eeg
- Example breathing signals for model inference: 
   - anonymized_negative_example.npz - sleep data from patient taking antidepressants
   - anonymized_positive_example.npz - sleep data from patient not taking antidepressants 
- Additional Dataset Labels:
   - anonymized_antidep_taxonomy_all_datasets_v6.csv - antidepressant labels
   - anonymized_shhs_mros_cfs_wsc_ahi.csv - apnea labels
   - anonymized_mros1-dataset-augmented-live.csv - psychotropic medications
   - anonymized_mros2-dataset-augmented-live.csv - psychotropic medications
   - anonymized_wsc-dataset-0.7.0.csv - zung index, psychotropic medications 
- Sleep Metrics:
   - anonymized_control_pwr_sleep.npy - predicted EEG band powers
   - anonymized_antidep_pwr_sleep.npy - predicted EEG band powers
   - anonymized_figure_draft_v16_rem_latency.csv - Sleep hypnogram features
   - anonymized_so_beta_powers_sleep_only.csv - EEG Band Powers