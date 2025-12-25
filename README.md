# AntidepBiomarker

This repository contains the following:
1. Executable model which takes in a breathing signal and outputs a score for antidepressant use, and an example positive and negative breathing signal 

2. Anonymized data tables for recreating model results (excluding MIT dataset due to IRB)

3. Scripts to generate all results and figures in the manuscript.

## Overview

The pipeline consists of several analysis components:
- Sleep stage analysis
- Baseline model comparisons
- Biomarker analysis and visualization
- Model inference

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
2. Run all analyses scripts, as needed.
- Excepted Output: 
Model inference:
The script outputs a single model score z between 0 and 1, with 1 meaning high likelihood of antidepressant use. 
Analyses scripts:
PNG files with the same file name as the python file are created. If data creation, appropriate csv is created in the data folder 
Expected Run Time for Analyses: < 30 minutes
- How to Run: 
Choose appropriate file. Anonymized data csv is provided in the data subfolder. Access to MIT data is restricted due to IRB restrictions. 

## Pipeline Execution



### 3. Baseline Model Comparisons

To generate Figure 4a and baseline comparisons:

1. **Generate model predictions:**
   ```bash
   python inference_export.py
   ```

2. **Train baseline models:**
   ```bash
   python baseline_models_V3.py
   ```
   - **Input:** Sleep stage data, EEG data, `all_dataset_sleepcycle.csv`
   - **Output:** `df_baseline.csv`, `df_baseline_eeg.csv`

3. **Generate comparison table:**
   ```bash
   python figure_4a.py
   ```
   - **Input:** `df_baseline.csv`, `df_baseline_eeg.csv`, `inference_v6emb_3920_all.csv`, `antidep_taxonomy_all_datasets_v6.csv`

### 4. Biomarker Analysis

First, ensure model predictions are exported:
```bash
python inference_export.py
```

Then generate the biomarker analysis figures:

- **Figure 4b:**
  ```bash
  python figure_4b.py
  ```

- **Figure 4c:**
  ```bash
  python figure_4c.py
  ```

- **Figure 4d:**
  ```bash
  python figure_4d.py
  ```
  - **Additional input:** Dataset-specific CSVs (mros1, mros2, wsc)

- **Figure 5:**
  ```bash
  python figure_5.py
  ```

**Common inputs for biomarker analysis:**
- `antidep_taxonomy_all_datasets_v6.csv`
- `inference_v6emb_3920_all.csv`

### 5. Model Training

To retrain the biomarker model from scratch:

1. **Train the model:**
   ```bash
   bash train.sh
   ```

2. **Generate predictions:**
   ```bash
   python inference_export.py
   ```

## Key Data Files

- `cycle_calculations_official.csv` - Processed sleep cycle features
- `all_dataset_sleepcycle.csv` - Raw sleep cycle data from all datasets
- `labels.csv` - Target labels for analysis
- `antidep_taxonomy_all_datasets_v6.csv` - Antidepressant taxonomy data
- `inference_v6emb_3920_all.csv` - Model predictions export
- `df_baseline.csv`, `df_baseline_eeg.csv` - Baseline model results

## Usage Notes

- Ensure all input data files are present before running each step
- The pipeline has dependencies between steps - follow the execution order
- Dataset-specific processing may be required for the R script step
- Model weights will be generated during training and used for inference

## Citation

If you use this code, please cite our paper: [Paper citation to be added]

## Contact

[Contact information to be added]