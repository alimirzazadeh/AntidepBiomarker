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

- Python 3.x
- R (for sleep cycle processing)
- Required Python packages (see requirements.txt if available)
- Access to sleep stage data, EEG data, and associated labels

## Pipeline Execution

### 1. Sleep Cycle Analysis

To generate Figure 2 and related sleep cycle analyses:

1. **Generate sleep cycle features:**
   ```bash
   # First, run the R script for each dataset individually
   Rscript production_sleep_cycle_official.R
   
   # Concatenate all dataset outputs into all_dataset_sleepcycle.csv
   ```

2. **Process sleep cycle features:**
   ```bash
   python process_sleep_cycle_features.py
   ```
   - **Input:** `all_dataset_sleepcycle.csv`, `labels.csv`, sleep stage data
   - **Output:** `cycle_calculations_official.csv`

3. **Generate Figure 2:**
   ```bash
   python figure_2.py
   ```
   - **Input:** `cycle_calculations_official.csv`

### 2. Ground Truth EEG Analysis

To generate Figure 3:

```bash
python figure_3.py
```
- **Input:** `cycle_calculations_official.csv`, `labels.csv`, sleep stage data, `c4_m1_multitaper` EEG data

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