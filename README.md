# AntidepBiomarker

This repository contains the complete codebase and analysis pipeline supporting the manuscript on **Noninvasive AI Biomarker to Detect Antidepressant Intake and Enable Remote Adherence Monitoring**. The software enables reproduction of all primary and supplemental analyses, as well as inference of antidepressant exposure likelihood from nocturnal respiration signal using proposed biomarker.

---

## Overview

The pipeline consists of the following components:

- Sleep cycle feature extraction and analysis  
- Ground-truth EEG-derived feature analysis  
- Baseline model comparisons  
- Biomarker discovery, analysis, and visualization  
- Model training and inference  

Each analysis script corresponds directly to one or more figures in the manuscript.

---

## System Requirements

### Operating System
- Tested on Ubuntu 20.04 and macOS
- Expected to run on standard desktop or laptop computers

### Python
- Python 3.10  
- Analysis scripts are compatible with Python ≥ 3.9

### Hardware
- No non-standard hardware is required  
- GPU acceleration is optional and not required for demo analyses or reproduction of results

### Software Dependencies (Exact Versions)

```text
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
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
```

---

## Installation

### Installation Instructions

1. Create and activate a Python 3.10 environment.
2. Install required dependencies using pip:

```bash
pip install -r requirements.txt
```

3. Download or clone this repository and preserve the directory structure.
4. Place the provided anonymized data files into the `data/` directory if not already present.

### Typical Installation Time
- Approximately **1–2 hours** on a standard desktop computer.

---

## Demo Data

This repository includes **anonymized datasets** sufficient to demonstrate all analysis pipelines and model inference steps. Please do not share this data as they are under DUA, they are only provided for reproducibility purposes. 

All demo data are located in the `data/` directory.

---

## Running the Analyses

### Example Analysis Script

To reproduce a representative manuscript figure (e.g., Figure 4a):

```bash
python biomarker/analysis/figure_4a.py
```

### Expected Outputs

**Analysis scripts**
- PNG files are generated with the same base filename as the script
- Generated figures correspond directly to manuscript figures
- When applicable, intermediate CSV outputs are saved to the `data/` directory

**Model inference**
- The model takes a nocturnal breathing signal as input
- Outputs a scalar antidepressant likelihood score in the range **[0, 1]**

### Expected Runtime
- Typical runtime per analysis: **< 30 minutes** on a standard desktop computer

---

## Model Inference

To run model inference, provide a nocturnal breathing signal file (`.npz` format). Example positive and negative samples are included in the `data/` directory. The signal should have a 'data' and 'fs' property, indicating the raw signal and the sampling rate, respectively. 

Run the corresponding inference script from the repository root. The output is a continuous score indicating the likelihood of antidepressant use.

---

## Using the Pipeline on New Data

To apply the pipeline to new datasets:

1. Format new inputs according to the anonymized examples in the `data/` directory.
2. Place the formatted data in the appropriate location.
3. Execute the desired analysis or inference script from the repository root.

---

## Reproducibility

All quantitative results and figures reported in the manuscript and supplemental materials can be reproduced by running the analysis scripts corresponding to each figure number. Script names map directly to manuscript figures (e.g., `figure_3a.py`, `figure_4c.py`).

---

## Code Structure

### Data Processing
- `data/anonymize_data.py`  
- `data/create_master_dataset.py`  
- `data/create_synthetic_dataset.py`

### Sleep Stage Analysis (Figure 2)
- `sleep_stage_analysis/Figure_2ab_calculate_metrics.py`  
- `sleep_stage_analysis/REM_ALIGNMENT_PROD.py`  
- `sleep_stage_analysis/Figure_2ab_generate_figures.py`  
- `sleep_stage_analysis/Figure_2c.py`

### Biomarker Analysis (Figures 3–6)
- `biomarker/analysis/figure_3a.py`  
- `biomarker/analysis/figure_3b.py`  
- `biomarker/analysis/figure_3c.py`  
- `biomarker/analysis/figure_3d.py`  
- `biomarker/analysis/figure_4.py`  
- `biomarker/analysis/figure_4a.py`  
- `biomarker/analysis/figure_4b.py`  
- `biomarker/analysis/figure_4c.py`  
- `biomarker/analysis/figure_4d.py`  
- `biomarker/analysis/figure_4e.py`  
- `biomarker/analysis/figure_5.py`  
- `biomarker/analysis/figure_6ab.py`  
- `biomarker/analysis/figure_6c.py`

### Supplemental Analyses
- `supplemental_figures/check_age_dose_correlation.py`  
- `supplemental_figures/check_auprc.py`  
- `supplemental_figures/check_dataset_durations.py`  
- `supplemental_figures/check_label_noise_sensitivity.py`  
- `supplemental_figures/check_other_tsne_lda.py`  
- `supplemental_figures/check_other_tsne.py`  
- `supplemental_figures/per_patient_variability.py`  
- `supplemental_figures/rem_latency_correlation_supplemental.py`

---

## Included Anonymized Data

### Biomarker Predictions
- `anonymized_inference_v6emb_3920_all.csv`

### Baseline Results
- `anonymized_df_baseline.csv` — sleep stage only  
- `anonymized_df_baseline_eeg.csv` — sleep stage + EEG  

### Example Inference Inputs
- `anonymized_negative_example.npz` — patient taking antidepressants  
- `anonymized_positive_example.npz` — patient not taking antidepressants  

### Additional Labels
- `anonymized_antidep_taxonomy_all_datasets_v6.csv` — antidepressant labels  
- `anonymized_shhs_mros_cfs_wsc_ahi.csv` — apnea labels  
- `anonymized_mros1-dataset-augmented-live.csv` — psychotropic medications  
- `anonymized_mros2-dataset-augmented-live.csv` — psychotropic medications  
- `anonymized_wsc-dataset-0.7.0.csv` — Zung index, psychotropic medications  

### Sleep Metrics
- `anonymized_control_pwr_sleep.npy` — predicted EEG band powers  
- `anonymized_antidep_pwr_sleep.npy` — predicted EEG band powers  
- `anonymized_figure_draft_v16_rem_latency.csv` — hypnogram features  
- `anonymized_so_beta_powers_sleep_only.csv` — EEG band powers  

