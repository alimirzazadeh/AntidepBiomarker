"""
Model Inference and Export Script for Antidepressant Prediction

This script performs inference on trained MageEncodingViT models across multiple folds
and exports predictions with probabilities for downstream analysis.

"""

import json
import os
import re
import math
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.stats import rankdata, norm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Local imports
from model import MageEncodingViT
from dataset import (
    CrossFold, 
    LabelHunter, 
    medication_antidep, 
    dosage_antidep,
    DataCollector_V2
)

# Set random seeds for reproducibility
torch.manual_seed(20)
np.random.seed(20)


class JSONToObject:
    """Convert JSON configuration to object with dot notation access."""
    
    def __init__(self, data):
        self.__dict__ = self._convert(data)

    def _convert(self, data):
        if isinstance(data, dict):
            return {key: self._convert(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert(item) for item in data]
        else:
            return data


def transform_modality(dataset):
    """Transform dataset names to modality indices."""
    return [1 if item.startswith('hchs') else 2 if item == 'rf' else 0 for item in dataset]


def logits_to_probs(logits):
    """Convert logits to calibrated probabilities using rank-based transformation."""
    ranks = rankdata(logits, method='average') / len(logits)
    z_scores = norm.ppf(ranks)
    return z_scores


def parse_date(filename, new_format=True):
    """Extract date from filename for RF dataset."""
    try:
        if 'rf' not in filename:
            return pd.NA
        
        dates = filename.split('/')[-1].split('_')
        return datetime(
            year=int(dates[1]),
            month=int(dates[2]),
            day=int(dates[3][:-len('.npz')])
        )
    except:
        return pd.NA


def parse_pid(filename, whichdataset):
    """Extract participant ID from filename based on dataset type."""
    if whichdataset == 'rf':
        return filename.split('/')[-2].split(' ')[0]
    elif whichdataset == 'cfs':
        return filename.split('/')[-1].split('-')[-1].split('.')[0]
    elif whichdataset == 'wsc':
        prefix = ('1' if 'visit1' in filename else 
                 '2' if 'visit2' in filename else 
                 '3' if 'visit3' in filename else 
                 '4' if 'visit4' in filename else 
                 '5' if 'visit5' in filename else None)
        return prefix + filename.split('/')[-1].split('-')[2]
    elif whichdataset == 'mros':
        prefix = '1' if 'visit1' in filename else '2' if 'visit2' in filename else None
        return prefix + filename.split('/')[-1].split('-')[2].split('.')[0]
    elif whichdataset == 'shhs':
        prefix = '1' if 'shhs1' in filename else '2' if 'shhs2' in filename else None
        return prefix + filename.split('/')[-1].split('-')[1].split('.')[0]
    elif whichdataset == 'hchs':
        return filename.split('/')[-1].split('-')[-1].split('.')[0]
    else:
        raise ValueError(f"Unknown dataset: {whichdataset}")


def parse_value(value, date, dataset):
    """Parse medication/dosage value based on date and dataset."""
    if dataset != 'rf':
        return pd.NA
    
    if isinstance(value, list):
        if isinstance(value[0], list):
            dose = (value[1] if date < value[0][0] else 
                   value[2] if date < value[0][1] else value[3])
        else:
            dose = value[1] if date < value[0] else value[2]
        return dose
    else:
        return value


def create_prediction_dataframe(args, labels, fold):
    """Create dataframe with predictions and metadata."""
    df = pd.DataFrame.from_dict(labels, orient='columns')
    df['filename'] = df['filepath']
    
    # Extract dataset information
    df['dataset'] = df['filename'].apply(
        lambda fname: ('hchs' if 'hchs' in fname else 
                      'rf' if 'rf' in fname else 
                      'mros' if 'mros' in fname else 
                      'shhs' if 'shhs' in fname else 
                      'wsc' if 'wsc' in fname else 
                      'cfs' if 'cfs' in fname else None)
    )
    
    # Parse participant IDs and dates
    df['pid'] = df.apply(lambda row: parse_pid(row['filename'], row['dataset']), axis=1)
    df['date'] = df.apply(lambda row: parse_date(row['filename'], args.CLEANED_DATA), axis=1)
    df['fold'] = fold
    
    # Parse medication information
    df['dosage'] = df.apply(
        lambda row: parse_value(dosage_antidep(row['pid']), row['date'], row['dataset']), 
        axis=1
    )
    df['medication'] = df.apply(
        lambda row: parse_value(medication_antidep(row['pid']), row['date'], row['dataset']), 
        axis=1
    )
    
    # Handle embeddings if requested
    if SAVE_EMBEDDINGS:
        embeddings = np.array(labels['latent'])
        for i in range(embeddings.shape[1]):
            df[f'latent_{i}'] = embeddings[:, i]
        df.drop('latent', axis=1, inplace=True)
    
    return df


def create_dataloaders(args, batch_size=256):
    """Create train and test data loaders."""
    dataset_name = args.dataset
    train_datasets, external_val_dataset = dataset_name.split("__")
    
    # Parse training datasets
    train_datasets = [item for item in train_datasets.split('_') if item]
    external_val_dataset = [item for item in external_val_dataset.split('_') if item]
    
    # Validate dataset names
    valid_datasets = ['hchs', 'mros1', 'mros2', 'rf', 'wsc', 'shhs1', 'shhs2', 'cfs']
    for item in train_datasets + external_val_dataset:
        assert item in valid_datasets, f"Invalid dataset: {item}"
    
    # Initialize data components
    dc = DataCollector_V2('mage', train_datasets + external_val_dataset, args.fold)
    lh = LabelHunter(args.label)
    cf = CrossFold(4, args, use_paper_split=True)
    args.num_datasets = 1
    
    # Load data for each dataset
    for dataset in dc.all_files:
        print(f"Loading dataset: {dataset}")
        labels = lh.get(dc.all_files[dataset], dataset)
        gender = lh.get_gender(dc.all_files[dataset], dataset)
        age = lh.get_age(dc.all_files[dataset], dataset)
        race = lh.get_race(dc.all_files[dataset], dataset)
        bmi = lh.get_bmi(dc.all_files[dataset], dataset)
        
        cf.add(
            labels, 
            dc.get_filepath(dataset), 
            is_rf=(dataset == 'rf'),
            is_external=(dataset in external_val_dataset),
            gender=gender,
            age=age,
            race=race,
            bmi=bmi
        )
    
    # Create data splits
    trainset, testset = cf.split(fold=args.fold, return_datasets=True)
    
    # Create data loaders
    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=20, 
        drop_last=False
    )
    test_loader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=20
    )
    
    return train_loader, test_loader


def load_model_and_args(model_folder, checkpoint_epoch, most_recent=False):
    """Load model and arguments from checkpoint."""
    if most_recent:
        available_checkpoints = [
            int(item.split('_')[-1].replace('.pt', '').replace('bce', ''))
            for item in os.listdir(model_folder) 
            if item.endswith('.pt')
        ]
        checkpoint_epoch = max(available_checkpoints)
    else:
        available_checkpoints = np.array([
            int(item.split('_')[-1].replace('.pt', '').replace('bce', ''))
            for item in os.listdir(model_folder) 
            if item.endswith('.pt')
        ])
        checkpoint_epoch = available_checkpoints[
            np.argmin(np.abs(available_checkpoints - checkpoint_epoch))
        ]
    
    print(f'Using checkpoint epoch: {checkpoint_epoch}')
    
    # Find model file
    model_files = [
        item for item in os.listdir(model_folder) 
        if item.endswith(f'{checkpoint_epoch}.pt')
    ]
    model_path = os.path.join(model_folder, model_files[0])
    
    # Load arguments
    args_path = os.path.join(model_folder, 'args.json')
    args = JSONToObject(json.load(open(args_path, 'r')))
    
    # Set default values for missing attributes
    default_attrs = {
        't5_demographics': False,
        't5_demographics_nomean': False,
        'age_input': False,
        'sex_input': False,
        'no_conv_proj': False,
        'minority_pos_oversample': False,
        'black_oversample': False,
        'modality_input': False
    }
    
    for attr, default_val in default_attrs.items():
        if not hasattr(args, attr):
            setattr(args, attr, default_val)
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return args, model_path, checkpoint_epoch


def run_inference():
    """Main inference function."""
    global dfs
    
    # Configuration
    RUN_PATH = '/data/scratch/alimirz/2023/SIMON/TENSORBOARD_2024/'
    #EXP_PATH = 'antidep_shhs1_shhs2_mros1_mros2_cfs_rf_hchs__wsc_lr_5e-05_bs_48_steps_4000_dpt_0.1_fold0_heads4_V5.0.6_nohchsrftune_featuredim_128_numtokenheads_4_trn_resmp___wd_0.01_bce'
    #EXP_PATH = 'antidep_shhs1_shhs2_mros1_mros2_cfs_hchs_rf__wsc_lr_5e-05_bs_48_steps_4000_dpt_0.1_fold0_heads4_rebuttal_rerun_plain_noduloxetine_featuredim_128_numtokenheads_4_trn_resmp___wd_0.01_bce'
    #EXP_PATH='antidep_shhs1_shhs2_mros1_mros2_cfs_hchs_rf__wsc_lr_5e-05_bs_48_steps_4000_dpt_0.1_fold0_heads4_labelnoise05_featuredim_128_numtokenheads_4_trn_resmp___wd_0.01_bce'
    #EXP_PATH='antidep_shhs1_shhs2_mros1_mros2_cfs_hchs_rf__wsc_lr_5e-05_bs_48_steps_4000_dpt_0.1_fold0_heads4_labelnoise10_featuredim_128_numtokenheads_4_trn_resmp___wd_0.01_bce'
    EXP_PATH = 'antidep_shhs1_shhs2_mros1_mros2_cfs_hchs_rf__wsc_lr_5e-05_bs_48_steps_4000_dpt_0.1_fold0_heads4_labelnoise01_featuredim_128_numtokenheads_4_trn_resmp___wd_0.01_bce'
    #EXP_PATH = 'antidep_shhs1_shhs2_mros1_mros2_cfs_hchs_rf__wsc_lr_5e-05_bs_48_steps_4000_dpt_0.1_fold0_heads4_rebuttal_rerun_plain_novenlafaxine_featuredim_128_numtokenheads_4_trn_resmp___wd_0.01_bce/'
    #EXP_PATH = 'antidep_shhs1_shhs2_mros1_mros2_cfs__wsc_lr_5e-05_bs_48_steps_4000_dpt_0.1_fold0_heads4_rebuttal_rerun_plain_nohchsrf_featuredim_128_numtokenheads_4_trn_resmp___wd_0.01_bce'

    CHECKPOINT_EPOCH = 4000
    WHICH_FOLDS = [0, 1, 2, 3]
    OLD_CHECKPOINT_EPOCH = CHECKPOINT_EPOCH
    MOST_RECENT_CHECKPOINT = CHECKPOINT_EPOCH == -1
    
    dfs = None
    
    # Process each fold
    for fold in WHICH_FOLDS:
        print(f"\nProcessing fold {fold}...")
        
        # Setup model folder path
        model_folder = os.path.join(RUN_PATH, EXP_PATH)
        model_folder = re.sub(r'fold\d+', f'fold{fold}', model_folder)
        
        # Load model and configuration
        args, model_path, actual_checkpoint = load_model_and_args(
            model_folder, CHECKPOINT_EPOCH, MOST_RECENT_CHECKPOINT
        )
        
        # Load T5 embeddings if needed
        if args.t5_demographics:
            t5_emb_dict = np.load(
                '/data/scratch/alimirz/DATA/T5_DEMOGRAPHICS/demographic_embeddings_v2.npz'
            )
        elif args.t5_demographics_nomean:
            t5_emb_dict = np.load(
                '/data/scratch/alimirz/DATA/T5_DEMOGRAPHICS/demographic_embeddings_v5_nomean_agenoise.npz'
            )
        else:
            t5_emb_dict = None
        
        # Create data loaders
        train_loader, test_loader = create_dataloaders(args)
        
        # Initialize and load model
        model = MageEncodingViT(args).to(args.device)
        model = torch.nn.DataParallel(model)
        
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Run inference
        for i, data in enumerate(tqdm(test_loader, desc=f"Fold {fold} inference")):
            inputs, y_batch = data
            inputs = inputs.to(args.device)
            
            # Prepare modality embeddings
            y_dataset = y_batch['dataset']
            y_dataset_emb = (
                torch.tensor(transform_modality(y_dataset), dtype=torch.int64, device=args.device)
                if args.modality_input else None
            )
            
            # Prepare T5 embeddings
            t5_emb = None
            if args.t5_demographics and t5_emb_dict is not None:
                t5_emb = torch.from_numpy(
                    np.stack([t5_emb_dict[item] for item in y_batch['filepath']], 0)
                ).to(args.device)
            elif args.t5_demographics_nomean and t5_emb_dict is not None:
                t5_emb = torch.from_numpy(
                    np.stack([
                        np.vstack([
                            t5_emb_dict[item],
                            np.zeros((13 - t5_emb_dict[item].shape[0], 768))
                        ]) for item in y_batch['filepath']
                    ], 0)
                ).to(args.device).float()
            
            # Forward pass
            with torch.no_grad():
                if SAVE_EMBEDDINGS:
                    outputs, embeddings = model(
                        inputs, 
                        t5_emb=t5_emb, 
                        return_embeddings_pred=True, 
                        modality=y_dataset_emb
                    )
                    y_batch['latent'] = embeddings[0].cpu().numpy().tolist()
                else:
                    outputs = model(inputs, t5_emb=t5_emb, modality=y_dataset_emb)
                
                outputs = outputs.cpu().numpy()
                y_batch['pred'] = list(outputs.flatten())
                
                # Create dataframe for this batch
                batch_df = create_prediction_dataframe(args, y_batch, fold)
                
                if dfs is None:
                    dfs = batch_df
                else:
                    dfs = pd.concat([dfs, batch_df], axis=0, ignore_index=True)
    
    # Calibrate predictions
    print("Calibrating predictions...")
    dfs['pred_calibrated'] = np.zeros(len(dfs))
    
    for fold in range(4):
        fold_mask = dfs['fold'] == fold
        if fold_mask.any():
            z_scores_fold = logits_to_probs(dfs[fold_mask]['pred'])
            dfs.loc[fold_mask, 'pred_calibrated'] = norm.cdf(z_scores_fold)
    
    # Save results
    model_folder = os.path.join(RUN_PATH, EXP_PATH)
    foldfolder = 'fold0' if 0 in WHICH_FOLDS else f'fold{WHICH_FOLDS[0]}'
    model_folder = re.sub(r'fold\d+', foldfolder, model_folder)
    
    dfs.loc[0, "foldername"] = EXP_PATH
    
    output_filename = (
        f"inference_{'v6' if not SAVE_EMBEDDINGS else 'v6emb'}_{actual_checkpoint}"
        f"{'_all' if len(WHICH_FOLDS) == 4 else ''}.csv"
    )
    
    output_path = os.path.join(model_folder, output_filename)
    dfs.to_csv(output_path, index=False)
    
    print(f"Results saved to: {output_path}")
    print(f"Total predictions: {len(dfs)}")


if __name__ == "__main__":
    # Global configuration
    SAVE_EMBEDDINGS = True
    
    # Run inference
    run_inference()
