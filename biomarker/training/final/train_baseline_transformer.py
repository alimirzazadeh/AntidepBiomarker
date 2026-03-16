import pandas as pd
from datetime import datetime
import itertools
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import BaselineViT
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from dataset import CrossFold, DataCollector_V2,  LabelHunter
from copy import deepcopy
from tqdm import tqdm
from ipdb import set_trace as bp
#import numpy as np
import argparse
import re
import json 
import torch.nn.functional as F
from metrics import Metrics
import numpy as np 
torch.cuda.empty_cache()
import sys 
import random
### A more updated version of train_antidep_with_rf_consistency (with filtered dataset)
USE_ONLY_STAGE_FEATURES = False 

class BinaryFocalLoss(nn.Module):
    ## AM2024: loss to focus on harder case examples
    def __init__(self, alpha=0.8, gamma=1.0, reduction='mean', logits=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.logits = logits

    def forward(self, inputs, targets):
        
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss) + 1e-8  # prevents nans when probability 0
        alpha_t = self.alpha * targets + (1 - targets) * (1 - self.alpha)
        F_loss = alpha_t * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def logger(writer, metrics, phase, epoch_index):

    for key, value in metrics.items():
        ## check for 2 class multiclass
        if type(value)!= float and len(value.shape) > 0 and value.shape[0] == 2:
            value = value[1]
        elif type(value)!= float and len(value.shape) > 0 and value.shape[0] > 2:
            bp()
        writer.add_scalar("%s/%s"%(phase, key), value, epoch_index)
    writer.flush()

def transform_modality(dataset):
    return [1 if item.startswith('hchs') else 2 if item == 'rf' else 0 for item in dataset]


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



def run_train_step(n_model, trainloader, mask=False):
    try:
        X_batch, y_batch = next(trainloader)
    except:
        trainloader = iter(train_loader)
        X_batch, y_batch = next(trainloader)
    
    X_batch = X_batch.to(device)
    
    if mask:
        print(f'Masking EEG features with ratio: {eeg_mask.sum() / len(eeg_mask)}')
        X_batch[:, eeg_mask] = 0
    
    
    
    n_model.train()
    with torch.enable_grad():
        y_label = y_batch['label']
        y_label = y_label.to(device)
        y_dataset = y_batch['dataset']
        

        t5_emb = None 
        
            
        y_pred = n_model(x=X_batch, t5_emb=t5_emb) 

        loss = loss_fn(y_pred.squeeze(1), y_label.float())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(itertools.chain(n_model.parameters()), max_norm=1.0)
        optimizer.step()
        metrics.fill_metrics(y_pred, y_label, y_dataset = y_dataset)
        
    return loss.item() 

def run_val_epoch(n_model, epoch):
    
    n_model.eval()

    with torch.no_grad():
        running_loss = 0
          
        for X_batch, y_batch in tqdm(test_loader):
            t5_emb = None 
            y_label = y_batch['label']
            y_dataset = y_batch['dataset']
            X_batch = X_batch.to(device)
            y_label = y_label.to(device)
            y_pred = n_model(x=X_batch) #gender=y_gender, age=y_age)
            loss = loss_fn(y_pred.squeeze(1), y_label.float())
            
            running_loss += loss.item()
            metrics.fill_metrics(y_pred, y_label, y_dataset = y_dataset) # feed the raw scores, not thresh'd




        
        epoch_loss = running_loss / len(test_loader)
        
        
        computed_metrics = metrics.compute_and_log_metrics(loss=epoch_loss) #, loss3=gender_loss)
            
        logger(writer, computed_metrics, 'val', epoch)
        
        metrics.clear_metrics()   
        if args.save_model: 
            print('saving at epoch: ', epoch)
            model_name = exp_name + str(epoch) + ".pt"
            model_save_path = os.path.join(exp_event_path, model_name)
            torch.save(n_model.state_dict(), model_save_path)
        
        
### load the features 

def load_and_prepare_data():
    """
    Load and prepare all datasets for analysis.
    
    Returns:
    --------
    tuple : (df, df_eeg, labels_model_baseline, model1_cols, model2_cols)
    """
    # Load datasets
    df = pd.read_csv(os.path.join(CSV_DIR,'df_baseline.csv'))
    df_eeg = pd.read_csv(os.path.join(CSV_DIR,'df_baseline_eeg.csv'))
    # labels = pd.read_csv(os.path.join(CSV_DIR,'rebuttal_nohchsrf_inference_v6emb_3920_all.csv'))
    labels = pd.read_csv(os.path.join(CSV_DIR,'inference_v6emb_3920_all.csv'))
    # labels = pd.read_csv(os.path.join(CSV_DIR,'inference_v6emb_3920_all_noise10.csv'))
    # labels = pd.read_csv(os.path.join(CSV_DIR,'inference_v6emb_2940_all_noise25.csv'))
    
    # Prepare sleep stage features dataset
    df = df.drop(columns=['dataset'])
    model1_cols = [col for col in df.columns if col not in ['filename', 'fold', 'dataset', 'label']]
    
    # Prepare EEG features dataset
    df_eeg = df_eeg.drop(columns=['dataset'])
    df_eeg = df_eeg.rename(columns = {key : key + '_eeg' for key in df_eeg.columns if key != 'filename'})
    df_eeg = df_eeg.merge(df, on='filename', how='inner')
    model2_cols = [col for col in df_eeg.columns if col not in ['filename', 'fold', 'dataset', 'label']]
    
    # Process our model predictions
    labels['filename'] = labels['filename'].apply(lambda x: x.split('/')[-1])
    labels = labels.groupby('filename', as_index=False).agg(
        lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0]
    )
    
    # Clean participant IDs across datasets
    for dataset in ['wsc', 'mros', 'shhs']:
        mask = labels['dataset'] == dataset
        labels.loc[mask, 'pid'] = labels.loc[mask, 'pid'].apply(lambda x: x[1:] if isinstance(x, str) and x else x)
    
    # Merge labels with feature datasets
    labels_subset = labels[['filename', 'label', 'fold', 'dataset', 'mit_gender', 'mit_age']]
    df = df.merge(labels_subset, on='filename', how='inner')
    df_eeg = df_eeg.merge(labels_subset, on='filename', how='inner')
    
    return df, df_eeg, model1_cols, model2_cols

class FeaturesAndDictDataset(Dataset):
    """Dataset that returns (features, y_dict) with y_dict = {'label', 'dataset', 'filename'}."""
    def __init__(self, features, labels, datasets, filenames):
        self.features = torch.from_numpy(features).float() if not isinstance(features, torch.Tensor) else features.float()
        self.labels = torch.from_numpy(labels).float() if not isinstance(labels, torch.Tensor) else labels.float()
        self.datasets = datasets
        self.filenames = filenames

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return (
            self.features[i],
            {
                'label': self.labels[i],
                'dataset': str(self.datasets[i]),
                'filename': str(self.filenames[i]),
            }
        )


def get_datasets():
    df, df_eeg, _, _ = load_and_prepare_data()
    
    train_mask = (df['fold'] != fold) & (df['dataset'] != 'wsc')
    test_mask = (df['fold'] == fold) | (df['dataset'] == 'wsc')
    
    # Sleep stage model data
    train_set = df[train_mask].copy()
    test_set = df[test_mask].copy()
    train_y = train_set['label'].values
    test_y = test_set['label'].values
    train_datasets = train_set['dataset'].values
    train_filenames = train_set['filename'].values
    test_datasets = test_set['dataset'].values
    test_filenames = test_set['filename'].values

    # EEG model data
    train_set_eeg = df_eeg[train_mask].copy()
    test_set_eeg = df_eeg[test_mask].copy()
    
    # Store metadata
    test_set_datasets = test_set['dataset'].values
    test_set_labels = test_set['label'].values
    
    # Prepare feature matrices
    train_features = train_set.drop(columns=['filename', 'fold', 'dataset', 'label'])
    train_features = train_features.values
    test_features = test_set.drop(columns=['filename', 'fold', 'dataset', 'label']).values
    
    feature_mean = np.nanmean(train_features, axis=0)
    feature_std = np.nanstd(train_features, axis=0)
    train_features = (train_features - feature_mean) / (feature_std + 1e-8)
    test_features = (test_features - feature_mean) / (feature_std + 1e-8)
    
    train_features_eeg = train_set_eeg.drop(columns=['filename', 'fold', 'dataset', 'label'])
    test_features_eeg = test_set_eeg.drop(columns=['filename', 'fold', 'dataset', 'label']).values
    eeg_mask = np.array([True if col.endswith('_eeg') else False for col in train_features_eeg.columns])
    train_features_eeg = train_features_eeg.values
    
    
    feature_mean_eeg = np.nanmean(train_features_eeg, axis=0)
    feature_std_eeg = np.nanstd(train_features_eeg, axis=0)
    train_features_eeg = (train_features_eeg - feature_mean_eeg) / (feature_std_eeg + 1e-8)
    test_features_eeg = (test_features_eeg - feature_mean_eeg) / (feature_std_eeg + 1e-8)
    
    
    train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
    test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)
    train_features_eeg = np.nan_to_num(train_features_eeg, nan=0.0, posinf=0.0, neginf=0.0)
    test_features_eeg = np.nan_to_num(test_features_eeg, nan=0.0, posinf=0.0, neginf=0.0)
    
    if USE_ONLY_STAGE_FEATURES:
        num_features = train_features.shape[1]
        train_dataset = FeaturesAndDictDataset(
            train_features, train_y, train_datasets, train_filenames
        )
        test_dataset = FeaturesAndDictDataset(
            test_features, test_y, test_datasets, test_filenames
        )
        return train_dataset, test_dataset, num_features, None
    else:
        num_features = train_features_eeg.shape[1]
        train_dataset = FeaturesAndDictDataset(
            train_features_eeg, train_y, train_datasets, train_filenames
        )
        test_dataset = FeaturesAndDictDataset(
            test_features_eeg, test_y, test_datasets, test_filenames
        )
        return train_dataset, test_dataset, num_features, eeg_mask
    ## train_features, train_y, test_features, test_y 





for fold in range(0,4):
    torch.manual_seed(20)
    np.random.seed(20)

    folder_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD_2024/"
    model_folder = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD_2024"
    CSV_DIR = '../../../data/'
    RUN_PATH = '/data/scratch/alimirz/2023/SIMON/TENSORBOARD_2024/'
    EXP_PATH = 'antidep_shhs1_shhs2_mros1_mros2_cfs_rf_hchs__wsc_lr_5e-05_bs_48_steps_4000_dpt_0.1_fold0_heads4_V5.0.6_nohchsrftune_featuredim_128_numtokenheads_4_trn_resmp___wd_0.01_bce'


    model_folder = os.path.join(RUN_PATH, EXP_PATH)
    model_folder = re.sub(r'fold\d+', f'fold{fold}', model_folder)
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
    args.lr = 1e-4


        
    # args = parser.parse_args()
    lr = args.lr
    task = args.task
    dataset_name = args.dataset
    label = args.label
    num_classes = args.num_classes
    num_class_name = f"class_{num_classes}"
    batch_size = args.bs
    debug = args.debug
    num_steps = 4000 #6000 #args.num_steps
    #num_folds = args.num_folds
    fold = args.fold
    add_name = args.add_name
    args.feature_dim = 32

    dpt = args.dropout
    dpt_str = f"_{dpt}"
    print("Label: ", label)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    args.device = device



    if args.focal_loss:
        loss_fn = BinaryFocalLoss(alpha=args.alpha, gamma=args.gamma, logits=True)
    else:
        if args.natural_reweight:
            loss_fn = nn.BCEWithLogitsLoss(reduce=False)
        else:
            loss_fn = nn.BCEWithLogitsLoss(reduce=True)

    lr_warmup_prop = 0.1


    ### Going to create two sets of datasets and loaders, one for contrastive task and the other for antidep task


    #### ANTIDEP TASK
    aa = dataset_name.split("__")
    train_datasets = [] 
    external_val_dataset = []
    for item in aa[0].split('_'):
        if item == '':
            continue
        assert item in ['hchs','mros1', 'mros2','rf', 'wsc', 'shhs1', 'shhs2', 'cfs', 'stages']
        train_datasets.append(item)
    for item in aa[1].split('_'):
        if item == '':
            continue
        assert item in ['hchs','mros1', 'mros2','rf', 'wsc', 'shhs1', 'shhs2', 'cfs', 'stages']
        assert item not in train_datasets
        external_val_dataset.append(item)



        
    trainset, testset, num_features, eeg_mask = get_datasets()
    
    trainset = torch.utils.data.Subset(trainset, range(48*8))

    args.MAGE_INPUT_SIZE = num_features
    model = BaselineViT(args.MAGE_INPUT_SIZE).to(device)

    print('Length of trainset: ', len(trainset))
    print('Length of testset: ', len(testset))

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True )
    test_loader = DataLoader(testset, batch_size=batch_size*4, shuffle=False, num_workers=20)

    print(next(iter(train_loader))[0].sum(dim=0))


    print('Baseline VIT Model')
        

    n_model = model 
    if not args.debug:
        n_model = torch.nn.DataParallel(n_model)

    n_model = n_model.to(device)

    print("Total Trainable Parameters: {:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    pretrained = '' 
    exp_name = f"BASELINE_fold{fold}_{'EEG+STAGE' if not USE_ONLY_STAGE_FEATURES else 'STAGE'}_FINAL" #{datetime.now().strftime('%Y%m%d-%H%M%S')}" #{args.label}_{dataset_name}_lr_{lr}_bs_{batch_size}_steps_{num_steps}_dpt_{args.dropout}_fold{fold}{pretrained}_heads{args.num_heads}_{add_name}_featuredim_{n_model.model.args.feature_dim}_numtokenheads_{args.num_token_heads}_{'trn_resmp' if args.training_resample else ''}_{'NOISE' if args.NOISE_PADDING else ''}_{'TAIL'+str(args.tail_length_vit) if args.tail_length_vit >=0 else ''}_wd_{str(round(args.weight_decay,4))}_{'focal'+str(round(args.gamma,2)) + str(round(args.alpha)) if args.focal_loss else 'bce'}"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder path '{folder_path}' created successfully.")
    exp_event_path = os.path.join(folder_path, exp_name)

    writer = SummaryWriter(log_dir=exp_event_path)

    argstosave = deepcopy(vars(args))
    argstosave.pop('device')
    json.dump(argstosave, open(os.path.join(exp_event_path,'args.json'),'w'), indent=4)
    with open(os.path.join(exp_event_path,'command.txt'),'w') as file: file.write(' '.join(sys.argv))




    assert task == 'binary' 


    optimizer = optim.AdamW(
        itertools.chain(n_model.parameters()),
        lr=0,
        weight_decay=args.weight_decay
    )

    scheduler = None

    metrics = Metrics(args)

    max_f1 = -1.0

    val_losses = [] 


    step = 0


    run_val_epoch(n_model, epoch=step)

    trainloader = iter(train_loader)

    running_loss = 0

    for step in tqdm(range(num_steps)):
        step += 1 
        
        if not USE_ONLY_STAGE_FEATURES:
            mask_ratio = 1 - step / 3000
            if random.random() < mask_ratio:
                mask = True 
            else:
                mask = False
        else:
            mask = False
        rl1 = run_train_step(n_model, trainloader, mask=mask)
        

        if step < args.num_steps * lr_warmup_prop:
            lr = args.lr * (step / (args.num_steps * lr_warmup_prop))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if scheduler is None:
                if not args.tuning:
                    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, int(args.num_steps * (1-lr_warmup_prop)), eta_min=0.00001)
                else:
                    scheduler = None

            else:
                scheduler.step() 
            
        
        running_loss += rl1
        
        
        if step % args.log_every_n_step == 0:
            epoch_loss = running_loss / (args.log_every_n_step) #len(train_loader)
            # epoch_loss_d = running_loss2/ (args.log_every_n_step) #len(train_loader)
            
            computed_metrics = metrics.compute_and_log_metrics(epoch_loss) #, hy_loss=epoch_loss_d) #loss3=epoch_loss3)
            logger(writer, computed_metrics, 'train', step )
            metrics.clear_metrics()
            
            running_loss = 0
            running_loss2 = 0
            
            run_val_epoch(n_model, epoch=step)
