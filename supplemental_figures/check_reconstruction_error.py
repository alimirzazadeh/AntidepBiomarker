import numpy as np
import os 
from tqdm import tqdm
from ipdb import set_trace as bp
import matplotlib.pyplot as plt
import pandas as pd
# gt_path = '/data/netmit/sleep_lab/filtered/c4_m1_multitaper/mros1'
gt_path = '/data/netmit/sleep_lab/filtered/MAGE/DATASET/c4_m1_multitaper'
pred_path = '/data/netmit/sleep_lab/filtered/MAGE/DATASET/mage/cv_0'

df = pd.read_csv('../data/master_dataset.csv',usecols=['filename','label'])
print(df.shape)
df['filename'] = df['filename'].apply(lambda x: x.split('/')[-1])

def calculate_l1_error(gt, pred):
    return np.abs(gt - pred)

def calculate_l2_error(gt, pred):
    return np.square(gt - pred)

def mean_percent_difference(observed, expected):
    a = observed
    b = expected
    b75 = np.nanpercentile(b, 75, 0) 
    b25 = np.nanpercentile(b, 25, 0)
    b_iqr = 1.5 * (b75 - b25)
    bmin = b25 - 1.5 * b_iqr
    bmax = b75 + 1.5 * b_iqr
    
    a = a - bmin
    b = b - bmin
    a = a / (bmax - bmin)
    b = b / (bmax - bmin)
    mean_a = np.nanmean(a, 0)
    mean_b = np.nanmean(b, 0)
    denom = mean_b
    
    return (mean_a - mean_b) / denom * 100

def normalize_gt(eeg, dataset ):
    aligned_stats = {
    "bwh": {
        "mean": -119.81631919400787,
        "std": 9.759356123639623
    },
    "ccshs": {
        "mean": -121.27895213781851,
        "std": 9.40345097820264
    },
    "cfs": {
        "mean": -121.19160358663257,
        "std": 9.59165062387462
    },
    "chat1": {
        "mean": -113.50761420277698,
        "std": 9.08179307747247
    },
    "chat2": {
        "mean": -114.89264007606901,
        "std": 9.716837725098982
    },
    "chat3": {
        "mean": -116.87718779950092,
        "std": 9.43450530792485
    },
    "mesa": {
        "mean": -122.68688353498257,
        "std": 9.60216508752495
    },
    "mgh": {
        "mean": -119.62848254989814,
        "std": 9.871857334916152
    },
    "mgh2": {
        "mean": -120.55325045281961,
        "std": 9.842009745631655
    },
    "mros1": {
        "mean": -120.98222843034381,
        "std": 9.363104704238623
    },
    "mros2": {
        "mean": -118.22438446922463,
        "std": 10.073216867662019
    },
    "p18c": {
        "mean": -120.74857898328052,
        "std": 10.054891007302812
    },
    "shhs1": {
        "mean": -121.10152070581054,
        "std": 9.97620536159769
    },
    "shhs2": {
        "mean": -121.54034787678135,
        "std": 9.872548910735638
    },
    "sof": {
        "mean": -119.19408423981945,
        "std": 10.18193411752617
    },
    "stages": {
        "mean": -118.39168242635253,
        "std": 9.337185496757476
    },
    "wsc": {
        "mean": -119.98596507662205,
        "std": 9.550418675015477
    }
    }
    target_mean = -120.
    target_std = 7.
    eeg = (eeg - aligned_stats[dataset]['mean']) / aligned_stats[dataset]['std'] * target_std + target_mean
    eeg = np.clip(eeg, -140, -90)
    eeg = ((eeg + 140) / 50).astype(np.float32)
    eeg = (eeg - 0.5) / 0.5
    return eeg

def calculate_reconstruction_error(file, dataset, gt_dir, pred_dir, method:str='l1'):
    gt = np.load(os.path.join(gt_dir, file))['data']
    pred = np.load(os.path.join(pred_dir, file))['pred']
    gt = normalize_gt(gt, dataset)
    if method == 'l1':
        error = calculate_l1_error(gt.mean(1), pred.mean(1))
    elif method == 'l2':
        error = calculate_l2_error(gt.mean(1), pred.mean(1))
    elif method == 'percent_diff':
        error = mean_percent_difference(gt.T, pred.T)
    return error

def main():
    fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
    for dataset in ['cfs','shhs1', 'shhs2','mros1','mros2','wsc']:
        gt_dir = gt_path.replace('DATASET',dataset)
        if not os.path.exists(gt_dir):
            gt_dir = gt_path.replace('DATASET',dataset+'_new')
        if not os.path.exists(gt_dir):
            print(f'{dataset} not found')
            continue
        pred_dir = pred_path.replace('DATASET',dataset)
        if not os.path.exists(pred_dir):
            pred_dir = pred_path.replace('DATASET',dataset+'_new')
        if not os.path.exists(pred_dir):
            print(f'{dataset} not found')
            continue
        gt_files = os.listdir(gt_dir)
        pred_files = os.listdir(pred_dir)
        all_files = np.intersect1d(gt_files, pred_files)
        all_errors_l1 = []
        all_errors_l2 = []
        all_labels = [] 
        all_errors_percent_diff = []
        for file in tqdm(all_files):
            error = calculate_reconstruction_error(file, dataset, gt_dir=gt_dir, pred_dir=pred_dir)
            all_errors_l1.append(error)
            error = calculate_reconstruction_error(file, dataset, gt_dir=gt_dir, pred_dir=pred_dir, method='l2')
            all_errors_l2.append(error)
            error = calculate_reconstruction_error(file, dataset, gt_dir=gt_dir, pred_dir=pred_dir, method='percent_diff')
            all_errors_percent_diff.append(error)
            all_labels.append(df[df['filename'] == file]['label'].values[0])
        all_errors_l1 = np.stack(all_errors_l1)
        all_errors_l2 = np.stack(all_errors_l2)
        all_errors_percent_diff = np.stack(all_errors_percent_diff)
        all_labels = np.array(all_labels)

        for i in range(2):
            ax[i, 0].plot(np.mean(all_errors_l1[all_labels == i], 0), label=f'{dataset} {['Control', 'Antidep'][i]}', alpha=0.5)
            ax[i, 1].plot(np.mean(all_errors_l2[all_labels == i], 0), label=f'{dataset} {['Control', 'Antidep'][i]}', alpha=0.5)
            ax[i, 2].plot(np.mean(all_errors_percent_diff[all_labels == i], 0), label=f'{dataset} {['Control', 'Antidep'][i]}', alpha=0.5)
            ax[i, 0].legend()
            ax[i, 1].legend()
            ax[i, 2].legend()
            ax[i, 0].set_title(f'{dataset} L1 Error')
            ax[i, 1].set_title(f'{dataset} L2 Error')
            ax[i, 2].set_title(f'{dataset} Percent Difference')
        if dataset == 'cfs':
            bp() 
    bp() 
    fig.savefig('reconstruction_error.png', dpi=300, bbox_inches='tight')
        # mean_error_l1 = np.mean(all_errors_l1, 0)
        # std_error_l1 = np.std(all_errors_l1, 0)
        # mean_error_l2 = np.mean(all_errors_l2, 0)
        # std_error_l2 = np.std(all_errors_l2, 0)
        # mean_error_percent_diff = np.mean(all_errors_percent_diff, 0)
        # std_error_percent_diff = np.std(all_errors_percent_diff, 0)
        # plt.plot(mean_error_l1, label='L1')
        # plt.fill_between(np.arange(0, mean_error_l1.shape[0]), mean_error_l1 - std_error_l1, mean_error_l1 + std_error_l1, alpha=0.2)
        
if __name__ == '__main__':
    main()