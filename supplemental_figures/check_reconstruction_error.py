import numpy as np
import os 
from tqdm import tqdm
from ipdb import set_trace as bp
import matplotlib.pyplot as plt
import pandas as pd
import sys 

DEBUG = sys.argv[1] == 'debug'

if DEBUG:
    DATASET_LIST = ['cfs','wsc']
else:
    DATASET_LIST = ['cfs','shhs1', 'shhs2','mros1','mros2','wsc']
# gt_path = '/data/netmit/sleep_lab/filtered/c4_m1_multitaper/mros1'
gt_path = '/data/netmit/sleep_lab/filtered/MAGE/DATASET/c4_m1_multitaper'
pred_path = '/data/netmit/sleep_lab/filtered/MAGE/DATASET/mage/cv_0'

df = pd.read_csv('../data/master_dataset.csv',usecols=['filename','label'])
print(df.shape)
df['filename'] = df['filename'].apply(lambda x: x.split('/')[-1])


def bootstrap_percent_difference(observed, expected, n_bootstrap=1000, ci=95):
    n_time = observed.shape[1]
    bootstrapped_diffs = np.zeros((n_bootstrap, n_time))

    for i in tqdm(range(n_bootstrap)):
        sample_obs = observed[np.random.choice(observed.shape[0], size=observed.shape[0], replace=True)]
        sample_exp = expected[np.random.choice(expected.shape[0], size=expected.shape[0], replace=True)]
        bootstrapped_diffs[i] = mean_percent_difference(sample_obs, sample_exp)

    lower = np.percentile(bootstrapped_diffs, (100 - ci) / 2, axis=0)
    upper = np.percentile(bootstrapped_diffs, 100 - (100 - ci) / 2, axis=0)
    mean_diff = mean_percent_difference(observed, expected)
    return mean_diff, lower, upper

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
    return error, gt.mean(1), pred.mean(1)

def main3():
    fig,ax = plt.subplots(4, 5, figsize=(20, 16))
    for fold in range(1):
        gt_dir = gt_path.replace('DATASET',f'wsc_new')
        pred_dir = pred_path.replace('DATASET',f'wsc_new').replace('cv_0',f'cv_{fold}')
        all_files = os.listdir(gt_dir)
        all_errors_l1 = []
        all_errors_l2 = []
        all_gts = []
        all_preds = []
        all_labels = [] 
        for file in tqdm(all_files):
            error, gt, pred = calculate_reconstruction_error(file, 'wsc', gt_dir=gt_dir, pred_dir=pred_dir)
            all_errors_l1.append(error)
            all_gts.append(gt)
            all_preds.append(pred)
            all_labels.append(df[df['filename'] == file]['label'].values[0])
        all_errors_l1 = np.stack(all_errors_l1)
        all_gts = np.stack(all_gts)
        all_preds = np.stack(all_preds)
        all_labels = np.array(all_labels)
        POWER_FIGURE = 0
        L1_ERROR_FIGURE = 3
        L1_ERROR_DIFFERENCE_FIGURE = 4
        GT_DIFFERENCE_FIGURE = 2
        PREDICTED_DIFFERENCE_FIGURE = 1
        ax[fold,L1_ERROR_FIGURE].plot(all_errors_l1[all_labels == 1].mean(0), label=f'Antidepressant', alpha=0.7, c='red')
        ax[fold,L1_ERROR_FIGURE].plot(all_errors_l1[all_labels == 0].mean(0), label=f'Control', alpha=0.7, c='blue')
        ax[fold,L1_ERROR_DIFFERENCE_FIGURE].plot(np.abs(all_errors_l1[all_labels == 1].mean(0) - all_errors_l1[all_labels == 0].mean(0)), alpha=0.8, c='black')
        ax[fold,POWER_FIGURE].plot(all_preds[all_labels == 0].mean(0), label=f'Control', alpha=0.7, c='blue')
        ax[fold,POWER_FIGURE].plot(all_preds[all_labels == 1].mean(0), label=f'Antidepressant', alpha=0.7, c='red')
        
        # ax[fold,3].plot(all_gts[all_labels == 0].mean(0), label=f'Control', alpha=0.5, c='blue', ls='dashed')
        # ax[fold,3].plot(all_gts[all_labels == 1].mean(0), label=f'Antidep', alpha=0.5, c='red')
        l1_error_diff = np.abs(all_preds[all_labels == 1].mean(0) - all_preds[all_labels == 0].mean(0))
        ax[fold,PREDICTED_DIFFERENCE_FIGURE].plot(l1_error_diff, alpha=0.7, c='black')
        gt_diff = np.abs(all_gts[all_labels == 1].mean(0) - all_gts[all_labels == 0].mean(0))
        ax[fold,GT_DIFFERENCE_FIGURE].plot(gt_diff, alpha=0.8, c='black')
        
        ax[fold,L1_ERROR_FIGURE].set_ylabel('Mean Absolute Error')
        ax[fold,POWER_FIGURE].set_ylabel('Predicted Power')
        ax[fold,PREDICTED_DIFFERENCE_FIGURE].set_ylabel('Absolute Difference')
        ax[fold,GT_DIFFERENCE_FIGURE].set_ylabel('Absolute Difference')
        ax[fold,L1_ERROR_DIFFERENCE_FIGURE].set_ylabel('Absolute Difference')
        ax[0,POWER_FIGURE].set_title('Predicted Power')
        ax[0,L1_ERROR_FIGURE].set_title('Mean Absolute Error of Antidepressants\nand Controls')
        ax[0,PREDICTED_DIFFERENCE_FIGURE].set_title('EEG Predictions Absolute Difference \n between Antidepressants and Control')
        ax[0,GT_DIFFERENCE_FIGURE].set_title('EEG Ground Truth Absolute Difference \n between Antidepressants and Control')
        ax[0,L1_ERROR_DIFFERENCE_FIGURE].set_title('Mean Absolute Error Difference \n between Antidepressants and Control')
        ax[0,L1_ERROR_FIGURE].legend()
        ax[0,POWER_FIGURE].legend()
        ax[fold,POWER_FIGURE].set_ylim(0,0.05)
        ax[fold,PREDICTED_DIFFERENCE_FIGURE].set_ylim(0,0.05)
        ax[fold,GT_DIFFERENCE_FIGURE].set_ylim(0,0.05)
        for i in range(5):
            ax[fold,i].set_xticks(np.arange(0, 257, 32))
            ax[fold,i].set_xticklabels(np.arange(0, 33, 4))
    plt.tight_layout()
    fig.savefig('reconstruction_error_per_fold_gt_pred.png', dpi=300, bbox_inches='tight')
    plt.close()
def main2():
    ## this one just does wsc, but does each fold separately 
    fig,ax = plt.subplots(4, 4, figsize=(16, 16))
    for fold in range(4):
        gt_dir = gt_path.replace('DATASET',f'wsc_new')
        pred_dir = pred_path.replace('DATASET',f'wsc_new').replace('cv_0',f'cv_{fold}')
        all_files = os.listdir(gt_dir)
        all_errors_l1 = []
        all_errors_l2 = []
        all_gts = []
        all_preds = []
        all_labels = [] 
        for file in tqdm(all_files):
            error, gt, pred = calculate_reconstruction_error(file, 'wsc', gt_dir=gt_dir, pred_dir=pred_dir)
            all_errors_l1.append(error)
            all_gts.append(gt)
            all_preds.append(pred)
            error, _, _ = calculate_reconstruction_error(file, 'wsc', gt_dir=gt_dir, pred_dir=pred_dir, method='l2')
            all_errors_l2.append(error)
            error, _, _ = calculate_reconstruction_error(file, 'wsc', gt_dir=gt_dir, pred_dir=pred_dir, method='percent_diff')
            all_labels.append(df[df['filename'] == file]['label'].values[0])
        all_errors_l1 = np.stack(all_errors_l1)
        all_errors_l2 = np.stack(all_errors_l2)
        all_gts = np.stack(all_gts)
        all_preds = np.stack(all_preds)
        all_labels = np.array(all_labels)
        all_errors_percent_diff_l1, all_errors_percent_diff_l1_lower, all_errors_percent_diff_l1_upper = bootstrap_percent_difference(all_errors_l1[all_labels == 1], all_errors_l1[all_labels == 0])
        all_errors_percent_diff_l2, all_errors_percent_diff_l2_lower, all_errors_percent_diff_l2_upper = bootstrap_percent_difference(all_errors_l2[all_labels == 1], all_errors_l2[all_labels == 0])
        ax[fold,0].plot(all_errors_l1[all_labels == 1].mean(0), label=f'Antidep', alpha=0.5, c='blue')
        ax[fold,0].plot(all_errors_l1[all_labels == 0].mean(0), label=f'Control', alpha=0.5, c='blue', ls='dashed')
        ax[fold,2].plot(all_errors_l2[all_labels == 1].mean(0), label=f'Antidep', alpha=0.5, c='red')
        ax[fold,2].plot(all_errors_l2[all_labels == 0].mean(0), label=f'Control', alpha=0.5, c='red', ls='dashed')
        ax[fold,1].plot(all_errors_percent_diff_l1, alpha=0.5)
        ax[fold,3].plot(all_errors_percent_diff_l2, alpha=0.5, c='red')
        ax[fold,1].set_ylabel('L1 Error Percent Difference \n(Antidepressants - Controls)')
        ax[fold,3].set_ylabel('L2 Error Percent Difference \n(Antidepressants - Controls)')
        ax[fold,0].set_ylabel('L1 Error')
        ax[fold,2].set_ylabel('L2 Error')
        ax[fold,0].set_xlabel('Frequency (Hz)')
        ax[fold,2].set_xlabel('Frequency (Hz)')
        ax[fold,1].set_xlabel('Frequency (Hz)')
        ax[fold,3].set_xlabel('Frequency (Hz)')
        ax[fold,0].set_xticks(np.arange(0, 257, 32))
        ax[fold,0].set_xticklabels(np.arange(0, 33, 4))
        ax[fold,2].set_xticks(np.arange(0, 257, 32))
        ax[fold,2].set_xticklabels(np.arange(0, 33, 4))
        ax[fold,1].set_xticks(np.arange(0, 257, 32))
        ax[fold,1].set_xticklabels(np.arange(0, 33, 4))
        ax[fold,3].set_xticks(np.arange(0, 257, 32))
        ax[fold,0].set_ylim(0.04,0.18)
        ax[fold,1].set_ylim(-12,12)
        ax[fold,3].set_xticklabels(np.arange(0, 33, 4))
        ax[fold,1].fill_between(np.arange(0, all_errors_percent_diff_l1.shape[0]), all_errors_percent_diff_l1_lower, all_errors_percent_diff_l1_upper, alpha=0.2, color='blue')
        ax[fold,3].fill_between(np.arange(0, all_errors_percent_diff_l2.shape[0]), all_errors_percent_diff_l2_lower, all_errors_percent_diff_l2_upper, alpha=0.2, color='red')
        ax[fold,2].set_ylim(0,0.04)
        ax[fold,3].set_ylim(-25,25)
        ax[fold,3].axhline(y=0, color='gray', ls='dashed', alpha=0.5)
        ax[fold,1].axhline(y=0, color='gray', ls='dashed', alpha=0.5)
        
        ax[fold,0].legend()
        ax[fold,2].legend()
        # ax[fold,2].legend(title='Percent Difference')
        # ax[fold,3].legend(title='Percent Difference')
        ax[fold,0].set_title(f'Fold {fold}\nL1 Error')
        ax[fold,2].set_title(f'L2 Error')
        ax[fold,1].set_title(f'L1 Error Percent Difference')
        ax[fold,3].set_title(f'L2 Error Percent Difference')
    plt.tight_layout()
    fig.savefig('reconstruction_error_per_fold.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    fig, ax = plt.subplots(1, 4, figsize=(30, 7), sharex=True)
    fig2, ax2 = plt.subplots(1, 4, figsize=(30, 7), sharex=True)
    all_datasets_l1_antidep = [] 
    all_datasets_l2_antidep = []
    all_datasets_l1_control = []
    all_datasets_l2_control = []
    # all_datasets_percent_diff_l1 = [] 
    # all_datasets_percent_diff_l2 = [] 
    for dataset_index, dataset in enumerate(DATASET_LIST):
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
        for file in tqdm(all_files):
            error = calculate_reconstruction_error(file, dataset, gt_dir=gt_dir, pred_dir=pred_dir)
            all_errors_l1.append(error)
            error = calculate_reconstruction_error(file, dataset, gt_dir=gt_dir, pred_dir=pred_dir, method='l2')
            all_errors_l2.append(error)
            error = calculate_reconstruction_error(file, dataset, gt_dir=gt_dir, pred_dir=pred_dir, method='percent_diff')
            all_labels.append(df[df['filename'] == file]['label'].values[0])
        all_errors_l1 = np.stack(all_errors_l1)
        all_errors_l2 = np.stack(all_errors_l2)
        all_labels = np.array(all_labels)
        all_errors_percent_diff_l1, all_errors_percent_diff_l1_lower, all_errors_percent_diff_l1_upper = bootstrap_percent_difference(all_errors_l1[all_labels == 1], all_errors_l1[all_labels == 0])
        all_errors_percent_diff_l2, all_errors_percent_diff_l2_lower, all_errors_percent_diff_l2_upper = bootstrap_percent_difference(all_errors_l2[all_labels == 1], all_errors_l2[all_labels == 0])
        all_datasets_l1_antidep.append(all_errors_l1[all_labels == 1])
        all_datasets_l2_antidep.append(all_errors_l2[all_labels == 1])
        all_datasets_l1_control.append(all_errors_l1[all_labels == 0])
        all_datasets_l2_control.append(all_errors_l2[all_labels == 0])
        # all_datasets_percent_diff_l1.append(all_errors_percent_diff_l1)
        # all_datasets_percent_diff_l2.append(all_errors_percent_diff_l2)

        ax[0].plot(all_errors_l1[all_labels == 1].mean(0), label=f'{dataset} Antidep', alpha=0.5, c=['blue','red','green','purple','orange','brown'][dataset_index])
        ax[0].plot(all_errors_l1[all_labels == 0].mean(0), label=f'{dataset} Control', alpha=0.5, c=['blue','red','green','purple','orange','brown'][dataset_index], ls='dashed')
        ax[1].plot(all_errors_l2[all_labels == 1].mean(0), label=f'{dataset} Antidep', alpha=0.5, c=['blue','red','green','purple','orange','brown'][dataset_index])
        ax[1].plot(all_errors_l2[all_labels == 0].mean(0), label=f'{dataset} Control', alpha=0.5, c=['blue','red','green','purple','orange','brown'][dataset_index], ls='dashed')
        ax[2].plot(all_errors_percent_diff_l1, label=f'{dataset}', alpha=0.5)
        ax[3].plot(all_errors_percent_diff_l2, label=f'{dataset}', alpha=0.5)
        ax[2].fill_between(np.arange(0, all_errors_percent_diff_l1.shape[0]), all_errors_percent_diff_l1_lower, all_errors_percent_diff_l1_upper, alpha=0.2, color=['blue','red','green','purple','orange','brown'][dataset_index])
        ax[3].fill_between(np.arange(0, all_errors_percent_diff_l2.shape[0]), all_errors_percent_diff_l2_lower, all_errors_percent_diff_l2_upper, alpha=0.2, color=['blue','red','green','purple','orange','brown'][dataset_index])
        ax[0].legend()
        ax[1].legend()
        ax[2].legend(title='Percent Difference')
        ax[3].legend(title='Percent Difference')
        ax[0].set_title(f'L1 Error')
        ax[1].set_title(f'L2 Error')
        ax[2].set_title(f'L1 Error Percent Difference')
        ax[3].set_title(f'L2 Error Percent Difference')
        ## tight layout 
        
        # for i in range(2):
        #     ax[i, 0].plot(np.mean(all_errors_l1[all_labels == i], 0), label=f'{dataset} {['Control', 'Antidep'][i]}', alpha=0.5)
        #     ax[i, 1].plot(np.mean(all_errors_l2[all_labels == i], 0), label=f'{dataset} {['Control', 'Antidep'][i]}', alpha=0.5)
        #     ax[i, 2].plot(np.mean(all_errors_percent_diff[all_labels == i], 0), label=f'{dataset} {['Control', 'Antidep'][i]}', alpha=0.5)
        #     ax[i, 0].legend()
        #     ax[i, 1].legend()
        #     ax[i, 2].legend()
        #     ax[i, 0].set_title(f'L1 Error')
        #     ax[i, 1].set_title(f'L2 Error')
        #     ax[i, 2].set_title(f'Percent Difference')
        # if dataset == 'cfs' and DEBUG:
        #     bp() 
    all_datasets_l1_antidep = np.concatenate(all_datasets_l1_antidep)
    all_datasets_l2_antidep = np.concatenate(all_datasets_l2_antidep)
    all_datasets_l1_control = np.concatenate(all_datasets_l1_control)
    all_datasets_l2_control = np.concatenate(all_datasets_l2_control)

    bootstrap_percent_diff_l1, bootstrap_percent_diff_l1_lower, bootstrap_percent_diff_l1_upper = bootstrap_percent_difference(all_datasets_l1_antidep, all_datasets_l1_control)
    bootstrap_percent_diff_l2, bootstrap_percent_diff_l2_lower, bootstrap_percent_diff_l2_upper = bootstrap_percent_difference(all_datasets_l2_antidep, all_datasets_l2_control)
    ax2[0].plot(np.mean(all_datasets_l1_antidep, 0), label='Antidep', alpha=0.5, c='red')
    ax2[0].plot(np.mean(all_datasets_l1_control, 0), label='Control', alpha=0.5, c='red', ls='dashed')

    ax2[1].plot(np.mean(all_datasets_l2_antidep, 0), label='Antidep', alpha=0.5, c='red')
    ax2[1].plot(np.mean(all_datasets_l2_control, 0), label='Control', alpha=0.5, c='red', ls='dashed')
    ax2[2].plot(bootstrap_percent_diff_l1, alpha=0.5, c='red')
    ax2[3].plot(bootstrap_percent_diff_l2, alpha=0.5, c='red')
    ax2[2].fill_between(np.arange(0, bootstrap_percent_diff_l1.shape[0]), bootstrap_percent_diff_l1_lower, bootstrap_percent_diff_l1_upper, alpha=0.2, color='red')
    ax2[3].fill_between(np.arange(0, bootstrap_percent_diff_l2.shape[0]), bootstrap_percent_diff_l2_lower, bootstrap_percent_diff_l2_upper, alpha=0.2, color='red')
    for i in range(4):
        ax[i].set_xticks(np.arange(0, 257, 32))
        ax[i].set_xticklabels(np.arange(0, 33, 4))
        ax2[i].set_xticks(np.arange(0, 257, 32))
        ax2[i].set_xticklabels(np.arange(0, 33, 4))
        ax[i].set_xlabel('Frequency (Hz)')
        ax2[i].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('L1 Error')
    ax[1].set_ylabel('L2 Error')
    ax[2].set_ylabel('L1 Error Percent Difference \n(Antidepressants - Controls)')
    ax[3].set_ylabel('L2 Error Percent Difference \n(Antidepressants - Controls)')
    ax2[0].set_ylabel('L1 Error')
    ax2[1].set_ylabel('L2 Error')
    ax2[2].set_ylabel('L1 Error Percent Difference \n(Antidepressants - Controls)')
    ax2[3].set_ylabel('L2 Error Percent Difference \n(Antidepressants - Controls)')


    plt.tight_layout()
    fig.savefig('reconstruction_error_per_dataset.png', dpi=300, bbox_inches='tight')
    fig2.savefig('reconstruction_error_overall.png', dpi=300, bbox_inches='tight')
        # mean_error_l1 = np.mean(all_errors_l1, 0)
        # std_error_l1 = np.std(all_errors_l1, 0)
        # mean_error_l2 = np.mean(all_errors_l2, 0)
        # std_error_l2 = np.std(all_errors_l2, 0)
        # mean_error_percent_diff = np.mean(all_errors_percent_diff, 0)
        # std_error_percent_diff = np.std(all_errors_percent_diff, 0)
        # plt.plot(mean_error_l1, label='L1')
        # plt.fill_between(np.arange(0, mean_error_l1.shape[0]), mean_error_l1 - std_error_l1, mean_error_l1 + std_error_l1, alpha=0.2)
        
if __name__ == '__main__':
    main3()