import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 
from tqdm import tqdm 
import scipy.stats
from numpy import convolve
from ipdb import set_trace as bp 

# Configuration
datasets = ['mros','wsc','shhs','cfs']
num_matches = 3
age_tolerance = 5 

# File paths
df = pd.read_csv('../data/master_dataset.csv')
df = df[['filename', 'label', 'dataset', 'mit_age', 'mit_gender']]
df = df[df['dataset'].isin(datasets)]  # Filter for datasets that have ground truth sleep stage and EEG 
df = df.groupby('filename').agg('first').reset_index()
print(df.shape)

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



def pearsonr(x, y, return_pval=False):
    nans = np.isnan(x) | np.isnan(y)
    x = x[~nans]
    y = y[~nans]
    if not return_pval:
        return np.round(scipy.stats.pearsonr(x, y).correlation, 3)
    else: 
        return np.round(scipy.stats.pearsonr(x, y).correlation, 3), float(f"{scipy.stats.pearsonr(x, y).pvalue:.0e}")

def smooth(arr, n):
    if n < 1:
        raise ValueError("n must be at least 1")
    pad_width = n // 2
    padded_arr = np.pad(arr, pad_width, mode='edge')
    smoothed = np.convolve(padded_arr, np.ones(n)/n, mode='valid')
    return smoothed

def get_dataset(file):
    if 'cfs' in file:
        return 'cfs'
    elif 'mros' in file:
        if 'visit1' in file:
            return 'mros1'
        elif 'visit2' in file:
            return 'mros2'
    elif 'wsc' in file:
        return 'wsc'
    elif 'shhs1' in file:
        return 'shhs1'
    elif 'shhs2' in file:
        return 'shhs2'
    print('Dataset not recognized for file:', file)
    return ''

def process_stages(stages):
    stages[stages < 0] = 0
    stages[stages > 5] = 0
    stages = stages.astype(int)
    mapping = np.array([0, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0], np.int64)
    return mapping[stages]

def get_mage_stage(filename, gt=True, dataset=None):
    filename = filename.split('/')[-1]
    STAGE_PREFIX = f'/data/netmit/wifall/ADetect/data/{dataset}/stage/'
    MAGE_PREFIX = f'/data/netmit/sleep_lab/filtered/MAGE/{dataset}_new/mage/cv_0/'
    if gt:
        gt_path = '/data/netmit/sleep_lab/filtered/MAGE/DATASET/c4_m1_multitaper'
        gt_dir = gt_path.replace('DATASET',dataset)
        if not os.path.exists(gt_dir):
            gt_dir = gt_path.replace('DATASET',dataset+'_new')
        if not os.path.exists(gt_dir):
            print(f'{dataset} not found')
            return None, None
        mage_gt = np.load(os.path.join(gt_dir, filename))['data']
        mage_gt = normalize_gt(mage_gt, dataset)
    key = 'pred'
    
    stage = np.load(os.path.join(STAGE_PREFIX, filename))
    stage = stage['data'][::int(30*stage['fs'])]
    stage = process_stages(stage)
    mage = np.load(os.path.join(MAGE_PREFIX, filename))[key]

    if mage.shape[1] < 4 * 60 * 2 or len(stage) < 4*60*2:
        print('less than 4 hrs mage')
        return None, None
    
    if mage.shape[1] != len(stage):
        if not len(stage) > mage.shape[1]:
            print('weird, stage is less than mage shape', len(stage), mage.shape[1])
            return None, None
    stage = stage[:mage.shape[1]]
    if gt:
        return mage, mage_gt[:,:mage.shape[1]], stage
    return mage, None, stage
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

def naive_power_post_onset(mage, stage, minutes=60, mean=True, which_stage=[1,2,3]):
    if mage is None:
        return None
    
    mage = mage.copy()
    stage = stage.copy()
    
    if type(which_stage) is list:
        is_n2 = np.zeros_like(stage).astype(bool)
        for stg in which_stage:
            is_n2 = is_n2 + (stage == stg)
    else:    
        is_n2 = (stage == which_stage)
    
    if np.sum(stage > 0) == 0:
        return None
    
    onset_idx = np.argwhere(stage > 0)[0][0]
    end_idx = min(len(stage), onset_idx + 2 * minutes)
    mage[:, ~is_n2] = np.nan 
    
    if mean:
        return np.nanmean(mage[:, onset_idx:end_idx], 1)
    else:
        return mage[:, onset_idx:end_idx]



CONTROL_MATCHING = False 
# Create age and gender matched control cohort
if CONTROL_MATCHING:
    mappings = {}
    for i, row in tqdm(df[df['label'] == 1].iterrows()):
        age = row['mit_age']
        gender = row['mit_gender']
        dataset = row['dataset']
        valid_matches = df[(df['label'] == 0) * (df['dataset'] == dataset) * (df['mit_gender'] == gender) * 
                        (df['mit_age'] >= age - age_tolerance) * (df['mit_age'] <= age + age_tolerance)]
        if len(valid_matches) == 0:
            print(f'No valid matches for {row["filename"]} in {dataset}')
            continue 
        valid_matches = valid_matches.sample(num_matches)
        mappings[row['filename']] = valid_matches['filename'].tolist()

    all_controls = [item for sublist in mappings.values() for item in sublist]
    all_antideps = list(mappings.keys())
    print(len(all_controls), len(all_antideps))
    print('done')

else:
    all_controls = df[df['label'] == 0]['filename'].tolist()
    all_antideps = df[df['label'] == 1]['filename'].tolist()


# Initialize data containers
control_pwr_sleep = [] 
antidep_pwr_sleep = [] 
antidep_pwr_sleep_gt = [] 
control_pwr_sleep_gt = [] 
# Process antidepressant files
for file in tqdm(all_antideps):
    dataset = get_dataset(file)
    mg, mage_gt, st = get_mage_stage(file, gt=True, dataset=dataset)
    if mg is None or st is None:
        continue
    
    mage2_sleep = naive_power_post_onset(mg, st, minutes=1000000, mean=True, which_stage=[1,2,3,4])
    mage2_sleep_gt = naive_power_post_onset(mage_gt, st, minutes=1000000, mean=True, which_stage=[1,2,3,4])

    if mage2_sleep is not None and ~np.any(np.isnan(mage2_sleep)) and ~np.any(np.isinf(mage2_sleep)):
        antidep_pwr_sleep.append(mage2_sleep)
    if mage2_sleep_gt is not None and ~np.any(np.isnan(mage2_sleep_gt)) and ~np.any(np.isinf(mage2_sleep_gt)):
        antidep_pwr_sleep_gt.append(mage2_sleep_gt)
# Process control files
for file in tqdm(all_controls):
    dataset = get_dataset(file)
    mg, mage_gt, st = get_mage_stage(file, gt=True, dataset=dataset)
    
    if mg is None or st is None:
        continue
    
    mage2_sleep = naive_power_post_onset(mg, st, minutes=1000000, mean=True, which_stage=[1,2,3,4])
    mage2_sleep_gt = naive_power_post_onset(mage_gt, st, minutes=1000000, mean=True, which_stage=[1,2,3,4]) 
    if mage2_sleep is not None and ~np.any(np.isnan(mage2_sleep)) and ~np.any(np.isinf(mage2_sleep)):
        control_pwr_sleep.append(mage2_sleep)
    if mage2_sleep_gt is not None and ~np.any(np.isnan(mage2_sleep_gt)) and ~np.any(np.isinf(mage2_sleep_gt)):
        control_pwr_sleep_gt.append(mage2_sleep_gt)
# Convert to numpy arrays
control_pwr_sleep = np.stack(control_pwr_sleep)
antidep_pwr_sleep = np.stack(antidep_pwr_sleep)




# Calculate percent differences with bootstrap
whole_sleep2, whole_sleep_lower, whole_sleep_upper = bootstrap_percent_difference(antidep_pwr_sleep, control_pwr_sleep)
whole_sleep_gt2, whole_sleep_gt_lower, whole_sleep_gt_upper = bootstrap_percent_difference(antidep_pwr_sleep_gt, control_pwr_sleep_gt)
if True:
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(smooth(whole_sleep2, 3), label='Sleep EEG Reconstruction', ls='dashed', color='black')
    ax.fill_between(np.arange(0, 256), smooth(whole_sleep_lower, 3), smooth(whole_sleep_upper, 3), alpha=0.1, color='black')
    ax.plot(smooth(whole_sleep_gt2, 3), label='Sleep EEG', ls='dashed', color='gray')
    ax.fill_between(np.arange(0, 256), smooth(whole_sleep_gt_lower, 3), smooth(whole_sleep_gt_upper, 3), alpha=0.1, color='black')
    ax.axhline(y=0, color='gray', ls='dotted', alpha=0.3)
    for x in [1, 4, 8, 12, 16]:
        ax.axvline(x * 8, color='gray', ls='dotted', alpha=0.3)

    ax.set_xlim(-0.01, 256.01)
    ax.set_xticks(np.arange(0, 257, 32))
    ax.set_xticklabels(np.arange(0, 33, 4))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Percent Difference in Power\n(Antidepressants - Controls)')
    ax.legend()
    plt.savefig(f'check_reconstruction_error_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
bp() 
print('done')