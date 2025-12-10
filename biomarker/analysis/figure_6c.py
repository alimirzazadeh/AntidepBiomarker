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
df = pd.read_csv('~/2023/SIMON/Sleep-Research/temp_analysis/labels.csv')
df = df[df['dataset'].isin(datasets)]  # Filter for datasets that have ground truth sleep stage and EEG 

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

def get_post_onset_trend(mage, stage, which_stage=2):
    if type(which_stage) is list:
        is_n2 = np.zeros_like(stage).astype(bool)
        for stg in which_stage:
            is_n2 = is_n2 + (stage == stg)
    else:    
        is_n2 = (stage == which_stage)

    if len(stage) == 0 or 2 not in stage:
        return None
    onset_idx = np.argwhere(stage > 0)[0][0]

    if len(stage) > onset_idx + 6 * 60 * 2:
        stage = stage[onset_idx:onset_idx + 6*60*2]
        mage = mage[:, onset_idx:onset_idx + 6*60*2]
        is_n2 = is_n2[onset_idx:onset_idx + 6*60*2]
    else:
        stage = stage[onset_idx:]
        mage = mage[:, onset_idx:]
        is_n2 = is_n2[onset_idx:]
            
    output = np.zeros((256, 6 * 60 * 2))
    output[:] = np.nan 
    res = mage[:, :len(stage)]
    is_n2 = is_n2[:len(stage)]
    res[:, ~is_n2] = np.nan 
    output[:, :len(stage)] = res
    return output 

def get_rem_supress(filename, dataset='stages', threshold=120):
    if '/' in filename:
        filename = filename.split('/')[-1]
    STAGE_PREFIX = f'/data/netmit/wifall/ADetect/data/{dataset}/stage/'
    stage = np.load(os.path.join(STAGE_PREFIX, filename))
    stage = stage['data'][::int(30*stage['fs'])]
    stage = process_stages(stage)
    if 4 not in stage:
        return True
    return (np.argwhere(stage == 4)[0][0] - np.argwhere(stage > 0)[0][0]) > (threshold * 2)


def get_rem_latency(stage):
    try:
        rem_latency = np.argwhere(stage == 4)[0][0] - np.argwhere(stage > 0)[0][0]
        return rem_latency / 2
    except:
        return np.nan

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

def get_mage_stage(filename, gt=True, dataset='stages'):
    filename = filename.split('/')[-1]
    STAGE_PREFIX = f'/data/netmit/wifall/ADetect/data/{dataset}/stage/'
    MAGE_PREFIX = f'/data/netmit/sleep_lab/filtered/c4_m1_multitaper/{dataset}/'
    key = 'data'
    
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
    
    return mage, stage

def get_pre_post_trend(mage, stage, stage_pre=2, stage_post=4, minutes_before=30, minutes_after=5, until_first_cycle=True):
    stage = stage.copy()
    if type(stage_pre) == list and 2 in stage_pre and 3 in stage_pre and 1 in stage_pre:
        stage[stage == 3] = 2
        stage[stage == 1] = 2
        stage_pre = 2
    
    filter_msk = ((2 * minutes_before * [stage_pre]) + (2 * minutes_after * [stage_post]))[1:]
    one_hot1 = np.eye(5)[stage]
    one_hot2 = np.eye(5)[filter_msk]
    
    conv_results = np.array([convolve(one_hot1[:, i], one_hot2[::-1, i], mode='valid') for i in range(5)])
    exact_match_sum = np.sum(conv_results, axis=0) > (len(filter_msk) * 0.05)
    candidates = np.argwhere(exact_match_sum == True).squeeze()
    
    filtered_candidates = [] 
    for idx in candidates:
        segment_pre = stage[idx - 2 * minutes_before : idx]
        segment_post = stage[idx : idx + 2 * minutes_after]
        if stage[idx] == stage_post and sum(segment_pre == stage_pre) > 0.1 * minutes_before * 2 and sum(segment_post == stage_post) > 0.1 * minutes_after * 2:
            if len(filtered_candidates) > 0 and idx - filtered_candidates[-1] > (2*(minutes_before+minutes_after)):
                filtered_candidates.append(idx)
            elif len(filtered_candidates) == 0:
                filtered_candidates.append(idx)

    outputs = []
    for idx in filtered_candidates:
        segment = np.copy(stage[idx - 2 * minutes_before : idx + 2 * minutes_after])
        segment_msk = (2 * minutes_before * [stage_pre]) + (2 * minutes_after * [stage_post])
        segment_msk = segment == segment_msk 
        mage_seg = np.copy(mage[:, idx - 2 * minutes_before : idx + 2 * minutes_after])
        mage_seg[:, ~segment_msk] = np.nan 
        outputs.append(mage_seg)
    
    if len(outputs) == 0:
        return None
    if until_first_cycle:
        return np.array(outputs[0])
    return np.nanmean(np.stack(outputs, 0), 0)

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
control_early_pwr_nrem = []
antidep_early_pwr_nrem = []
control_pwr_nrem = []
antidep_pwr_nrem = []
control_pwr_rem = [] 
antidep_pwr_rem = [] 
control_pwr_sleep = [] 
antidep_pwr_sleep = [] 
control_latency = [] 
control_latency_collection = [] 
antidep_latency = []
antidep_group = [] 
antidep_onset_nrem = [] 
antidep_onset_nrem_nosupress = [] 
control_onset_nrem = [] 
control_rem_entry = [] 
antidep_rem_entry = [] 

# Process antidepressant files

for file in tqdm(all_antideps):
    dataset = get_dataset(file)
    mg, st = get_mage_stage(file, gt=True, dataset=dataset)
    if mg is None or st is None:
        continue
    #bp() 
    mage_nrem = naive_power_post_onset(mg, st, minutes=60, mean=True, which_stage=[1,2,3])
    mage2_sleep = naive_power_post_onset(mg, st, minutes=1000000, mean=True, which_stage=[1,2,3,4])
    mage2_nrem = naive_power_post_onset(mg, st, minutes=1000000, mean=True, which_stage=[1,2,3])
    mage2_rem = naive_power_post_onset(mg, st, minutes=1000000, mean=True, which_stage=4)
    
    try:
        prerem_n2 = get_pre_post_trend(mg, st, minutes_before=20, minutes_after=5, stage_pre=[1,2,3])
    except:
        prerem_n2 = None 
    
    if mage_nrem is not None and ~np.any(np.isnan(mage_nrem)) and ~np.any(np.isinf(mage_nrem)):
        antidep_early_pwr_nrem.append(mage_nrem)
        antidep_group.append(get_rem_supress(file, dataset=dataset))
    
    if mage2_nrem is not None and ~np.any(np.isnan(mage2_nrem)) and ~np.any(np.isinf(mage2_nrem)):
        antidep_pwr_nrem.append(mage2_nrem)
    
    if mage2_rem is not None and ~np.any(np.isnan(mage2_rem)) and ~np.any(np.isinf(mage2_rem)):
        antidep_pwr_rem.append(mage2_rem)

    if mage2_sleep is not None and ~np.any(np.isnan(mage2_sleep)) and ~np.any(np.isinf(mage2_sleep)):
        antidep_pwr_sleep.append(mage2_sleep)
    
    if not get_rem_supress(file, dataset=dataset) and prerem_n2 is not None:
        antidep_rem_entry.append(prerem_n2)
        antidep_onset_nrem_nosupress.append(get_post_onset_trend(mg, st, which_stage=[1,2,3]))
    elif get_rem_supress(file, dataset=dataset):
        antidep_onset_nrem.append(get_post_onset_trend(mg, st, which_stage=[1,2,3]))

bp() 
# Process control files
for file in tqdm(all_controls):
    dataset = get_dataset(file)
    mg, st = get_mage_stage(file, gt=True, dataset=dataset)
    
    if mg is None or st is None:
        continue
    
    post_onset = get_post_onset_trend(mg, st, which_stage=[1,2,3])
    if post_onset is None:
        continue 
    control_onset_nrem.append(post_onset)
    
    mage_nrem = naive_power_post_onset(mg, st, minutes=60, mean=True, which_stage=[1,2,3])
    mage2_sleep = naive_power_post_onset(mg, st, minutes=1000000, mean=True, which_stage=[1,2,3,4])
    mage2_nrem = naive_power_post_onset(mg, st, minutes=1000000, mean=True, which_stage=[1,2,3])
    mage2_rem = naive_power_post_onset(mg, st, minutes=1000000, mean=True, which_stage=4)
    
    try:
        prerem_n2 = get_pre_post_trend(mg, st, minutes_before=20, minutes_after=5, stage_pre=[1,2,3])
    except:
        prerem_n2 = None 
    
    if mage_nrem is not None and ~np.any(np.isnan(mage_nrem)) and ~np.any(np.isinf(mage_nrem)):
        control_early_pwr_nrem.append(mage_nrem)
        control_latency_collection.append(get_rem_latency(st))
    
    if mage2_nrem is not None and ~np.any(np.isnan(mage2_nrem)) and ~np.any(np.isinf(mage2_nrem)):
        control_pwr_nrem.append(mage2_nrem)
    
    if mage2_rem is not None and ~np.any(np.isnan(mage2_rem)) and ~np.any(np.isinf(mage2_rem)):
        control_pwr_rem.append(mage2_rem)
    if mage2_sleep is not None and ~np.any(np.isnan(mage2_sleep)) and ~np.any(np.isinf(mage2_sleep)):
        control_pwr_sleep.append(mage2_sleep)

    if prerem_n2 is not None:
        control_rem_entry.append(prerem_n2)

# Convert to numpy arrays
control_early_pwr_nrem = np.stack(control_early_pwr_nrem)
antidep_early_pwr_nrem = np.stack(antidep_early_pwr_nrem)
control_pwr_nrem = np.stack(control_pwr_nrem)
antidep_pwr_nrem = np.stack(antidep_pwr_nrem)
control_pwr_rem = np.stack(control_pwr_rem)
antidep_pwr_rem = np.stack(antidep_pwr_rem)
control_pwr_sleep = np.stack(control_pwr_sleep)
antidep_pwr_sleep = np.stack(antidep_pwr_sleep)
antidep_onset_nrem = np.stack(antidep_onset_nrem)
antidep_onset_nrem_nosupress = np.stack(antidep_onset_nrem_nosupress)
control_onset_nrem = np.stack(control_onset_nrem)
cre = np.stack(control_rem_entry)
are = np.stack(antidep_rem_entry)

# Create main figure with REM onset and sleep onset analysis
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])
ax10 = fig.add_subplot(gs[1, 0])
ax11 = fig.add_subplot(gs[1, 1])
ax = [[ax00, ax01], [ax10, ax11]]


bp() 
# Calculate percent differences with bootstrap
whole_sleep2, whole_sleep_lower, whole_sleep_upper = bootstrap_percent_difference(antidep_pwr_sleep, control_pwr_sleep)
whole_nrem2, whole_nrem_lower, whole_nrem_upper = bootstrap_percent_difference(antidep_pwr_nrem, control_pwr_nrem)
whole_rem2, whole_rem_lower, whole_rem_upper = bootstrap_percent_difference(antidep_pwr_rem, control_pwr_rem)

# Create power spectrum comparison figure
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(smooth(whole_nrem2, 3), label='NREM', ls='dashed', color='purple')
ax.fill_between(np.arange(0, 256), smooth(whole_nrem_lower, 3), smooth(whole_nrem_upper, 3), alpha=0.1, color='purple')
ax.plot(smooth(whole_rem2, 3), label='REM', ls='dashed', color='red')
ax.fill_between(np.arange(0, 256), smooth(whole_rem_lower, 3), smooth(whole_rem_upper, 3), alpha=0.1, color='red')

ax.axhline(y=0, color='gray', ls='dotted', alpha=0.3)
for x in [1, 4, 8, 12, 16]:
    ax.axvline(x * 8, color='gray', ls='dotted', alpha=0.3)

ax.set_xlim(-0.01, 256.01)
ax.set_xticks(np.arange(0, 257, 32))
ax.set_xticklabels(np.arange(0, 33, 4))
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Percent Difference in Power\n(Antidepressants - Controls)')
ax.legend()
plt.savefig(f'figure_3a_{"control_matching" if CONTROL_MATCHING else "no_control_matching"}_v2.png', dpi=300, bbox_inches='tight')
plt.close()

if True:
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(smooth(whole_sleep2, 3), label='Sleep', ls='dashed', color='black')
    ax.fill_between(np.arange(0, 256), smooth(whole_sleep_lower, 3), smooth(whole_sleep_upper, 3), alpha=0.1, color='black')

    ax.axhline(y=0, color='gray', ls='dotted', alpha=0.3)
    for x in [1, 4, 8, 12, 16]:
        ax.axvline(x * 8, color='gray', ls='dotted', alpha=0.3)

    ax.set_xlim(-0.01, 256.01)
    ax.set_xticks(np.arange(0, 257, 32))
    ax.set_xticklabels(np.arange(0, 33, 4))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Percent Difference in Power\n(Antidepressants - Controls)')
    ax.legend()
    plt.savefig(f'figure_3a_allsleep_{"control_matching" if CONTROL_MATCHING else "no_control_matching"}_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
bp() 
print('done')
