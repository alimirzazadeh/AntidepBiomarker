import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
from ipdb import set_trace as bp 
from tqdm import tqdm

""" 
The goal of this script is to run two baseline models on datasets with baseline EEG and sleep stage:
  Datasets: shhs, mros, cfs, wsc
  Use the same folds as in the main script
  

"""

CSV_DIR = '../../../data/'
cycle_df = pd.read_csv(CSV_DIR + 'all_dataset_sleepcycle.csv')
def get_first_cycle_idx(filename):
    cycle = cycle_df[cycle_df['file'] == filename]['cycle_2']
    if cycle.empty:
        return None
    cycle = cycle.values[0]
    if np.isnan(cycle):
        return None
    cycle = int(cycle)
    if cycle < 0:
        return None
    return cycle
## returns durations of each stage, and total sleep duration 
def calculate_stage_durations(stage):
    results = []
    for i in range(5):
        results.append(np.sum(stage == i))
    results.append(np.sum(stage == 0) + np.sum(stage == 1) + np.sum(stage == 2) + np.sum(stage == 3) + np.sum(stage == 4))
    return {
        'NREM1 Duration': results[0],
        'NREM2 Duration': results[1],
        'NREM3 Duration': results[2],
        'REM Duration': results[3],
        'Wake Duration': results[4],
        'Total Sleep Duration': results[5]
    }


## returns the sleep latencies (both sleep onset and REM latency)
def calculate_stage_latencies(stage, basic=True):
    if basic:
        sleep_onset = np.where(stage > 0)[0][0]
        rem_latency = np.where(stage == 4)[0][0] - sleep_onset
        return {
            'Sleep Onset Latency': sleep_onset,
            'REM Latency': rem_latency
        }
    else:
        return ValueError("Not implemented yet")

## returns WASO, number of awakenings, and sleep efficiency 
def calculate_wakenings(stage, basic=True):
    if basic:
        sleep_seg = stage[np.where(stage == 0)[0][0]:np.where(stage == 0)[0][-1]]
        waso = np.sum(sleep_seg == 0)
        is_sleep = sleep_seg > 0 
        num_awakenings = np.sum(np.diff(is_sleep.astype(int)) == -1)
        sleep_efficiency = np.sum(stage > 0) / len(stage)
        return {
            'WASO': waso,
            'Number of Awakenings': num_awakenings,
            'Sleep Efficiency': sleep_efficiency
        }
    else:
        return ValueError("Not implemented yet")

## returns sleep stage transition counts: between all 5 stages 
def calculate_stage_transitions(stage):
    output = {}
    for i in range(5):
        for j in range(5):
            if i != j:
                output[f'{i} to {j}'] = np.sum((stage[:-1] == i) & (stage[1:] == j))
    return output

def process_stages( stages):
    stages[stages < 0] = 0
    stages[stages > 5] = 0
    stages = stages.astype(int)
    mapping = np.array([0, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0], np.int64)
    return mapping[stages]

## now calculate the eeg 
def calculate_band_powers(eeg):
    band_powers = {} 
    band_powers['SO'] = np.mean(eeg[0*8: 1*8])
    band_powers['Delta'] = np.mean(eeg[1*8: 4*8])
    band_powers['Theta'] = np.mean(eeg[4*8: 8*8])
    band_powers['Alpha'] = np.mean(eeg[8*8: 12*8])
    band_powers['Sigma'] = np.mean(eeg[12*8: 16*8])
    band_powers['Beta I'] = np.mean(eeg[16*8: 24*8])
    band_powers['Beta II'] = np.mean(eeg[24*8: 32*8])

    return band_powers

def calculate_band_powers_by_stage(eeg, stage):
    band_powers = {}
    for i in range(5):
        stg = ['Wake', 'NREM1', 'NREM2', 'NREM3', 'REM'][i]
        band_powers[f'SO_{stg}'] = np.mean(eeg[:,stage == i][0*8: 1*8])
        band_powers[f'Delta_{stg}'] = np.mean(eeg[:,stage == i][1*8: 4*8])
        band_powers[f'Theta_{stg}'] = np.mean(eeg[:,stage == i][4*8: 8*8])
        band_powers[f'Alpha_{stg}'] = np.mean(eeg[:,stage == i][8*8: 12*8])
        band_powers[f'Sigma_{stg}'] = np.mean(eeg[:,stage == i][12*8: 16*8])
        band_powers[f'Beta I_{stg}'] = np.mean(eeg[:,stage == i][16*8: 24*8])
        band_powers[f'Beta II_{stg}'] = np.mean(eeg[:,stage == i][24*8: 32*8])
    return band_powers

def calculate_sleep_ratio(eeg, stage, first_cycle_idx):
    band_powers = {} 
    n2_powers = np.mean(eeg[:,:first_cycle_idx][:, stage[:first_cycle_idx] == 2],1)
    n3_powers = np.mean(eeg[:,:first_cycle_idx][:, stage[:first_cycle_idx] == 3],1)
    rem_powers = np.mean(eeg[:,:first_cycle_idx][:, stage[:first_cycle_idx] == 4],1)
    
    n2_powers_rest = np.mean(eeg[:,first_cycle_idx:][:, stage[first_cycle_idx:] == 2],1)
    n3_powers_rest = np.mean(eeg[:,first_cycle_idx:][:, stage[first_cycle_idx:] == 3],1)
    rem_powers_rest = np.mean(eeg[:,first_cycle_idx:][:, stage[first_cycle_idx:] == 4],1)
    
    n2_powers = n2_powers / np.mean(n2_powers_rest)
    n3_powers = n3_powers / np.mean(n3_powers_rest)
    rem_powers = rem_powers / np.mean(rem_powers_rest)
    
    band_powers['SO_n2'] = np.mean(n2_powers[0*8: 1*8])
    band_powers['Delta_n2'] = np.mean(n2_powers[1*8: 4*8])
    band_powers['Theta_n2'] = np.mean(n2_powers[4*8: 8*8])
    band_powers['Alpha_n2'] = np.mean(n2_powers[8*8: 12*8])
    band_powers['Sigma_n2'] = np.mean(n2_powers[12*8: 16*8])
    band_powers['Beta I_n2'] = np.mean(n2_powers[16*8: 24*8])
    band_powers['Beta II_n2'] = np.mean(n2_powers[24*8: 32*8])
    band_powers['SO_n3'] = np.mean(n3_powers[0*8: 1*8])
    band_powers['Delta_n3'] = np.mean(n3_powers[1*8: 4*8])
    band_powers['Theta_n3'] = np.mean(n3_powers[4*8: 8*8])
    band_powers['Alpha_n3'] = np.mean(n3_powers[8*8: 12*8])
    band_powers['Sigma_n3'] = np.mean(n3_powers[12*8: 16*8])
    band_powers['Beta I_n3'] = np.mean(n3_powers[16*8: 24*8])
    band_powers['Beta II_n3'] = np.mean(n3_powers[24*8: 32*8])
    band_powers['SO_rem'] = np.mean(rem_powers[0*8: 1*8])
    band_powers['Delta_rem'] = np.mean(rem_powers[1*8: 4*8])
    band_powers['Theta_rem'] = np.mean(rem_powers[4*8: 8*8])
    band_powers['Alpha_rem'] = np.mean(rem_powers[8*8: 12*8])
    band_powers['Sigma_rem'] = np.mean(rem_powers[12*8: 16*8])
    band_powers['Beta I_rem'] = np.mean(rem_powers[16*8: 24*8])
    band_powers['Beta II_rem'] = np.mean(rem_powers[24*8: 32*8])
    return band_powers
    # band_powers['SO'] = 
### now calculate by dataset and put in a dataframe

if __name__ == "__main__":
    datasets = ['shhs1_new','shhs2_new','mros1_new', 'mros2_new', 'cfs', 'wsc_new']
    stage_dir = '/Users/alimirz/mnt3/data/netmit/wifall/ADetect/data/DATASET/stage/'
    eeg_dir = '/Users/alimirz/mnt3/data/netmit/sleep_lab/filtered/c4_m1_multitaper/DATASET/'
    df = pd.DataFrame()
    df_eeg = pd.DataFrame()
    
    for dataset in datasets:
        print(dataset)
        stage_files = os.listdir(stage_dir.replace('DATASET', dataset))
        eeg_files = os.listdir(eeg_dir.replace('DATASET', dataset.replace('_new', '')))
        available_files = np.intersect1d(stage_files, eeg_files)
        print(f"Dataset: {dataset}, Number of available files: {len(available_files)}")
        for file in tqdm(available_files):
            stage = np.load(stage_dir.replace('DATASET', dataset) + file)
            if stage['fs'] > 1/30:
                fs = round(stage['fs'] * 30)
                data = stage['data'][::fs]
            elif stage['fs'] == 1/30:
                data = stage['data']
            else:
                assert False 
            stage = process_stages(data)
            stage = np.array(stage)
            if sum(stage > 0) < 2 * 60 * 4 or 4 not in stage:
                continue
            
            eeg = np.load(eeg_dir.replace('DATASET', dataset.replace('_new', '')) + file)['data']
            stage_durations = calculate_stage_durations(stage)
            stage_latencies = calculate_stage_latencies(stage)
            awakenings = calculate_wakenings(stage)
            transitions = calculate_stage_transitions(stage)
            
            
            eeg_powers = calculate_band_powers(eeg)
            eeg_powers_by_stage = calculate_band_powers_by_stage(eeg, stage)

            first_cycle_idx = get_first_cycle_idx(file)
            if first_cycle_idx is None:
                continue
            sleep_ratio = calculate_sleep_ratio(eeg, stage, first_cycle_idx)
            
            # join all the dicts together 
            # merged = dict1 | dict2 | dict3 | dict4
            merged = stage_durations | stage_latencies | awakenings | transitions
            merged2 = eeg_powers | eeg_powers_by_stage | sleep_ratio
            merged['dataset'] = dataset
            merged2['dataset'] = dataset
            merged['filename'] = file
            merged2['filename'] = file
            df = pd.concat([df, pd.DataFrame(merged, index=[0])], ignore_index=True)
            df_eeg = pd.concat([df_eeg, pd.DataFrame(merged2, index=[0])], ignore_index=True)
    df.to_csv(CSV_DIR+'df_baseline.csv', index=False)
    df_eeg.to_csv(CSV_DIR+'df_baseline_eeg.csv', index=False)
    print('done')