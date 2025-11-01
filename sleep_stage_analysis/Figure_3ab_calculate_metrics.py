import numpy as np 
import pandas as pd 
import os 
from ipdb import set_trace as bp 
import copy 
from tqdm import tqdm 
import seaborn as sns
from scipy.stats import pearsonr
from scipy import stats
import dataframe_image as dfi

def calculate_rem_latency(sleep_stage):
    if 4 not in sleep_stage:
        return np.nan 
    return np.argwhere(sleep_stage == 4)[0][0] - np.argwhere(sleep_stage > 0)[0][0]

def calculate_sws_duration(sleep_stage):
    return np.sum(sleep_stage == 3) 

def calculate_rem_duration(sleep_stage):
    return np.sum(sleep_stage == 4) 

## this is sleep efficiency 
def calculate_sleep_continuity(sleep_stage):
    return np.sum(sleep_stage > 0) / (np.sum(sleep_stage > 0) + np.sum(sleep_stage == 0))


def process_stages(stages):
    stages = stages['data'][::int(30*stages['fs'])]
    stages[stages < 0] = 0
    stages[stages > 5] = 0
    stages = stages.astype(int)
    # mapping = np.array([0, 2, 2, 3, 3, 1, 0, 0, 0, 0], int)
    mapping = np.array([0, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0], np.int64)
    return mapping[stages]

df = pd.read_csv('../data/antidep_taxonomy_all_datasets_v6.csv')
df_loc = pd.read_csv('../data/inference_v6emb_3920_all.csv')
df_loc['filename'] = df_loc['filename'].apply(lambda x: x.split('/')[-1])
control_df = copy.deepcopy(df[df['taxonomy'] == 'C'])
antidep_df = copy.deepcopy(df[df['taxonomy'] != 'C'])

EEG_STAGE_DIR = "/data/netmit/wifall/ADetect/data"
STAGE_PREDICTION_DIR = "/data/netmit/wifall/chaoli/best_stage_predictions"
for idx, row in tqdm(control_df.iterrows()):
    # if idx % 10 != 0:
    #     continue 
    try:
        dataset = df_loc[df_loc['filename'] == row['filename']]['dataset'].values[0]
        if 'mros-visit1' in row['filename']:
            dataset = 'mros1_new'
        elif 'mros-visit2' in row['filename']:
            dataset = 'mros2_new'
        elif 'shhs1' in row['filename']:
            dataset = 'shhs1_new'
        elif 'shhs2' in row['filename']:
            dataset = 'shhs2_new'
        elif 'wsc' in row['filename']:
            dataset = 'wsc_new'
    except:
        pass
    if dataset in ['hchs', 'rf']:
        continue 
    filepath = f"{EEG_STAGE_DIR}/{dataset}/stage/{row['filename']}"
    try:
        sleep_stage = np.load(filepath)
    except:
        print('skipping', row['filename'])
        continue 
    sleep_stage = process_stages(sleep_stage)
    if np.sum(sleep_stage > 0) < 4 * 60 * 2:
        print('2skipping', row['filename'])
        continue 
    control_df.loc[idx, 'rem_latency_gt'] = calculate_rem_latency(sleep_stage)
    control_df.loc[idx, 'sws_duration_gt'] = calculate_sws_duration(sleep_stage)
    control_df.loc[idx, 'rem_duration_gt'] = calculate_rem_duration(sleep_stage)
    control_df.loc[idx, 'sleep_efficiency_gt'] = calculate_sleep_continuity(sleep_stage)
    
    pred_filepath = f"{STAGE_PREDICTION_DIR}/{dataset}/thorax/{row['filename']}"
    try:
        pred_sleep_stage = np.load(pred_filepath)['data'].flatten()
    except:
        print('3skipping', row['filename'])
        continue 
    control_df.loc[idx, 'dataset'] = dataset 
    control_df.loc[idx, 'rem_latency_pred'] = calculate_rem_latency(pred_sleep_stage)
    control_df.loc[idx, 'sws_duration_pred'] = calculate_sws_duration(pred_sleep_stage)
    control_df.loc[idx, 'rem_duration_pred'] = calculate_rem_duration(pred_sleep_stage)
    control_df.loc[idx, 'sleep_efficiency_pred'] = calculate_sleep_continuity(pred_sleep_stage)

for idx, row in antidep_df.iterrows():
    # if idx % 10 != 0:
    #     continue 
    try:
        dataset = df_loc[df_loc['filename'] == row['filename']]['dataset'].values[0]
        if 'mros-visit1' in row['filename']:
            dataset = 'mros1_new'
        elif 'mros-visit2' in row['filename']:
            dataset = 'mros2_new'
        elif 'shhs1' in row['filename']:
            dataset = 'shhs1_new'
        elif 'shhs2' in row['filename']:
            dataset = 'shhs2_new'
        elif 'wsc' in row['filename']:
            dataset = 'wsc_new'
    except:
        pass
    if dataset in ['hchs', 'rf']:
        continue 

    filepath = f"{EEG_STAGE_DIR}/{dataset}/stage/{row['filename']}"
    try:
        sleep_stage = np.load(filepath)
    except:
        print('skipping', row['filename'])
        continue 
    sleep_stage = process_stages(sleep_stage)
    if np.sum(sleep_stage > 0) < 4 * 60 * 2:
        print('skipping', row['filename'])
        continue 
    antidep_df.loc[idx, 'rem_latency_gt'] = calculate_rem_latency(sleep_stage)
    antidep_df.loc[idx, 'sws_duration_gt'] = calculate_sws_duration(sleep_stage)
    antidep_df.loc[idx, 'rem_duration_gt'] = calculate_rem_duration(sleep_stage)
    antidep_df.loc[idx, 'sleep_efficiency_gt'] = calculate_sleep_continuity(sleep_stage)
    
    pred_filepath = f"{STAGE_PREDICTION_DIR}/{dataset}/thorax/{row['filename']}"
    try:
        pred_sleep_stage = np.load(pred_filepath)['data'].flatten()
    except:
        print('skipping', row['filename'])
        continue 
    antidep_df.loc[idx, 'dataset'] = dataset 
    antidep_df.loc[idx, 'rem_latency_pred'] = calculate_rem_latency(pred_sleep_stage)
    antidep_df.loc[idx, 'sws_duration_pred'] = calculate_sws_duration(pred_sleep_stage)
    antidep_df.loc[idx, 'rem_duration_pred'] = calculate_rem_duration(pred_sleep_stage)
    antidep_df.loc[idx, 'sleep_efficiency_pred'] = calculate_sleep_continuity(pred_sleep_stage)

control_df['label'] = 0
antidep_df['label'] = 1
pd.concat([control_df[['filename', 'rem_latency_gt', 'rem_latency_pred', 'label']], antidep_df[['filename', 'rem_latency_gt', 'rem_latency_pred', 'label']]]).to_csv('figure_draft_v16_rem_latency.csv', index=False)

