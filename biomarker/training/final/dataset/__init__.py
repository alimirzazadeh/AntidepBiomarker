from __future__ import print_function, division
import os
from typing import List
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset 
from tqdm import tqdm 
from ipdb import set_trace as bp 
from sklearn.model_selection import KFold
import sys 
from copy import deepcopy
from collections import defaultdict 
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import json
from datetime import datetime

sys.path.append(os.path.abspath('..'))



def transform_four_stages_three(stages):
    stages = stages.flatten()
    output = np.zeros(len(stages))
    output[stages == 4] = 1
    output[stages == 1] = 2
    output[stages == 2] = 2
    output[stages == 3] = 3
    return output 
    
    
def calculate_features_from_stages(stages):
    ## need rem sleep latency, 
    ## 
    ## waso , 
    stages = np.array(stages)

    
    tst = sum(stages > 0) / 2
    deep_duration = sum(stages == 3) / 2
    deep_percent = deep_duration / tst 
    rem_duration = sum(stages == 1) / 2
    rem_percent = rem_duration / tst 
    try:
        first_rem = np.argwhere(stages == 1)[0][0]
        first_sleep = np.argwhere(stages > 0)[0][0]
        last_sleep = np.argwhere(stages > 0)[-1][0]
        rem_latency = (first_rem - first_sleep) / 2
        
    except:
        first_sleep = np.argwhere(stages > 0)[0][0]
        last_sleep = np.argwhere(stages > 0)[-1][0]
        
        rem_latency = tst 
    waso = sum(stages[first_sleep:last_sleep] == 0) / 2.0
    return np.array([rem_latency, rem_duration, 100*rem_percent, deep_duration, 100*deep_percent, waso, tst, first_sleep / 2], dtype=float)

def process_stages(stages):
    stages[stages < 0] = 0
    stages[stages > 5] = 0
    stages = stages.astype(int)
    mapping = np.array([0, 2, 2, 3, 3, 1, 0, 0, 0, 0], int)
    # mapping = np.array([0, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0], np.int64)
    return mapping[stages]

def get_nth_element(generator, n):
    for _ in range(n):
        next(generator)  # Discard the elements up to the n'th element
    return next(generator) 

def load_json_with_datetime(path):
    def parse(item):
        if isinstance(item, str):
            try:
                return datetime.fromisoformat(item)
            except ValueError:
                return item
        elif isinstance(item, list):
            return [parse(subitem) for subitem in item]
        else:
            return item

    with open(path, 'r') as f:
        raw = json.load(f)
    return {k: parse(v) for k, v in raw.items()}

def medication_antidep(item):
    item = item.replace('_V2', '').replace('_V3', '')
    return _medication_data.get(item, 0)

def dosage_antidep(item):
    item = item.replace('_V2', '').replace('_V3', '')
    return _dosage_data.get(item, 0.0)

def parse_dataset_name(filepath):
    if 'shhs' in filepath:
        return 'shhs'
    elif 'wsc' in filepath:
        return 'wsc'
    elif 'cfs' in filepath:
        return 'cfs'
    elif 'hchs' in filepath:
        return 'hchs'
    elif 'mros' in filepath:
        return 'mros'
    elif 'rf' in filepath:
        return 'rf'
    elif 'stages' in filepath: #filepath.startswith(tuple(['BOGN', 'GSBB', 'GSDV', 'GSLH', 'GSSA', 'GSSW', 'MAYO', 'MSMI', 'MSNF', 'MSQW', 'MSTH', 'MSTR', 'STLK', 'STNF'])):
        return 'stages'
    else:
        raise NotImplementedError


_medication_data = load_json_with_datetime('../../../../data/medication_antidep.json')
_dosage_data = load_json_with_datetime('../../../../data/dosage_antidep.json')



class DataCollector_V2():
    def __init__(self, mode: str, datasets: List[str], fold: int) -> None:
        self.datasets = [] 
        
        for dataset in datasets:
            assert dataset in ['wsc', 'shhs1', 'shhs2', 'cfs', 'hchs', 'mros1', 'mros2','rf','stages']

            if dataset != 'rf':
                dataset += "_new"
            
            self.datasets.append(dataset)
            
        assert mode in ['thorax','mage','stage','abdominal','mage_gt']
        
        
        self.mode = mode
        self.all_files = {}
        self.dataset_filepaths = {} 
        
        for dataset in self.datasets:
            if mode == 'mage':
                print(dataset)
                if dataset == 'rf':
                    # rf_path = '/data/netmit/wifall/ali/RF_MAGE_ANTIDEP/mage/rf_c4_m1/'
                    """The non 18 hr, fixed on sleep onset mage embeddings """
                    rf_path = f"/data/netmit/sleep_lab/rf/mage/cv_{fold}/"
                    
                    self.dataset_filepaths['rf'] = rf_path
                    bad_nights = np.load('/data/netmit/wifall/ali/RF_MAGE_ANTIDEP/bad_nights_v6.npz')
                    self.all_files['rf'] = [pid + '/' + file for pid in os.listdir(rf_path) for file in os.listdir(os.path.join(rf_path,pid))]
                    self.all_files['rf'] = [file for file in self.all_files['rf'] if file.split('/')[0] in bad_nights and file.split('/')[1] not in bad_nights[file.split('/')[0]]] 
                else:
                    self.dataset_filepaths[dataset.split('_')[0]] = '/data/netmit/sleep_lab/filtered/MAGE/'+dataset+f'/mage/cv_{fold}/'
                    self.all_files[dataset.split('_')[0]] = os.listdir('/data/netmit/sleep_lab/filtered/MAGE/'+dataset+f'/mage/cv_{fold}/')
                    
                    try:
                        print('cleaning...')
                        arr = np.load(f"/data/netmit/wifall/ali/RF_MAGE_ANTIDEP/bad_nights_{dataset.replace('_new','')}.npy")
                        print(len(self.all_files[dataset.split('_')[0]]))
                        self.all_files[dataset.split('_')[0]] = [item for item in self.all_files[dataset.split('_')[0]] if item not in arr]
                        print(len(self.all_files[dataset.split('_')[0]]))
                    except:
                        print('not cleaning ', dataset)
            elif mode == 'mage_gt':
                raise NotImplementedError 
            else:
                raise NotImplementedError 
    def get_filepath(self, dataset: str) -> str:
        return self.dataset_filepaths[dataset]

class DataCollector():
    def __init__(self, mode: str, datasets: List[str]) -> None:
        self.datasets = [] 
        
        for dataset in datasets:
            assert dataset in ['wsc', 'shhs1', 'shhs2', 'cfs', 'hchs', 'mros1', 'mros2','rf','stages']
            if dataset in ['wsc','hchs']:
                dataset += "_new"
            if dataset == 'rf':
                dataset += '_c4_m1'
            
            self.datasets.append(dataset)
            
        assert mode in ['thorax','mage','stage','abdominal','mage_gt']
        
        
        self.mode = mode
        self.all_files = {}
        self.dataset_filepaths = {} 
        
        for dataset in self.datasets:
            if mode == 'mage':
                if dataset == 'rf_c4_m1':
                    # rf_path = '/data/netmit/wifall/ali/RF_MAGE_ANTIDEP/mage/rf_c4_m1/'
                    """The non 18 hr, fixed on sleep onset mage embeddings """
                    rf_path = '/data/netmit/sleep_lab/ali_3/rf_trimmed_c4_m1/'
                    
                    self.dataset_filepaths['rf'] = rf_path
                    bad_nights = np.load('/data/netmit/wifall/ali/RF_MAGE_ANTIDEP/bad_nights_v6.npz')
                    self.all_files['rf'] = [pid + '/' + file for pid in os.listdir(rf_path) for file in os.listdir(os.path.join(rf_path,pid))]
                    self.all_files['rf'] = [file for file in self.all_files['rf'] if file.split('/')[0] in bad_nights and file.split('/')[1] not in bad_nights[file.split('/')[0]]]
                    
                elif dataset == 'hchs_new':
                    self.dataset_filepaths['hchs'] = '/data/netmit/wifall/ali/RF_MAGE_ANTIDEP/mage/'+dataset+'/airflow_c4_m1/'
                    self.all_files[dataset.split('_')[0]] = os.listdir('/data/netmit/wifall/ali/RF_MAGE_ANTIDEP/mage/'+dataset+'/airflow_c4_m1/')
                    
                    arr = np.load(f"/data/netmit/wifall/ali/RF_MAGE_ANTIDEP/bad_nights_{dataset.split('_')[0]}.npy")
                    print(len(self.all_files[dataset.split('_')[0]]))
                    self.all_files[dataset.split('_')[0]] = [item for item in self.all_files[dataset.split('_')[0]] if item not in arr]
                else:
                    self.dataset_filepaths[dataset.split('_')[0]] = '/data/netmit/wifall/ali/RF_MAGE_ANTIDEP/mage/'+dataset+'/abdominal_c4_m1/'
                    self.all_files[dataset.split('_')[0]] = os.listdir('/data/netmit/wifall/ali/RF_MAGE_ANTIDEP/mage/'+dataset+'/abdominal_c4_m1/')
                    
                    try:
                        print('cleaning...')
                        arr = np.load(f'/data/netmit/wifall/ali/RF_MAGE_ANTIDEP/bad_nights_{dataset}.npy')
                        print(len(self.all_files[dataset.split('_')[0]]))
                        self.all_files[dataset.split('_')[0]] = [item for item in self.all_files[dataset.split('_')[0]] if item not in arr]
                        print(len(self.all_files[dataset.split('_')[0]]))
                    except:
                        print('not cleaning ', dataset)
            elif mode == 'mage_gt':
                if dataset == 'rf_c4_m1':
                    assert False 
                elif dataset == 'hchs_new':
                    assert False 
                else:
                    self.dataset_filepaths[dataset.split('_')[0]] = '/data/netmit/wifall/ADetect/data/'+dataset+'/c4_m1_multitaper/'
                    self.all_files[dataset.split('_')[0]] = os.listdir('/data/netmit/wifall/ADetect/data/'+dataset+'/c4_m1_multitaper/')
                    
                    try:
                        print('cleaning...')
                        arr = np.load(f'/data/netmit/wifall/ali/RF_MAGE_ANTIDEP/bad_nights_{dataset}.npy')
                        print(len(self.all_files[dataset.split('_')[0]]))
                        self.all_files[dataset.split('_')[0]] = [item for item in self.all_files[dataset.split('_')[0]] if item not in arr]
                        print(len(self.all_files[dataset.split('_')[0]]))
                    except:
                        print('not cleaning ', dataset)
            else:
                raise NotImplementedError 
    def get_filepath(self, dataset: str) -> str:
        return self.dataset_filepaths[dataset]

class LabelHunter():
    def __init__(self, label:str) -> None:
        SUPPORTED_LABELS = ['antidep', 'benzos','depression', 'bupropion', 'mirtazapine', 'antidep_6_class', 'antidep_14_class', 'antidep_taxonomy_5', 'depression_classification', 'antidep_response','depression_noantidep','depression_regression','depression_regression_2']
        
        self.label = label 
        assert self.label in SUPPORTED_LABELS

        
        
        self.label_translation = {
            'wsc':   {'depression_regression_2':['depression_regression_2'], 'depression_regression':['depression_regression'], 'depression_noantidep':[], 'antidep_response': [], 'depression_classification': [], 'antidep_6_class':['antidep_6_class'], 'antidep_14_class':['antidep_14_class'], 'antidep_taxonomy_5': ['taxonomy'], 'bupropion':['only_bupropion'], 'mirtazapine':['only_mirtazapine'], 'antidep':['depression_med','dep_tca_med','dep_ssri_med'], 'benzos':['dr863'], 'depression':['zung_index']},
            'shhs1': {'depression_regression_2':['depression_regression_2'], 'depression_regression':['depression_regression'], 'depression_noantidep':['depression_noantidep'],'antidep_response': ['antidep_response'], 'antidep':['tca1', 'ntca1'], 'benzos':['benzod1'], 'depression':[], 'antidep_taxonomy_5': ['taxonomy'], 'depression_classification':['sf36_mental_sum']},
            'shhs2': {'depression_regression_2':['depression_regression_2'], 'depression_regression':['depression_regression'], 'depression_noantidep':['depression_noantidep'],'antidep_response': ['antidep_response'], 'antidep':['tca2', 'ntca2'], 'benzos':['benzod2'], 'depression':[], 'antidep_taxonomy_5': ['taxonomy'], 'depression_classification':['sf36_mental_sum']},
            'mros1': {'depression_regression_2':['depression_regression_2'], 'depression_regression':['depression_regression'], 'depression_noantidep':['depression_noantidep'],'antidep_response': ['antidep_response'], 'antidep_6_class':['antidep_6_class'], 'antidep_14_class':['antidep_14_class'],'antidep_taxonomy_5': ['taxonomy'], 'bupropion':['only_bupropion'], 'mirtazapine':['only_mirtazapine'], 'antidep':['m1adepr'], 'benzos':['m1benzo'], 'depression':['mhdepr'], 'depression_classification': ['depression_gds_cutoff']},
            'mros2': {'depression_regression_2':['depression_regression_2'], 'depression_regression':['depression_regression'], 'depression_noantidep':['depression_noantidep'],'antidep_response': ['antidep_response'], 'antidep_6_class':['antidep_6_class'], 'antidep_14_class':['antidep_14_class'],'antidep_taxonomy_5': ['taxonomy'], 'bupropion':['only_bupropion'], 'mirtazapine':['only_mirtazapine'], 'antidep':['m1adepr'], 'benzos':['m1benzo'], 'depression':['mhdepr'], 'depression_classification': ['depression_gds_cutoff']},
            'cfs':   {'depression_regression_2':['depression_regression_2'], 'depression_regression':['depression_regression'], 'depression_noantidep':['depression_noantidep'],'antidep_response': ['antidep_response'], 'antidep':['antidep'], 'benzos':['psychmeds'], 'depression':['depdiag'], 'antidep_taxonomy_5': ['taxonomy'], 'depression_classification': ['sf36_mental_sum']},
            'hchs':  {'depression_regression_2':['depression_regression_2'], 'depression_regression':[], 'depression_noantidep':[],'antidep_response': [], 'depression_classification': [], 'antidep':['med_antidepress', 'med_antidepre_tricyc', 'med_antidepre_ssri'], 'antidep_taxonomy_5': ['taxonomy'], 'benzos':['med_antipsycho'], 'depression':['cesd10']},
            'rf':    {'depression_regression_2':['depression_regression_2'], 'depression_regression':[], 'depression_noantidep':[],'antidep_response': [], 'depression_classification': [] , 'antidep_6_class':['antidep_6_class'], 'antidep_14_class':['antidep_14_class'],'antidep_taxonomy_5': ['taxonomy'], 'bupropion':['only_bupropion'], 'mirtazapine':['only_mirtazapine'], 'antidep':['ESCITALOPRAM', 'TRAZODONE', 'CITALOPRAM', 'VENLAFAXINE', 'FLUOXETINE', 'SERTRALINE', 'PAROXETINE' , 'DESVENLAFAXINE', 'IMIPRAMINE', 'DULOXETINE' , 'VORTIOXETINE', 'MIRTAZAPINE' , 'BUPROPRION' , 'NORTRYPTILINE'], 'benzos':[], 'depression':[]},
            'stages': {'depression_regression_2':['depression_regression_2'], 'depression_regression':['depression_regression'], 'depression_noantidep':['depression_noantidep'],'antidep_response': ['antidep_response'], 'antidep_taxonomy_5': ['taxonomy'], 'antidep': ['antidep'], 'depression_classification': ['depression_phq_cutoff']}
        }
        
        self.datasets = {} 
        
        self.datasets['wsc'] = pd.read_csv("/data/netmit/sleep_lab/ali_csv/wsc-dataset-0.7.0.csv", encoding='mac_roman')
        self.datasets['wsc']['filename'] = self.datasets['wsc'].apply(lambda x: 'wsc-visit' + str(x['wsc_vst']) + '-' + str(x['wsc_id']) + '-nsrr.npz',axis=1)
        self.datasets['wsc']['mit_gender'] = self.datasets['wsc']['sex'].apply(lambda x: 1 if x == 'M' else 2)
        self.datasets['wsc']['mit_age'] = self.datasets['wsc']['age'].apply(lambda x: float(x))
        self.datasets['wsc']['mit_race'] = self.datasets['wsc']['race'].apply(lambda x: [3,2,4,5,0,1][int(x)])
        self.datasets['wsc']['mit_bmi'] = self.datasets['wsc']['bmi'].apply(lambda x: float(x))
        self.datasets['wsc']['depression_regression'] = self.normalize(self.datasets['wsc']['zung_index'].apply(lambda x: x if x > 25 else np.nan))
        self.datasets['wsc']['antidep'] = (self.datasets['wsc']['depression_med'].astype(bool) + self.datasets['wsc']['dep_tca_med'].astype(bool) + self.datasets['wsc']['dep_ssri_med'].astype(bool)).astype(float)
        self.datasets['wsc']['depression_regression_2'] = self.datasets['wsc']['depression_regression'] * self.datasets['wsc']['antidep'].apply(lambda x: 1 if x == 0 else np.nan)
        ## repeat these rows with _fold0, _fold1, etc.
        #if multicopy_wsc:
        #    self.datasets['wsc'] = pd.concat(
        #        [self.datasets['wsc'].assign(filename=self.datasets['wsc']['filename'].str.replace('.npz', f'_fold{i}.npz'),
        #                                    fold=i) 
        #        for i in range(4)],
        #        ignore_index=True
        #    )

        self.datasets['shhs1'] = pd.read_csv("/data/netmit/wifall/ADetect/data/csv/shhs1-dataset-augmented.csv", encoding='mac_roman')
        self.datasets['shhs1']['filename'] = self.datasets['shhs1'].apply(lambda x: 'shhs1-' + str(x['nsrrid']) + '.npz',axis=1)
        self.datasets['shhs1']['sf36_mental_sum'] = (self.datasets['shhs1']['rawre_s1'] + self.datasets['shhs1']['rawmh_s1'] + self.datasets['shhs1']['rawvt_s1'] + self.datasets['shhs1']['rawsf_s1']) 
        self.datasets['shhs1']['depression_regression'] = self.normalize(-1 * self.datasets['shhs1']['sf36_mental_sum'].apply(lambda x: x if x != 70 else np.nan))
        self.datasets['shhs1']['sf36_mental_sum'] = self.datasets['shhs1']['sf36_mental_sum'].apply(lambda x: 1 if x <= 40 else 0 if (x >= 57 and x < 70) else np.nan)
        self.datasets['shhs1']['sf36_physical_sum3'] = (self.datasets['shhs1']['rawpf_s1'] + self.datasets['shhs1']['rawrp_s1'] + self.datasets['shhs1']['rawbp_s1'])
        self.datasets['shhs1']['antidep'] = (self.datasets['shhs1']['tca1'].astype(bool) + self.datasets['shhs1']['ntca1'].astype(bool)).astype(float)
        self.datasets['shhs1']['antidep_response'] = 1 - self.datasets['shhs1']['sf36_mental_sum'].astype(float) * self.datasets['shhs1']['antidep'].apply(lambda x: 1 if x == 1 else np.nan)
        self.datasets['shhs1']['depression_noantidep'] = self.datasets['shhs1']['sf36_mental_sum'].astype(float) * self.datasets['shhs1']['antidep'].apply(lambda x: 1 if x == 0 else np.nan)
        self.datasets['shhs1']['depression_regression_2'] = self.datasets['shhs1']['depression_regression'] * self.datasets['shhs1']['antidep'].apply(lambda x: 1 if x == 0 else np.nan)
        
        self.datasets['shhs2'] = pd.read_csv("/data/netmit/wifall/ADetect/data/csv/shhs2-dataset-augmented.csv", encoding='mac_roman')
        self.datasets['shhs2']['filename'] = self.datasets['shhs2'].apply(lambda x: 'shhs2-' + str(x['nsrrid']) + '.npz',axis=1)
        self.datasets['shhs2']['sf36_mental_sum'] = (self.datasets['shhs2']['rawre_s2'] + self.datasets['shhs2']['rawmh_s2'] + self.datasets['shhs2']['rawvt_s2'] + self.datasets['shhs2']['rawsf_s2'])
        self.datasets['shhs2']['depression_regression'] = self.normalize(-1 * self.datasets['shhs2']['sf36_mental_sum'].apply(lambda x: x if x != 70 else np.nan))
        self.datasets['shhs2']['sf36_mental_sum'] = self.datasets['shhs2']['sf36_mental_sum'].apply(lambda x: 1 if x <= 40 else 0 if (x >= 57 and x < 70) else np.nan)
        self.datasets['shhs2']['sf36_physical_sum3'] = (self.datasets['shhs2']['rawpf_s2'] + self.datasets['shhs2']['rawrp_s2'] + self.datasets['shhs2']['rawbp_s2'])
        # self.datasets['shhs2']['antidep_response'] = self.datasets['shhs2']['sf36_mental_sum'].astype(bool) * (self.datasets['shhs2']['tca2'] + self.datasets['shhs2']['ntca2'])
        self.datasets['shhs2']['antidep'] = (self.datasets['shhs2']['tca2'].astype(bool) + self.datasets['shhs2']['ntca2'].astype(bool)).astype(float)
        self.datasets['shhs2']['antidep_response'] = 1 - self.datasets['shhs2']['sf36_mental_sum'].astype(float) * self.datasets['shhs2']['antidep'].apply(lambda x: 1 if x == 1 else np.nan)
        self.datasets['shhs2']['depression_noantidep'] = self.datasets['shhs2']['sf36_mental_sum'].astype(float) * self.datasets['shhs2']['antidep'].apply(lambda x: 1 if x == 0 else np.nan)
        self.datasets['shhs2']['depression_regression_2'] = self.datasets['shhs2']['depression_regression'] * self.datasets['shhs2']['antidep'].apply(lambda x: 1 if x == 0 else np.nan)

        self.datasets['cfs'] = pd.read_csv("/data/netmit/wifall/ADetect/data/csv/cfs-dataset-augmented.csv", encoding='mac_roman')
        self.datasets['cfs']['filename'] = self.datasets['cfs'].apply(lambda x: "cfs-visit5-" + str(x['nsrrid']) + '.npz',axis=1)
        self.datasets['cfs'] = self.datasets['cfs'][self.datasets['cfs']['mit_age'] >= 20]
        self.datasets['cfs']['sf36_mental_sum'] = (self.datasets['cfs']['sf36_rawre'] + self.datasets['cfs']['sf36_rawvt'] + self.datasets['cfs']['sf36_rawmh'] + self.datasets['cfs']['sf36_rawsf'])
        self.datasets['cfs']['depression_regression'] = self.normalize(-1 * self.datasets['cfs']['sf36_mental_sum'].apply(lambda x: x if x != 70 else np.nan))
        self.datasets['cfs']['sf36_mental_sum'] = self.datasets['cfs']['sf36_mental_sum'].apply(lambda x: 1 if x <= 40 else 0 if (x >= 57 and x < 70) else np.nan)
        self.datasets['cfs']['sf36_physical_sum3'] = (self.datasets['cfs']['sf36_rawpf'] + self.datasets['cfs']['sf36_rawrp'] + self.datasets['cfs']['sf36_rawbp'])
        self.datasets['cfs']['antidep'] = (self.datasets['cfs']['antidepr'].astype(bool) + self.datasets['cfs']['othantid'].astype(bool) + self.datasets['cfs']['prozac'].astype(bool) + self.datasets['cfs']['zoloft'].astype(bool) + self.datasets['cfs']['paxil'].astype(bool) + self.datasets['cfs']['celexa'].astype(bool)).astype(float)
        # self.datasets['cfs']['antidep_response'] = self.datasets['cfs']['sf36_mental_sum'].astype(bool) * self.datasets['cfs']['antidepr']
        self.datasets['cfs']['antidep_response'] = 1 - self.datasets['cfs']['sf36_mental_sum'].astype(float) * self.datasets['cfs']['antidep'].apply(lambda x: 1 if x == 1 else np.nan)
        self.datasets['cfs']['depression_noantidep'] = self.datasets['cfs']['sf36_mental_sum'].astype(float) * self.datasets['cfs']['antidep'].apply(lambda x: 1 if x == 0 else np.nan)
        self.datasets['cfs']['depression_regression_2'] = self.datasets['cfs']['depression_regression'] * self.datasets['cfs']['antidep'].apply(lambda x: 1 if x == 0 else np.nan)
        
        self.datasets['hchs'] = pd.read_csv("/data/netmit/wifall/ADetect/data/csv/hchs-dataset-augmented.csv", encoding='mac_roman')
        self.datasets['hchs']['filename'] = self.datasets['hchs'].apply(lambda x: f"{str(x['uid'])}.npz",axis=1)
        self.datasets['hchs']['mit_race'] = self.datasets['hchs']['race'].apply(lambda x: 0 if np.isnan(x) else [0,5,3,0,2,1,4,4,0,4][int(x)])
        self.datasets['hchs']['mit_bmi'] = self.datasets['hchs']['bmi'].apply(lambda x: float(x))
        self.datasets['hchs'] = self.datasets['hchs'][self.datasets['hchs']['mit_age'] >= 20]
        

        self.datasets['mros1'] = pd.read_csv("/data/netmit/sleep_lab/ali_csv/mros1-dataset-augmented-live-v2.csv", encoding='mac_roman')
        self.datasets['mros1']['filename'] = self.datasets['mros1'].apply(lambda x: f"mros-visit1-aa{str(x['nsrrid'][2:])}.npz",axis=1)
        self.datasets['mros1']['mit_race'] = self.datasets['mros1']['girace'].apply(lambda x: [0,1,2,3,0,5,0,0][int(x)])
        self.datasets['mros1']['depression_regression'] = self.normalize(self.datasets['mros1']['DPGDS15'].apply(lambda x: x if x > 0 else np.nan))
        self.datasets['mros1']['depression_gds_cutoff'] = self.datasets['mros1']['DPGDS15'].apply(lambda x: 1 if x >= 7 else 0 if (x <= 3 and x >= 0) else np.nan)
        self.datasets['mros1']['antidep_response'] = 1 - self.datasets['mros1']['depression_gds_cutoff'].astype(float) * self.datasets['mros1']['m1adepr'].apply(lambda x: 1 if x == 1 else np.nan)
        self.datasets['mros1']['depression_noantidep'] = self.datasets['mros1']['depression_gds_cutoff'].astype(float) * self.datasets['mros1']['m1adepr'].apply(lambda x: 1 if x == 0 else np.nan)
        self.datasets['mros1']['depression_regression_2'] = self.datasets['mros1']['depression_regression'] * self.datasets['mros1']['m1adepr'].apply(lambda x: 1 if x == 0 else np.nan)
        
        self.datasets['mros2'] = pd.read_csv("/data/netmit/sleep_lab/ali_csv/mros2-dataset-augmented-live-v2.csv", encoding='mac_roman')
        self.datasets['mros2']['filename'] = self.datasets['mros2'].apply(lambda x: f"mros-visit2-aa{str(x['nsrrid'][2:])}.npz",axis=1)
        self.datasets['mros2']['mit_race'] = self.datasets['mros2']['girace'].apply(lambda x: [0,1,2,3,0,5,0,0][int(x)])
        self.datasets['mros2']['depression_regression'] = self.normalize(self.datasets['mros2']['DPGDS15'].apply(lambda x: x if x > 0 else np.nan))
        self.datasets['mros2']['depression_gds_cutoff'] = self.datasets['mros2']['DPGDS15'].apply(lambda x: 1 if x >= 7 else 0 if (x <= 3 and x >= 0) else np.nan) # changes from < 5 > 0
        self.datasets['mros2']['antidep_response'] = 1 - self.datasets['mros2']['depression_gds_cutoff'].astype(float) * self.datasets['mros2']['m1adepr'].apply(lambda x: 1 if x == 1 else np.nan)
        self.datasets['mros2']['depression_noantidep'] = self.datasets['mros2']['depression_gds_cutoff'].astype(float) * self.datasets['mros2']['m1adepr'].apply(lambda x: 1 if x == 0 else np.nan)
        self.datasets['mros2']['depression_regression_2'] = self.datasets['mros2']['depression_regression'] * self.datasets['mros2']['m1adepr'].apply(lambda x: 1 if x == 0 else np.nan)

        self.datasets['rf'] = pd.read_csv('/data/netmit/sleep_lab/ali_csv/rf-dataset-augmented_v2.csv')
        self.datasets['rf']['file'] = deepcopy(self.datasets['rf']['filename'])
        self.datasets['rf']['filename'] = self.datasets['rf'].apply(lambda x: x['pid'] + "/" + x['file'],axis=1)
        self.datasets['rf']['mit_bmi'] = np.nan
        
        self.datasets['stages'] = pd.read_csv("/data/netmit/sleep_lab/ali_csv/stages-dataset-augmented-taxonomy.csv", encoding='mac_roman')
        self.datasets['stages']['filename'] = self.datasets['stages']['subject_code'].apply(lambda x: x + '.npz')
        self.datasets['stages']['mit_race'] = pd.NA
        self.datasets['stages']['mit_bmi'] = pd.NA 
        self.datasets['stages']['mit_age'] = pd.NA
        self.datasets['stages']['antidep'] = self.datasets['stages']['taxonomy'].apply(lambda x: x != 'C')
        self.datasets['stages']['depression_regression'] = self.normalize(self.datasets['stages']['phq_1000'].apply(lambda x: x if x > 0 else np.nan))
        self.datasets['stages']['depression_phq_cutoff'] = self.datasets['stages']['phq_1000'].apply(lambda x: 1 if x >= 9 else 0 if (x <= 5 and x > 0) else np.nan)
        self.datasets['stages']['antidep_response'] = 1 - self.datasets['stages']['depression_phq_cutoff'].astype(float) * self.datasets['stages']['antidep'].apply(lambda x: 1 if x == 1 else np.nan)
        self.datasets['stages']['depression_noantidep'] = self.datasets['stages']['depression_phq_cutoff'].astype(float) * self.datasets['stages']['antidep'].apply(lambda x: 1 if x == 0 else np.nan)
        self.datasets['stages']['depression_regression_2'] = self.datasets['stages']['depression_regression'] * self.datasets['stages']['antidep'].apply(lambda x: 1 if x == 0 else np.nan)
        
        # self.combined_datasets = pd.concat([self.datasets['wsc'], self.datasets['shhs1'], self.datasets['shhs2'], self.datasets['cfs'], self.datasets['hchs'], self.datasets['mros1'], self.datasets['mros2'], self.datasets['rf'], self.datasets['stages']])
        # 
        # self.combined_datasets.reset_index(drop=True, inplace=True)
        # self.combined_datasets.set_index('filename', inplace=True)
        
        
        if 'taxonomy' in self.label:
            self.taxonomy = pd.read_csv('/data/netmit/sleep_lab/ali_csv/antidep_taxonomy_all_datasets_v2.csv')
            self.taxonomy = self.taxonomy[['filename','taxonomy']]
            # MEDICATION_MAPPING = {0: 'TCA', 1: 'Bupropion', 2: 'Mirtazapine', 3: 'SNRI', 4: 'SSRI'}
            def taxonomy_to_5_class(item):
                items = item.split(',')
                output = [0,0,0,0,0]
                if 'C' in item:
                    return output 
                
                for item in items:
                    if item.startswith('T') and output[0] == 0:
                        output[0] += 1
                    if item.startswith('NB') and output[1] == 0:
                        output[1] += 1
                    if item.startswith('NM') and output[2] == 0:
                        output[2] += 1
                    if item.startswith('NN') and output[3] == 0:
                        output[3] += 1
                    if item.startswith('NS') and output[4] == 0:
                        output[4] += 1
                return output
                
            self.taxonomy['taxonomy'] = self.taxonomy['taxonomy'].apply(lambda x: str(taxonomy_to_5_class(x)))
            len_before = self.datasets['wsc'].shape[0]
            self.datasets['wsc'] = self.datasets['wsc'].merge(self.taxonomy, on='filename', how='inner')
            len_after = self.datasets['wsc'].shape[0]
            print(f'wsc: {len_before} -> {len_after}')
            len_before = self.datasets['shhs1'].shape[0]
            self.datasets['shhs1'] = self.datasets['shhs1'].merge(self.taxonomy, on='filename', how='inner')
            len_after = self.datasets['shhs1'].shape[0]
            print(f'shhs1: {len_before} -> {len_after}')
            len_before = self.datasets['shhs2'].shape[0]
            self.datasets['shhs2'] = self.datasets['shhs2'].merge(self.taxonomy, on='filename', how='inner')
            len_after = self.datasets['shhs2'].shape[0]
            print(f'shhs2: {len_before} -> {len_after}')
            len_before = self.datasets['cfs'].shape[0]
            self.datasets['cfs'] = self.datasets['cfs'].merge(self.taxonomy, on='filename', how='inner')
            len_after = self.datasets['cfs'].shape[0]
            print(f'cfs: {len_before} -> {len_after}')
            len_before = self.datasets['hchs'].shape[0]
            self.datasets['hchs'] = self.datasets['hchs'].merge(self.taxonomy, on='filename', how='inner')
            len_after = self.datasets['hchs'].shape[0]
            print(f'hchs: {len_before} -> {len_after}')
            len_before = self.datasets['mros1'].shape[0]
            self.datasets['mros1'] = self.datasets['mros1'].merge(self.taxonomy, on='filename', how='inner')
            len_after = self.datasets['mros1'].shape[0]
            print(f'mros1: {len_before} -> {len_after}')
            len_before = self.datasets['mros2'].shape[0]
            self.datasets['mros2'] = self.datasets['mros2'].merge(self.taxonomy, on='filename', how='inner')
            len_after = self.datasets['mros2'].shape[0]
            print(f'mros2: {len_before} -> {len_after}')

            len_before = self.datasets['stages'].shape[0]
            self.datasets['stages'] = self.datasets['stages'].merge(self.taxonomy, on='filename', how='inner')
            len_after = self.datasets['stages'].shape[0]
            print(f'stages: {len_before} -> {len_after}')
            
            len_before = self.datasets['rf'].shape[0]
            
            rf_taxonomy = self.taxonomy.copy()
            rf_taxonomy['file'] = rf_taxonomy['filename']
            rf_taxonomy.drop('filename',axis=1,inplace=True)
            
            self.datasets['rf'] = self.datasets['rf'].merge(rf_taxonomy, on='file', how='inner')
            len_after = self.datasets['rf'].shape[0]
            print(f'rf: {len_before} -> {len_after}')
        
        self.datasets['wsc'].set_index('filename', inplace=True)
        self.datasets['shhs1'].set_index('filename', inplace=True)
        self.datasets['shhs2'].set_index('filename', inplace=True)
        self.datasets['cfs'].set_index('filename', inplace=True)
        self.datasets['hchs'].set_index('filename', inplace=True)
        self.datasets['mros1'].set_index('filename', inplace=True)
        self.datasets['mros2'].set_index('filename', inplace=True)
        self.datasets['rf'].set_index('filename', inplace=True)
        self.datasets['stages'].set_index('filename', inplace=True)
        
        # self.datasets['mros1'] = pd.read_csv("/data/netmit/wifall/ADetect/data/csv/hchs-dataset-augmented.csv", encoding='mac_roman')
        
        ## still need mros, hchs, rf_c4_m1
        
    def normalize(self, series) -> pd.Series:
        # if series.min() < 0:
        #     series = series - series.min()
        # series = np.log(series + 0.1)
        return (series - series.mean(skipna=True)) / series.std(skipna=True)
        
    def get(self, filename: str) -> bool:
        if 'shhs1' in filename:
            return self.datasets['shhs1'].loc[filename,self.label_translation['shhs1'][self.label]].any()
        elif 'shhs2' in filename:
            return self.datasets['shhs2'].loc[filename,self.label_translation['shhs2'][self.label]].any()
        elif 'wsc' in filename: 
            return self.datasets['wsc'].loc[filename,self.label_translation['wsc'][self.label]].any()
        elif 'mros-visit1' in filename:
            return self.datasets['mros1'].loc[filename,self.label_translation['mros1'][self.label]].any()
        elif 'mros-visit2' in filename:
            return self.datasets['mros2'].loc[filename,self.label_translation['mros2'][self.label]].any()
        elif 'cfs' in filename:
            return self.datasets['cfs'].loc[filename,self.label_translation['cfs'][self.label]].any()
        elif 'hchs' in filename:
            return self.datasets['hchs'].loc[filename,self.label_translation['hchs'][self.label]].any()
        elif 'rf_c4_m1' in filename:
            return self.datasets['rf'].loc[filename,self.label_translation['rf'][self.label]].any()
        elif filename.startswith(tuple(['BOGN', 'GSBB', 'GSDV', 'GSLH', 'GSSA', 'GSSW', 'MAYO', 'MSMI', 'MSNF', 'MSQW', 'MSTH', 'MSTR', 'STLK', 'STNF'])):
            return self.datasets['stages'].loc[filename,self.label_translation['stages'][self.label]].any()
    ### now we will create a get for an entire list of filenames
    def get(self, filenames: List[str], dataset: str) -> List:
        ## use .loc to quickly get set of rows and then perform .any on the columns
        if 'shhs1' in dataset:
            if len(self.label_translation['shhs1'][self.label]) > 1:
                return self.datasets['shhs1'].loc[filenames,self.label_translation['shhs1'][self.label]].any(axis=1)
            else:
                return self.datasets['shhs1'].loc[filenames,self.label_translation['shhs1'][self.label]]
        elif 'shhs2' in dataset:
            if len(self.label_translation['shhs2'][self.label]) > 1:
                return self.datasets['shhs2'].loc[filenames,self.label_translation['shhs2'][self.label]].any(axis=1)
            else:
                return self.datasets['shhs2'].loc[filenames,self.label_translation['shhs2'][self.label]]
        elif 'wsc' in dataset:
            # filenames = [item.replace('_fold0','').replace('_fold1','').replace('_fold2','').replace('_fold3','') for item in filenames]
            if len(self.label_translation['wsc'][self.label]) > 1:
                return self.datasets['wsc'].loc[filenames,self.label_translation['wsc'][self.label]].any(axis=1)
            else:
                return self.datasets['wsc'].loc[filenames,self.label_translation['wsc'][self.label]]
        elif 'mros1' in dataset:
            if len(self.label_translation['mros1'][self.label]) > 1:
                return self.datasets['mros1'].loc[filenames,self.label_translation['mros1'][self.label]].any(axis=1)
            else:
                return self.datasets['mros1'].loc[filenames,self.label_translation['mros1'][self.label]]
        elif 'mros2' in dataset:
            # if len(self.label_translation['mros2'][self.label]) > 1:
            #     return self.datasets['mros2'].loc[filenames,self.label_translation['mros2'][self.label]].any(axis=1)
            # else:
            #     return self.datasets['mros2'].loc[filenames,self.label_translation['mros2'][self.label]]
            if len(self.label_translation['mros2'][self.label]) > 1:
                return self.datasets['mros2'].reindex(filenames)[self.label_translation['mros2'][self.label]].any(axis=1)
            else:
                return self.datasets['mros2'].reindex(filenames)[self.label_translation['mros2'][self.label]]
        elif 'cfs' in dataset:
            # if len(self.label_translation['cfs'][self.label]) > 1:
            #     return self.datasets['cfs'].loc[filenames,self.label_translation['cfs'][self.label]].any(axis=1)
            # else:
            #     return self.datasets['cfs'].loc[filenames,self.label_translation['cfs'][self.label]]
            if len(self.label_translation['cfs'][self.label]) > 1:
                return self.datasets['cfs'].reindex(filenames)[self.label_translation['cfs'][self.label]].any(axis=1)
            else:
                return self.datasets['cfs'].reindex(filenames)[self.label_translation['cfs'][self.label]]
        elif 'stages' in dataset:
            if len(self.label_translation['stages'][self.label]) > 1:
                return self.datasets['stages'].reindex(filenames)[self.label_translation['stages'][self.label]].any(axis=1)
            else:
                return self.datasets['stages'].reindex(filenames)[self.label_translation['stages'][self.label]]
            
        elif 'hchs' in dataset:
            if len(self.label_translation['hchs'][self.label]) > 1:
                return self.datasets['hchs'].reindex(filenames)[self.label_translation['hchs'][self.label]].any(axis=1)
            else:
                return self.datasets['hchs'].reindex(filenames)[self.label_translation['hchs'][self.label]]

        elif 'rf' in dataset:
            if len(self.label_translation['rf'][self.label]) > 1:
                return self.datasets['rf'].reindex(filenames)[self.label_translation['rf'][self.label]].any(axis=1)
                # return self.datasets['rf'].loc[filenames,self.label_translation['rf'][self.label]].any(axis=1)
            else:
                # return self.datasets['rf'].loc[filenames,self.label_translation['rf'][self.label]]
                return self.datasets['rf'].reindex(filenames)[self.label_translation['rf'][self.label]]
            
    ## this is if they come from different datasets
    def get_list(self, filenames: List[str], dataset: List[str]) -> List:
        
        filenames = [item.split('/')[-1] for item in filenames]
        
        shhs1_mask = ['shhs1' in item for item in dataset]
        shhs2_mask = ['shhs2' in item for item in dataset]
        wsc_mask = ['wsc' in item for item in dataset]
        mros1_mask = ['mros1' in item for item in dataset]
        mros2_mask = ['mros2' in item for item in dataset]
        cfs_mask = ['cfs' in item for item in dataset]
        hchs_mask = ['hchs' in item for item in dataset]
        rf_mask = ['rf' in item for item in dataset]
        stages_mask = ['stages' in item for item in dataset]
        
        output = np.zeros(len(filenames))
        if np.any(shhs1_mask):
            output[shhs1_mask] = self.get(list(np.array(filenames,dtype=object)[shhs1_mask]), 'shhs1').values.flatten()
        if np.any(shhs2_mask):
            output[shhs2_mask] = self.get(list(np.array(filenames,dtype=object)[shhs2_mask]), 'shhs2').values.flatten()
        if np.any(wsc_mask):
            output[wsc_mask] = np.nan #self.get(list(np.array(filenames,dtype=object)[wsc_mask]), 'wsc').values.flatten()
        if np.any(mros1_mask):
            output[mros1_mask] = self.get(list(np.array(filenames,dtype=object)[mros1_mask]), 'mros1').values.flatten()
        if np.any(mros2_mask):
            output[mros2_mask] = self.get(list(np.array(filenames,dtype=object)[mros2_mask]), 'mros2').values.flatten()
        if np.any(cfs_mask):
            output[cfs_mask] = self.get(list(np.array(filenames,dtype=object)[cfs_mask]), 'cfs').values.flatten()
        if np.any(hchs_mask):
            output[hchs_mask] = np.nan #self.get(list(np.array(filenames,dtype=object)[hchs_mask]), 'hchs')
        if np.any(rf_mask):
            output[rf_mask] = np.nan ## NEED TO IMPLEMENT #self.get(list(np.array(filenames,dtype=object)[rf_mask]), 'rf')
        if np.any(stages_mask): 
            output[stages_mask] = self.get(list(np.array(filenames,dtype=object)[stages_mask]), 'stages').values.flatten()
        
        return list(output)
        
        
        
    def get_gender(self, filenames: List[str], dataset: str) -> List:
        # if dataset == 'wsc':
        #     filenames = [item.replace('_fold0','').replace('_fold1','').replace('_fold2','').replace('_fold3','') for item in filenames]
        if dataset == 'mros2' or dataset == 'cfs' or dataset == 'hchs' or dataset=='stages' or dataset == 'rf':
            return self.datasets[dataset].reindex(filenames)['mit_gender']
        return self.datasets[dataset].loc[filenames, 'mit_gender']
    def get_age(self, filenames: List[str], dataset: str) -> List:
        # if dataset == 'wsc':
        #     filenames = [item.replace('_fold0','').replace('_fold1','').replace('_fold2','').replace('_fold3','') for item in filenames]
        if dataset == 'mros2' or dataset == 'cfs' or dataset == 'hchs' or dataset=='stages' or dataset == 'rf':
            return self.datasets[dataset].reindex(filenames)['mit_age']
        return self.datasets[dataset].loc[filenames, 'mit_age']
    def get_bmi(self, filenames: List[str], dataset: str) -> List:
        # if dataset == 'wsc':
        #     filenames = [item.replace('_fold0','').replace('_fold1','').replace('_fold2','').replace('_fold3','') for item in filenames]
        if dataset == 'mros2' or dataset == 'cfs' or dataset == 'hchs' or dataset=='stages' or dataset == 'rf':
            return self.datasets[dataset].reindex(filenames)['mit_bmi']
        return self.datasets[dataset].loc[filenames, 'mit_bmi']
    def get_race(self, filenames: List[str], dataset: str) -> List:
        # if dataset == 'wsc':
        #     filenames = [item.replace('_fold0','').replace('_fold1','').replace('_fold2','').replace('_fold3','') for item in filenames]
        if dataset == 'mros2' or dataset == 'cfs' or dataset == 'hchs' or dataset=='stages' or dataset == 'rf':
            return self.datasets[dataset].reindex(filenames)['mit_race']
        return self.datasets[dataset].loc[filenames, 'mit_race']
    
class Subset(Dataset):
    def __init__(self, dataset, indices, oversample_pos_class=0):
        self.dataset = dataset
        self.indices = indices
        self.oversample_pos_class = oversample_pos_class
        self.num_positive = 0
        if oversample_pos_class > 0:
            self.augment_pos_class()
        # if type(dataset) in [EEG_Encoding_HCHS_Dataset, EEG_Encoding_MrOS_Dataset]:
        #     self.indices = np.append(self.indices, (self.indices))
        #     print('doubling size of this dataset ')

    def augment_pos_class(self):
        pos_classes = [i for i,item in enumerate(self.dataset.all_valid_files) if (i in self.indices and self.dataset.get_label(item) == 1)]
        self.indices = np.append(self.indices, np.repeat(pos_classes,self.oversample_pos_class))

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        tempidx = self.indices[idx]
        if type(tempidx) != int:
            tempidx = int(tempidx)
        return self.dataset[tempidx]

    def __getitems__(self, indices):
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self):
        # if self.oversample_pos_class > 0:
        #     return len(self.indices) + self.num_positive * self.oversample_pos_class
        # else:
            return len(self.indices) 

class DatasetCombiner(Dataset):
    def __init__(self, datasets, phase, fold=0, pos_class_oversample=0, args=None):
        self.datasets = []
        self.datasets_len = []
        self.which_dataset = []
        for i,dataset in enumerate(datasets) :
            generator1 = torch.Generator().manual_seed(20)
            #trainset, valset = random_split(dataset, [int(0.7 * len(dataset)), len(dataset) - int(0.7 * len(dataset))], generator=generator1)
            # train_size = int(0.8 * len(dataset))
            # test_size = len(dataset) - train_size
            
            # trainset, valset = random_split(dataset, [train_size, test_size], generator=generator1)

            ## 
            if type(phase) == list:
                thisphase = phase[i]
            else:
                thisphase = phase 
            
            if type(dataset) != EEG_Encoding_RF_Dataset:
                if args is not None and args.CLEANED_DATA and type(dataset) not in [EEG_Encoding_HCHS_Dataset, EEG_Encoding_WSC_Dataset]:
                    if os.path.exists('/mnt/mit-data'):
                        test_ids = pd.read_csv(f'/mnt/mit-data/ali/data_4fold/fold/test.{str(args.fold)}.txt',header=None)[0].values
                        train_ids = pd.read_csv(f'/mnt/mit-data/ali/data_4fold/fold/train.{str(args.fold)}.txt',header=None)[0].values
                    else:
                        test_ids = pd.read_csv(f'/data/scratch/alimirz/DATA/RF_MAGE_ANTIDEP/fold/test.{str(args.fold)}.txt',header=None)[0].values
                        train_ids = pd.read_csv(f'/data/scratch/alimirz/DATA/RF_MAGE_ANTIDEP/fold/train.{str(args.fold)}.txt',header=None)[0].values
                        
                    train_idx = [] 
                    test_idx = [] 
                    [train_idx.append(dataset.all_valid_files.index(item.split(' ')[0].split('/')[-1]+'.npz')) for item in tqdm(train_ids) if item.split(' ')[0].split('/')[-1]+'.npz' in dataset.data_dict]
                    [test_idx.append(dataset.all_valid_files.index(item.split(' ')[0].split('/')[-1]+'.npz')) for item in tqdm(test_ids) if item.split(' ')[0].split('/')[-1]+'.npz' in dataset.data_dict]
                    assert len(train_idx) + len(test_idx) == len(dataset)
                else:
                    kf = KFold(n_splits=4, shuffle=True, random_state=20)
                    train_idx, test_idx = get_nth_element(kf.split(dataset),fold)
                    print(type(dataset), np.mean(train_idx), np.mean(test_idx))
                # trainset = torch.utils.data.Subset(dataset, train_idx)
                if thisphase == 'train':
                    if False: #type(dataset) == EEG_Encoding_MrOS_Dataset:
                        print('doubling training for mros oversampling of positive')
                        trainset = Subset(dataset, train_idx, oversample_pos_class=2 * pos_class_oversample)
                    else:
                        trainset = Subset(dataset, train_idx, oversample_pos_class=pos_class_oversample)
                else:
                    trainset = 'hello' ## time saver
                valset = torch.utils.data.Subset(dataset, test_idx)
            else:
                ## handled in the traning loop
                trainset = dataset 
                valset = dataset 
                dataset = dataset
                

            
            if thisphase == 'train':
                self.datasets.append(trainset)
                self.datasets_len.append(len(trainset))
                self.which_dataset.extend([i]*len(trainset))
            elif thisphase == 'val':
                self.datasets.append(valset)
                self.datasets_len.append(len(valset))
                self.which_dataset.extend([i]*len(valset))
            elif thisphase == 'whole':
                self.datasets.append(dataset)
                self.datasets_len.append(len(dataset))
                self.which_dataset.extend([i]*len(dataset))
                # self.datasets_len.append(len(valset))

    def __getitem__(self, idx):
        if False: #self.phase == 'whole':
            return self.datasets[idx], self.which_dataset[idx]
        else:
            aaa = datetime.datetime.now()
            tempidx = idx
            i = 0
            while True:
                if tempidx - self.datasets_len[i] < 0:
                    break 
                else:
                    tempidx = tempidx - self.datasets_len[i]
                    i += 1
            output = self.datasets[i][tempidx], self.which_dataset[idx]
            # print(datetime.datetime.now() - aaa, str(self.datasets[i]))
            return output
    def __len__(self):
        if False: #self.phase == 'whole':
            return self.datasets_len
        else:
            return sum(self.datasets_len)

class CrossFold():
    def __init__(self, num_folds, args, use_kaiwen_split=False, use_balanced_med_split=False, use_paper_split=False):
        self.num_folds = num_folds 
        self.use_kaiwen_split = use_kaiwen_split
        self.use_balanced_med_split = use_balanced_med_split
        self.use_paper_split = use_paper_split
        self.dataset = None
        self.args = args 
        """args parameters used: None"""
        if use_kaiwen_split or use_paper_split:
            assert self.num_folds == 4
        if use_balanced_med_split:
            assert self.num_folds == 4
        
    """Assuming the df shares a common filepath"""
    def add(self, df: pd.DataFrame, filepath: str, is_rf: bool = False, is_external: bool=False, gender=None, age=None, race=None, bmi=None):

        if type(df) == pd.DataFrame and df.shape[1] == 1:
            df = df[df.columns[0]]
        if type(df) == pd.Series:
            df = df.to_frame(name=self.args.label)

        if df[self.args.label].isna().any():
            print('Dropping NAs')
            df = df[df[self.args.label].notna()]
        
        df['filename'] = df.index.values
        df['filepath'] = df['filename'].apply(lambda x: filepath + x)
        df['is_rf'] = is_rf
        df['is_external'] = is_external
        df['pid'] = ''
        if is_rf:
            df['pid'] = df['filepath'].apply(lambda x: x.split('/')[-2])
        if gender is not None:
            df['mit_gender'] = gender 
        if age is not None:
            df['mit_age'] = age 
        if race is not None:
            df['mit_race'] = race 
        if bmi is not None:
            df['mit_bmi'] = bmi 
        # df['filepath'] = [filepath + x for x in df['filename']]
        
        if self.dataset is None:
            self.dataset = df
        else:
            self.dataset = pd.concat([self.dataset, df])
          
    def filter(self, column, value):
        if self.dataset is None:
            raise ValueError('Dataset is empty')
        print('Filtering ', column, value)
        print('Before filtering ', self.dataset.shape)
        self.dataset = self.dataset[self.dataset[column] == value]
        print('After filtering ', self.dataset.shape)
      
    def split_from_arr(self, all_train_ids: list, val=False, return_datasets=True):
        print(self.dataset.shape)
        print(len(all_train_ids))

        # all_train_ids = [item.split('/')[-1] for item in all_train_ids]
        all_train_ids = self.dataset[self.dataset['filepath'].isin(all_train_ids)].index
        # all_train_ids = self.dataset['filepath'].intersection(all_train_ids)
        print(self.dataset.shape)
        print(len(all_train_ids))
        
        if 'mit_gender' in self.dataset.columns:
            all_train_gender = self.dataset.loc[all_train_ids,'mit_gender'].values 
        else:
            all_train_gender = None 

        if 'mit_age' in self.dataset.columns:
            all_train_age = self.dataset.loc[all_train_ids,'mit_age'].values 
        else:
            all_train_age = None 

        if 'mit_race' in self.dataset.columns:
            all_train_race = self.dataset.loc[all_train_ids,'mit_race'].values 
        else:
            all_train_race = None 
            
        if 'mit_bmi' in self.dataset.columns:
            all_train_race = self.dataset.loc[all_train_ids,'mit_bmi'].values 
        else:
            all_train_race = None 
        
        all_train_labels = self.dataset.loc[all_train_ids,self.args.label].values
        all_train_pids = self.dataset.loc[all_train_ids,'pid'].values
        if all(all_train_pids == ''):
            all_train_pids = None
        all_train_ids = self.dataset.loc[all_train_ids,'filepath'].values
        
        if return_datasets:
            return self.create_dataset(all_train_ids, all_train_labels, pids=all_train_pids, gender=all_train_gender, age=all_train_age, race=all_train_race, val=val)
        else:
            train_df = pd.DataFrame({
                'id': all_train_ids,
                'label': all_train_labels,
                'pid': all_train_pids,
                'mit_gender': all_train_gender,
                'mit_race': all_train_race,
                'mit_age': all_train_age,
            })
            return train_df
        
    def split(self, fold:int, return_datasets: bool = False):
        
        all_train_ids = []
        all_test_ids = self.dataset[self.dataset['is_external']]['filename'].values.tolist()
        dataset = deepcopy(self.dataset[~self.dataset['is_external']])
        
        if self.use_kaiwen_split or self.use_paper_split:
            if self.use_kaiwen_split:
                test_ids = pd.read_csv(f'/data/scratch/alimirz/DATA/RF_MAGE_ANTIDEP/fold/test.{str(fold)}.txt',header=None)[0].values
                train_ids = pd.read_csv(f'/data/scratch/alimirz/DATA/RF_MAGE_ANTIDEP/fold/train.{str(fold)}.txt',header=None)[0].values
                train_ids = [item.split(' ')[0].split('/')[-1]+'.npz' for item in train_ids]
                test_ids = [item.split(' ')[0].split('/')[-1]+'.npz' for item in test_ids]
            else:
                train_ids = pd.read_csv(f'/data/netmit/sleep_lab/filtered/MAGE/TRAINING/fold_splits/train.{str(fold)}.txt',header=None)[0].values
                test_ids = pd.read_csv(f'/data/netmit/sleep_lab/filtered/MAGE/TRAINING/fold_splits/test.{str(fold)}.txt',header=None)[0].values
                ## adding the rf and hchs train and test ids as well to keep consistent 
                df_others = pd.read_csv('/data/netmit/sleep_lab/filtered/MAGE/TRAINING/fold_splits/mapping_hchs_rf_charlie.csv')
                train_ids = np.array([item.split(' ')[0].split('/')[-1]+'.npz' for item in train_ids])
                test_ids = np.array([item.split(' ')[0].split('/')[-1]+'.npz' for item in test_ids])
                train_ids = np.append(df_others[df_others['fold'] != fold]['filepath'].apply(lambda x: x.split('/')[-2]+'/'+x.split('/')[-1] if '/rf/' in x else x.split('/')[-1]), train_ids)
                test_ids = np.append(df_others[df_others['fold'] == fold]['filepath'].apply(lambda x: x.split('/')[-2]+'/'+x.split('/')[-1] if '/rf/' in x else x.split('/')[-1]), test_ids)
                

            print('len before', len(train_ids))
            
            train_ids = list(np.intersect1d(train_ids, dataset['filename'].values))
            test_ids = list(np.intersect1d(test_ids, dataset['filename'].values))
            print('len after', len(train_ids))
            dataset['in_kaiwen_split'] = False 
            dataset.loc[train_ids,'in_kaiwen_split'] = True
            dataset.loc[test_ids,'in_kaiwen_split'] = True
            
            train_ids = dataset.loc[train_ids,'filename'].values.tolist()
            test_ids = dataset.loc[test_ids,'filename'].values.tolist()
            """Handling the case where the dataset is RF, need to split per patient"""
            # kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=20)
            ## kf is not used in the paper split, but is used in the kaiwen split
            kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=20)
            try:
                ## in the paper split, d1 is shape 0 
                d1 = dataset[(~dataset['in_kaiwen_split'])*(~dataset['is_rf'])]
                if d1.shape[0] == 0:
                    all_train_ids = all_train_ids + train_ids
                    all_test_ids = all_test_ids + test_ids
                    print('no data outside rf and kaiwen split')
                else:
                    train_idx, test_idx = get_nth_element(kf.split(d1, d1[self.args.label]),fold)
                    all_train_ids = all_train_ids + train_ids + d1.iloc[train_idx]['filename'].values.tolist()
                    all_test_ids = all_test_ids + test_ids + d1.iloc[test_idx]['filename'].values.tolist()
            except:
                
                print('No non-rf data')
            
            # in the paper split, this is false 
            if dataset['is_rf'].any() and not dataset['in_kaiwen_split'].any():
                rf_data = dataset[(~dataset['in_kaiwen_split'])*(dataset['is_rf'])]
                pids = rf_data['pid'].unique()
                try:
                    rf_data['median_label'] = rf_data.groupby('pid')[self.args.label].transform('median')
                    median_labels = rf_data.drop_duplicates('pid')['median_label'] 
                except:
                    median_labels = np.zeros_like(pids)
                    median_labels[::2] = 1
                    median_labels = median_labels.astype(bool)
                ### pid take out the v1 of the pid if v2 exists
                kk = np.array([item.split(' ')[0].split('_')[0] for item in pids])
                duplicates = np.unique(kk[np.unique(kk, return_index=True, return_counts=True)[2] > 1])
                if len(duplicates) > 0:
                    bp() 
                    raise NotImplementedError
                
                train_idx, test_idx = get_nth_element(kf.split(pids, median_labels),fold)
                train_pids = pids[train_idx]
                test_pids = pids[test_idx]
                train_filenames = rf_data[rf_data['pid'].isin(train_pids)]['filename'].values.tolist()
                test_filenames = rf_data[rf_data['pid'].isin(test_pids)]['filename'].values.tolist()
                
                all_train_ids = all_train_ids + train_filenames 
                all_test_ids = all_test_ids + test_filenames
            
            if 'mit_gender' in self.dataset.columns:
                all_train_gender = self.dataset.loc[all_train_ids,'mit_gender'].values 
                all_test_gender = self.dataset.loc[all_test_ids,'mit_gender'].values 
            else:
                all_train_gender = None 
                all_test_gender = None 

            if 'mit_age' in self.dataset.columns:
                all_train_age = self.dataset.loc[all_train_ids,'mit_age'].values 
                all_test_age = self.dataset.loc[all_test_ids,'mit_age'].values 
            else:
                all_train_age = None 
                all_test_age = None 
                
            if 'mit_bmi' in self.dataset.columns:
                all_train_bmi = self.dataset.loc[all_train_ids,'mit_bmi'].values 
                all_test_bmi = self.dataset.loc[all_test_ids,'mit_bmi'].values 
            else:
                all_train_bmi = None 
                all_test_bmi = None 

            if 'mit_race' in self.dataset.columns:
                all_train_race = self.dataset.loc[all_train_ids,'mit_race'].values 
                all_test_race = self.dataset.loc[all_test_ids,'mit_race'].values 
            else:
                all_train_race = None 
                all_test_race = None 
            all_train_labels = self.dataset.loc[all_train_ids,self.args.label].values
            all_train_pids = self.dataset.loc[all_train_ids,'pid'].values
            all_test_labels = self.dataset.loc[all_test_ids,self.args.label].values
            
            all_train_ids = self.dataset.loc[all_train_ids,'filepath'].values
            all_test_ids = self.dataset.loc[all_test_ids,'filepath'].values
            
            self.all_train_ids = all_train_ids
            self.all_test_ids = all_test_ids
            
            if return_datasets:
                if False: #self.args.task == 'multilabel':
                    print('Filtering Multilabel: ')
                    print(len(all_train_labels), len(all_test_labels))
                    idxs = np.zeros(len(all_train_labels))
                    
                    for i,item in enumerate(all_train_labels):
                        try:
                            if sum(eval(item)) > 0:
                                idxs[i] = 1
                        except:
                            print('error', item)
                            continue
                    
                    all_train_ids = all_train_ids[idxs == 1]
                    all_train_labels = all_train_labels[idxs == 1]
                    all_train_pids = all_train_pids[idxs == 1]
                    idxs = np.zeros(len(all_test_labels))
                    for i,item in enumerate(all_test_labels):
                        try:
                            if sum(eval(item)) > 0:
                                idxs[i] = 1
                        except:
                            print('error', item)
                            continue
                    all_test_ids = all_test_ids[idxs == 1]
                    all_test_labels = all_test_labels[idxs == 1]
                    print(len(all_train_labels), len(all_test_labels))
                
                return self.create_dataset(all_train_ids, all_train_labels, pids=all_train_pids, gender=all_train_gender, age=all_train_age, race=all_train_race, bmi=all_train_bmi), self.create_dataset(all_test_ids, all_test_labels, val=True, gender=all_test_gender, age=all_test_age, race=all_test_race, bmi=all_test_bmi)
            else:
                train_df = pd.DataFrame({
                    'id': all_train_ids,
                    'label': all_train_labels,
                    'pid': all_train_pids,
                    'mit_gender': all_train_gender,
                    'mit_race': all_train_race,
                    'mit_age': all_train_age,
                    'mit_bmi': all_train_bmi
                })

                test_df = pd.DataFrame({
                    'id': all_test_ids,
                    'label': all_test_labels,
                    'mit_gender': all_test_gender,
                    'mit_race': all_test_race,
                    'mit_age': all_test_age,
                    'mit_bmi': all_test_bmi
                })
                return train_df, test_df #[all_train_ids, all_train_labels], [all_test_ids, all_test_labels]
        elif self.use_balanced_med_split:
            test_ids = pd.read_csv(f'/data/scratch/alimirz/DATA/RF_MAGE_ANTIDEP/fold_v4/test.{str(fold)}.txt',header=None)[0].values
            train_ids = pd.read_csv(f'/data/scratch/alimirz/DATA/RF_MAGE_ANTIDEP/fold_v4/train.{str(fold)}.txt',header=None)[0].values
            
            print('len before', len(train_ids))
            
            ## for the rf filenames, make sure to add the prefix corresponding to pid 
            mapping_translation = self.dataset[self.dataset['is_rf']]['filename'].tolist()
            mapping_translation_key = [ x.split('/')[-1] for x in mapping_translation]
            mapping_dict = dict(zip(mapping_translation_key, mapping_translation))
            
            train_ids = [mapping_dict[item] if item in mapping_dict else item for item in train_ids]
            test_ids = [mapping_dict[item] if item in mapping_dict else item for item in test_ids]
            
            print('len middle', len(train_ids))
            
            
            
            train_ids = list(np.intersect1d(train_ids, dataset['filename'].values))
            test_ids = list(np.intersect1d(test_ids, dataset['filename'].values))
            
            print('len after', len(train_ids))
            
            
            train_ids = dataset.loc[train_ids,'filename'].values.tolist()
            test_ids = dataset.loc[test_ids,'filename'].values.tolist()

            all_train_ids = all_train_ids + train_ids 
            all_test_ids = all_test_ids + test_ids

            if 'mit_gender' in self.dataset.columns:
                all_train_gender = self.dataset.loc[all_train_ids,'mit_gender'].values 
                all_test_gender = self.dataset.loc[all_test_ids,'mit_gender'].values 
            else:
                all_train_gender = None 
                all_test_gender = None 

            if 'mit_age' in self.dataset.columns:
                all_train_age = self.dataset.loc[all_train_ids,'mit_age'].values 
                all_test_age = self.dataset.loc[all_test_ids,'mit_age'].values 
            else:
                all_train_age = None 
                all_test_age = None 
                
            if 'mit_bmi' in self.dataset.columns:
                all_train_bmi = self.dataset.loc[all_train_ids,'mit_bmi'].values 
                all_test_bmi = self.dataset.loc[all_test_ids,'mit_bmi'].values 
            else:
                all_train_bmi = None 
                all_test_bmi = None 

            if 'mit_race' in self.dataset.columns:
                all_train_race = self.dataset.loc[all_train_ids,'mit_race'].values 
                all_test_race = self.dataset.loc[all_test_ids,'mit_race'].values 
            else:
                all_train_race = None 
                all_test_race = None 
            all_train_labels = self.dataset.loc[all_train_ids,self.args.label].values
            all_train_pids = self.dataset.loc[all_train_ids,'pid'].values
            if all(all_train_pids == ''):
                all_train_pids = None
            
            all_test_labels = self.dataset.loc[all_test_ids,self.args.label].values
            
            all_train_ids = self.dataset.loc[all_train_ids,'filepath'].values
            all_test_ids = self.dataset.loc[all_test_ids,'filepath'].values
            
            self.all_train_ids = all_train_ids
            self.all_test_ids = all_test_ids
            if return_datasets:
                return self.create_dataset(all_train_ids, all_train_labels, pids=all_train_pids, gender=all_train_gender, age=all_train_age, race=all_train_race, bmi=all_train_bmi), self.create_dataset(all_test_ids, all_test_labels, val=True, gender=all_test_gender, age=all_test_age, race=all_test_race, bmi=all_test_bmi)
            else:
                train_df = pd.DataFrame({
                    'id': all_train_ids,
                    'label': all_train_labels,
                    'pid': all_train_pids,
                    'mit_gender': all_train_gender,
                    'mit_race': all_train_race,
                    'mit_age': all_train_age,
                    'mit_bmi': all_train_bmi
                })

                test_df = pd.DataFrame({
                    'id': all_test_ids,
                    'label': all_test_labels,
                    'mit_gender': all_test_gender,
                    'mit_race': all_test_race,
                    'mit_age': all_test_age,
                    'mit_bmi': all_test_bmi
                })
                return train_df, test_df #[all_train_ids, all_train_labels], [all_test_ids, all_test_labels]
            

        else:
            raise NotImplementedError
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=20)
            train_idx, test_idx = get_nth_element(kf.split(self.dataset),fold)
            
        
    def create_dataset(self, datapaths: List[str], labels: List[str], pids = None, val: bool = False, gender = None, age = None, race = None, bmi=None) -> Dataset:
        if pids is None:
            return UniversalDataset(datapaths, labels, self.args, val=val, gender=gender, age=age, race=race, bmi=bmi)
        else:
            from torch.utils.data import ConcatDataset
            pids = np.array(pids)
            datapaths = np.array(datapaths)
            labels = np.array(labels)
            if gender is not None:
                gender1 = gender[pids=='']
                gender2 = gender[pids!='']
            else:
                gender1 = None
                gender2 = None
            if age is not None:
                age1 = age[pids=='']
                age2 = age[pids!='']
            else:
                age1 = None
                age2 = None
            if race is not None:
                race1 = race[pids=='']
                race2 = race[pids!='']
            else:
                race1 = None
                race2 = None

            if bmi is not None:
                bmi1 = bmi[pids=='']
                bmi2 = bmi[pids!='']
            else:
                bmi1 = None
                bmi2 = None
                
            dataset1 = UniversalDataset(datapaths[pids == ''], labels[pids == ''], self.args, val=val, gender=gender1, age=age1, race=race1, bmi=bmi1)
            dataset2 = ResampledTrainDataset(datapaths[pids != ''], labels[pids != ''], pids[pids != ''], self.args, gender=gender2, age=age2, race=race2, bmi=bmi2)
            return ConcatDataset([dataset1, dataset2])

class ResampledTrainDataset(Dataset):
    """This dataset is meant for datasets with multisample aptients, so every time it is called, it will return a different sample"""
    def __init__(self, data, labels, pids, args, gender=None, age=None, race=None, bmi=None):
        self.data = data
        self.labels = labels
        self.gender = gender
        self.age = age 
        self.race = race
        self.bmi = bmi
        # self.pids = pids
        self.args = args 
        self.pids = np.unique(pids)
        self.training_resample = self.args.training_resample_number if self.args.training_resample and hasattr(self.args,'training_resample_number') else 60 if self.args.training_resample else None 
        self.pid_indices = defaultdict(list)
        for i, pid in enumerate(pids):
            self.pid_indices[pid].append(i)
        # 
        print('done')
    def __len__(self):
        if self.training_resample is None:
            return len(self.data)
        return len(self.pids) * self.training_resample
    def __getitem__(self, idx):
        if self.training_resample is None:
            feature = UniversalDataset.fetch_data(self.data[idx], self.args)
            # label= self.labels[idx]
            label = self.labels[idx]
            if type(label) == str:
                label= torch.tensor(eval(label), dtype=torch.int64)
            else:
                label= torch.tensor(label, dtype=torch.int64)
            
            label_dict = {'filepath': self.data[idx], 'label':label, 'dataset': parse_dataset_name(self.data[idx])}
            if self.gender is not None:
                label_dict['mit_gender'] = self.gender[idx]
            if self.age is not None:
                label_dict['mit_age'] = self.age[idx]
            if self.race is not None:
                label_dict['mit_race'] = self.race[idx]
            if self.bmi is not None:
                label_dict['mit_bmi'] = self.bmi[idx]
                
            return feature, label_dict
        else:
            pid = self.pids[idx % len(self.pids)]
            # idx = np.random.choice(np.where(self.pids == pid)[0])
            random_idx = np.random.choice(self.pid_indices[pid])
            feature = UniversalDataset.fetch_data(self.data[random_idx], self.args)
            label = self.labels[random_idx]
            if type(label) == str:
                label= torch.tensor(eval(label), dtype=torch.int64)
            else:
                label= torch.tensor(label, dtype=torch.int64)
            label_dict = {'filepath': self.data[random_idx], 'label':label, 'dataset': parse_dataset_name(self.data[random_idx])}
            if self.gender is not None:
                label_dict['mit_gender'] = self.gender[random_idx]
            if self.age is not None:
                label_dict['mit_age'] = self.age[random_idx]
            if self.race is not None:
                label_dict['mit_race'] = self.race[random_idx]
            
            return feature, label_dict
            # return self.data[idx], self.labels[idx]

class FairUniversalDataset(Dataset):
    def __init__(self, dataframe, args, label_column='label', categories=['mit_age', 'mit_race', 'mit_gender'], 
                 target_positive_ratio=0.2, resample_size=None):
        """
        Initializes the dataset with resampling to achieve balanced positive proportion and class proportion.
        
        Args:
            dataframe (pd.DataFrame): Dataset as a DataFrame with binary labels and demographic categories.
            label_column (str): Column name of the binary label.
            categories (list): List of demographic category column names to balance on.
            target_positive_ratio (float): Target ratio of positive samples within each category.
            resample_size (int or None): Fixed number of samples per epoch (None means use original dataset size).
        """
        self.dataframe = dataframe
        self.label_column = label_column
        self.categories = categories
        self.args = args 
        self.target_positive_ratio = target_positive_ratio
        self.resample_size = resample_size if resample_size else len(dataframe)
        self.age_bins = np.array([20, 30, 40, 50, 60, 70, 80, 100])
        # Resample the data to achieve target proportions and positive ratios per category
        self.resampled_df = self._resample_data()
    
    def _resample_data(self):
        
        self.dataframe['mit_age_raw'] = deepcopy(self.dataframe['mit_age'].values)
        
        buc_val = np.digitize(self.dataframe['mit_age'].values, bins=self.age_bins, right=True)
        self.dataframe['mit_age'] = np.clip(buc_val,0,7)
        
        age_bins = 8
        gender_bins = 2 
        race_bins = 6
        outputs = [] 
        
        for label_val in [1,0]:
            age_prop = np.zeros((age_bins,))
            gender_prop = np.zeros((gender_bins,))
            race_prop = np.zeros((race_bins,))
            
            df_subset = None
            
            if label_val == 1:
                target_pos_count = int(self.target_positive_ratio * self.resample_size)
            else:
                target_pos_count = self.resample_size - int(self.target_positive_ratio * self.resample_size)
            ## we're going to priritize by rarity of the category, to fill the positives, then fill the negatives
            pos_df = self.dataframe[self.dataframe[self.label_column] == label_val]
            
            for age in range(age_bins):
                age_prop[age] = (pos_df['mit_age'] == age).mean()
            for race in range(race_bins):
                race_prop[race] = (pos_df['mit_race'] == race).mean()
            for gender in range(gender_bins):
                gender_prop[gender] = (pos_df['mit_gender'] == gender+1).mean()
            
            idx_age = 0
            idx_race = 0 
            idx_gender = 0
            
            age_prop2 = np.argsort(age_prop)
            gender_prop2 = np.argsort(gender_prop)
            race_prop2 = np.argsort(race_prop)
            
            idx = 0
            while idx < age_bins + gender_bins + race_bins:
                candidates = [] 
                if idx_age < age_bins:
                    candidates.append(age_prop[age_prop2[idx_age]])
                else:
                    candidates.append(1e9)
                if idx_gender < gender_bins:
                    candidates.append(gender_prop[gender_prop2[idx_gender]])
                else:
                    candidates.append(1e9)
                if idx_race < race_bins:
                    candidates.append(race_prop[race_prop2[idx_race]])
                else:
                    candidates.append(1e9)
                
                
                target = np.argmin(candidates)
                sampled = None
                
                if target == 0:
                    num_to_sample = target_pos_count//age_bins
                    if df_subset is not None:
                        num_to_sample = num_to_sample - (df_subset['mit_age'] == age_prop2[idx_age]).sum()
                    
                    subset = pos_df[pos_df['mit_age'] == age_prop2[idx_age]]
                    if num_to_sample > 0 and subset.shape[0] > 0:
                        # print('age', age_prop2[idx_age], num_to_sample, subset.shape)
                        sampled = subset.sample(n=num_to_sample, replace=True)
                    idx_age += 1
                elif target == 1:
                    num_to_sample = target_pos_count//gender_bins
                    if df_subset is not None:
                        num_to_sample = num_to_sample - (df_subset['mit_gender'] == 1+gender_prop2[idx_gender]).sum()
                    
                    subset = pos_df[pos_df['mit_gender'] == 1+gender_prop2[idx_gender]]
                    if num_to_sample > 0 and subset.shape[0] > 0:
                        # print('gender', gender_prop2[idx_gender], num_to_sample, subset.shape)
                        sampled = subset.sample(n=num_to_sample, replace=True)
                    idx_gender += 1
                elif target == 2:
                    num_to_sample = target_pos_count//race_bins
                    if df_subset is not None:
                        num_to_sample = num_to_sample - (df_subset['mit_race'] == race_prop2[idx_race]).sum()
                    
                    subset = pos_df[pos_df['mit_race'] == race_prop2[idx_race]]
                    if num_to_sample > 0 and subset.shape[0] > 0:
                        # print('race', race_prop2[idx_race], num_to_sample, subset.shape)
                        sampled = subset.sample(n=num_to_sample, replace=True)
                    idx_race += 1
                else:
                    assert False 
                
                
                if sampled is not None:
                    if df_subset is None:
                        df_subset = sampled 
                    else:
                        df_subset = pd.concat([df_subset, sampled])
                
                idx += 1
            
            print('trimming extra')
            extra_patients = df_subset.shape[0] - target_pos_count
            age_extras = df_subset['mit_age'].value_counts() - target_pos_count//age_bins
            race_extras = df_subset['mit_race'].value_counts() - target_pos_count//race_bins
            gender_extras = df_subset['mit_gender'].value_counts() - target_pos_count//gender_bins
            df_subset = df_subset.sample(frac=1).reset_index(drop=True)
            toss_indices = [] 
            for i,row in df_subset.iterrows():
                if extra_patients > 0 and age_extras[row['mit_age']] > 0 and race_extras[row['mit_race']] > 0 and gender_extras[row['mit_gender']] > 0:
                    print('tossing', row)
                    toss_indices.append(i)
                    extra_patients -= 1
                    age_extras[row['mit_age']] -= 1
                    gender_extras[row['mit_gender']] -= 1
                    race_extras[row['mit_race']] -= 1
            df_subset = df_subset.drop(toss_indices)
            outputs.append(deepcopy(df_subset))
        print('done')
        outputs = pd.concat(outputs).sample(frac=1).reset_index(drop=True)

        return outputs

    def __len__(self):
        return len(self.resampled_df)

    def __getitem__(self, idx):
        row = self.resampled_df.iloc[idx]
        label = row[self.label_column]

        feature = UniversalDataset.fetch_data(row['id'], self.args)
        
        
        if type(label) == str:
            label = torch.tensor(eval(row['label']), dtype=torch.int64)
        else:
            label = torch.tensor(int(row['label']), dtype=torch.int64)
            
        label_dict = {'filepath': row['id'], 'label':label, 'dataset': parse_dataset_name(row['id'])}
        
        if row['mit_gender'] is not None:
            label_dict['mit_gender'] = row['mit_gender']
        if row['mit_age'] is not None:
            label_dict['mit_age'] = row['mit_age']
        if row['mit_race'] is not None:
            label_dict['mit_race'] = row['mit_race']
        
        return feature, label_dict

class UniversalDataset(Dataset):
    def __init__(self, data, labels, args, val=False, gender=None, age=None, race=None, bmi=None): 
        """args parameters used: stage_input, mage_pred_input, use_gt_as_mage_pred, NOISE_PADDING"""
        self.args = args 
        self.data = data
        self.labels = labels
        self.gender = gender 
        self.age = age 
        self.race = race 
        self.bmi = bmi 
        
        if not val and self.args.pos_class_oversample > 0:
            pos_class = np.where(self.labels == 1)[0]
            self.data = np.append(self.data, np.repeat(self.data[pos_class], 1 + self.args.pos_class_oversample))
            self.labels = np.append(self.labels, np.repeat(self.labels[pos_class], 1 + self.args.pos_class_oversample))
            if self.gender is not None:
                self.gender = np.append(self.gender, np.repeat(self.gender[pos_class], 1 + self.args.pos_class_oversample))
            if self.age is not None:
                self.age = np.append(self.age, np.repeat(self.age[pos_class], 1 + self.args.pos_class_oversample))
            if self.race is not None:
                self.race = np.append(self.race, np.repeat(self.race[pos_class], 1 + self.args.pos_class_oversample))
            if self.bmi is not None:
                self.bmi = np.append(self.bmi, np.repeat(self.bmi[pos_class], 1 + self.args.pos_class_oversample))


        if not val and self.args.minority_pos_oversample > 0 and self.race is not None:
            ## only oversample the positive class
            pos_class = np.where((self.labels == 1)*((self.race == 5)+(self.race == 2)+(self.race == 3)))[0]
            self.data = np.append(self.data, np.repeat(self.data[pos_class], 1 + self.args.minority_pos_oversample))
            self.labels = np.append(self.labels, np.repeat(self.labels[pos_class], 1 + self.args.minority_pos_oversample))
            if self.gender is not None:
                self.gender = np.append(self.gender, np.repeat(self.gender[pos_class], 1 + self.args.minority_pos_oversample))
            if self.age is not None:
                self.age = np.append(self.age, np.repeat(self.age[pos_class], 1 + self.args.minority_pos_oversample))
            if self.race is not None:
                self.race = np.append(self.race, np.repeat(self.race[pos_class], 1 + self.args.minority_pos_oversample))
            if self.bmi is not None:
                self.bmi = np.append(self.bmi, np.repeat(self.bmi[pos_class], 1 + self.args.minority_pos_oversample))

            
        elif not val and self.args.black_oversample > 0 and self.race is not None:
            ## oversample both classes
            pos_class = np.where((self.race == 2))[0]
            self.data = np.append(self.data, np.repeat(self.data[pos_class], 1 + self.args.black_oversample))
            self.labels = np.append(self.labels, np.repeat(self.labels[pos_class], 1 + self.args.black_oversample))
            if self.gender is not None:
                self.gender = np.append(self.gender, np.repeat(self.gender[pos_class], 1 + self.args.black_oversample))
            if self.age is not None:
                self.age = np.append(self.age, np.repeat(self.age[pos_class], 1 + self.args.black_oversample))
            if self.race is not None:
                self.race = np.append(self.race, np.repeat(self.race[pos_class], 1 + self.args.black_oversample))
            if self.bmi is not None:
                self.bmi = np.append(self.bmi, np.repeat(self.bmi[pos_class], 1 + self.args.black_oversample))
    
    def __len__(self):
        # return len(self.data_dict)
        return len(self.labels)
    
    def __getitem__(self, idx):
        filepath = self.data[idx]
        label = self.labels[idx]
        
        feature = self.fetch_data(filepath, self.args)
        
        if self.args.task == 'regression':
            label = torch.tensor(label, dtype=torch.float32)
        else:
            if type(label) == str:
                label = torch.tensor(eval(label), dtype=torch.int64)
            else:
                label = torch.tensor(label, dtype=torch.int64)
            
        label_dict = {'filepath': filepath, 'label':label, 'dataset': parse_dataset_name(filepath)}
        if self.gender is not None:
            label_dict['mit_gender'] = self.gender[idx]
        if self.age is not None:
            label_dict['mit_age'] = self.age[idx]
        if self.race is not None:
            label_dict['mit_race'] = self.race[idx]
        if self.bmi is not None:
            label_dict['mit_bmi'] = self.bmi[idx]
        
        return feature, label_dict
    @staticmethod
    def fetch_data(filepath, args):
        
        x = np.load(filepath)
        
        if hasattr(args,'stage_input') and args.stage_input:
            input_size = 2 * 60 * 10
            feature = x['data']
            stages_fs = x['fs']
            factor = round(stages_fs * 30)
            feature = feature[::factor]
            feature = process_stages(feature)
            
            if len(feature) > input_size:
                feature = feature[:input_size]
            else:
                feature = np.concatenate((feature, np.zeros((input_size-len(feature)),dtype=int)), axis=0)
        elif args.mage_pred_input:
            if args.use_gt_as_mage_pred:
                X = x['data']
            else:
                X = x['pred']
            X_len = X.shape[1]
            if X_len > 1024:
                X = X[:, :1024]
            elif X_len < 1024:
                X = np.pad(X, ((0, 0), (0, 1024 - X_len)), mode="constant", constant_values=0)

            # normalize X
            # X_mean = np.mean(X)
            # X_std = np.std(X)
            # X = (X - X_mean) / X_std
            feature = X.astype(np.float32)
        else:
            feature = x['decoder_eeg_latent']
            if False: #not self.embedding_v2:
                feature = feature.squeeze(0)
            else:
                feature = np.moveaxis(feature, 0,-1)
                # feature = feature.reshape(feature.shape[0], feature.shape[1] * feature.shape[2])
            if feature.shape[0] >= args.MAGE_INPUT_SIZE:
                feature = feature[:args.MAGE_INPUT_SIZE, :,:]
            else:
                # feature = np.concatenate((feature, np.zeros_like(feature,dtype=np.float32)[:150-feature.shape[0]]), axis=0)
                if args.NOISE_PADDING:
                    padding2 = np.random.randn(args.MAGE_INPUT_SIZE-feature.shape[0],feature.shape[-2],feature.shape[-1]).astype(np.float32)
                    split_idx = np.random.randint(0, padding2.shape[0])
                    feature = np.concatenate((padding2[:split_idx], feature, padding2[split_idx:]), axis=0)
                else:
                    padding2 = np.zeros((args.MAGE_INPUT_SIZE-feature.shape[0],feature.shape[-2],feature.shape[-1]),dtype=np.float32)
                    feature = np.concatenate((feature, padding2), axis=0)
        feature = torch.from_numpy(feature)
        return feature 
