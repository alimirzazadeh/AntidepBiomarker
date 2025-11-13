
import numpy as np 
import os 
from tqdm import tqdm
from ipdb import set_trace as bp
import sys 

DEBUG = len(sys.argv) > 1 and sys.argv[1] == 'debug'
if DEBUG:
    DATASET_LIST = ['cfs']
else:
    DATASET_LIST = ['cfs','shhs1', 'shhs2','mros1','mros2','wsc', 'hchs']

signal_dir = '/data/netmit/sleep_lab/filtered/MAGE/DATASET_new/abdominal/'
all_durations = []
all_durations_dataset = {}
for dataset in tqdm(DATASET_LIST):
    all_durations_dataset[dataset] = []
    dataset_dir = signal_dir.replace('DATASET', dataset)
    if dataset == 'hchs':
        dataset_dir = '/data/netmit/wifall/ADetect/data/hchs/airflow/'
    # elif dataset == 'rf':
    #     dataset_dir = '/data/netmit/sleep_lab/filtered/MAGE/DATASET_new/rf/'
    signal_files = os.listdir(dataset_dir)
    for file in tqdm(signal_files):
        signal = np.load(dataset_dir + file)
        fs = signal['fs']
        data = signal['data']
        duration = len(data) / fs / 3600 # in hours
        all_durations.append(duration)
        all_durations_dataset[dataset].append(duration)
print(np.mean(all_durations), np.std(all_durations))
for dataset in DATASET_LIST:
    print(dataset, np.mean(all_durations_dataset[dataset]), np.std(all_durations_dataset[dataset]))