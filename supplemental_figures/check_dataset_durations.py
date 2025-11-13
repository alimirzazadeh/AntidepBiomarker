
import numpy as np 
import os 
from tqdm import tqdm
from ipdb import set_trace as bp
import sys 

DEBUG = sys.argv[1] == 'debug'
if DEBUG:
    DATASET_LIST = ['cfs']
else:
    DATASET_LIST = ['cfs','shhs1', 'shhs2','mros1','mros2','wsc']

signal_dir = '/data/netmit/sleep_lab/filtered/MAGE/DATASET_new/abdominal/'
all_durations = []
for dataset in tqdm(DATASET_LIST):
    signal_files = os.listdir(signal_dir.replace('DATASET', dataset))
    for file in tqdm(signal_files):
        signal = np.load(signal_dir.replace('DATASET', dataset) + file)
        fs = signal['fs']
        data = signal['data']
        duration = data.shape[1] / fs / 3600 # in hours
        all_durations.append(duration)
print(np.mean(all_durations), np.std(all_durations))