import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from ipdb import set_trace as bp
from tqdm import tqdm

start_stop_dict = np.load('/data/netmit/sleep_lab/rf/start_end_idx_padded.npz')
df = pd.read_csv('../data/inference_v6emb_3920_all.csv')

def get_bad_signal_time(filename):
    pid = filename.split('_')[0]
    data = np.load(f'/data/netmit/sleep_lab/rf/snr/{pid}/{filename}')['data']
    start_idx = start_stop_dict[filename][0]
    end_idx = start_stop_dict[filename][1]
    stage = np.load(f'/data/netmit/sleep_lab/rf/stage/{pid}/{filename}')['data']

    sleep_idx = list(stage > 0).index(True)
    wake_idx = list(stage > 0)[::-1].index(True)
    
    sleep_idx = sleep_idx * 150
    wake_idx = wake_idx * 150
    return (data[start_idx + sleep_idx:end_idx - wake_idx] < 0.5).sum() / (5 * 60)

output = {} 
for filename in tqdm(df[df['pid'] == '1007']['filename'].values):
    filename = filename.split('/')[-1]
    bad_signal_time = get_bad_signal_time(filename)
    output[filename] = bad_signal_time

#     raise ValueError("If using all scalar values, you must pass an index")
# ValueError: If using all scalar values, you must pass an index
output = pd.DataFrame(output.items(), columns=['filename', 'bad_signal_time'])
output.to_csv('../data/bad_signal_time_1007.csv', index=False)