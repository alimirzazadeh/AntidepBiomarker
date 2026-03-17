import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from ipdb import set_trace as bp
from tqdm import tqdm
import matplotlib.pyplot as plt
## medfilt
from scipy.signal import medfilt
start_stop_dict = np.load('/data/netmit/sleep_lab/rf/start_end_idx_padded.npz')
df = pd.read_csv('../data/inference_v6emb_3920_all.csv')

def fill_ends_with_1(arr, right=True, left=True):
    ones = np.where(arr == 1)[0]
    if len(ones) == 0:
        return arr
    if left:
        arr[:ones[0]] = 1
    if right:
        arr[ones[-1] + 1:] = 1
    return arr
def get_bad_signal_time(filename, plot=False ):
    ## skip the first: 0min, 15min, 30min, 1hr 
    ## total min: 3hrs, 4hrs, 6hrs, 9hrs 
    ## trim end left, trim end right, trim none, trim both 
    
    pid = filename.split('_')[0]
    data = np.load(f'/data/netmit/sleep_lab/rf/snr/{pid}/{filename}')['data']
    start_idx = start_stop_dict[filename][0]
    end_idx = start_stop_dict[filename][1]
    stage = np.load(f'/data/netmit/sleep_lab/rf/stage/{pid}/{filename}')['data']
    
    data_med = medfilt(data, kernel_size=150*4*5 + 1)
    sleep_idx = list(stage > 0).index(True)
    wake_idx = list(stage > 0)[::-1].index(True)
    
    sleep_idx = sleep_idx * 150
    # wake_idx = wake_idx * 150
    # end_idx - wake_idx
    
    versions = {} 
    for end_idx in [3, 4, 6, 9]:
        for skip_first in [0, 0.25, 0.5, 1]:
            for trim_setting in ['left', 'right', 'both', 'none']:
                v1 = (data_med[start_idx + sleep_idx + int(skip_first * 60 * 60 * 5):start_idx + sleep_idx + end_idx * 60 * 60 * 5] > 0.5)
                v1 = fill_ends_with_1(v1, right=trim_setting in ['right', 'both'], left=trim_setting in ['left', 'both'])
                versions[f'{skip_first}h_{end_idx}h_TRIM_{trim_setting}'] = (v1 == 0).sum() / (5 * 60)
    if plot:
        fig, ax = plt.subplots(3, figsize=(10, 5))
        for key, value in versions.items():
            ax[0].plot(data[start_idx + sleep_idx + int(0.25 * 60 * 60 * 5):end_idx])
            ax[1].scatter(np.arange(start_idx + sleep_idx + int(0.25 * 60 * 60 * 5), end_idx), v1)
            ax[2].scatter(np.arange(start_idx + sleep_idx + int(0.25 * 60 * 60 * 5), end_idx), v2)
            ax[0].set_title('Original Signal')
            ax[1].set_title('Median Filtered')
            ax[2].set_title('Binary Filtered')
            # plt.savefig(f'../data/plots/snr_comparison_{filename}.png')
            plt.show()
    return versions



output = {}
for filename in tqdm(df[df['pid'].isin(['1007', '1022', 'NIHYM875FLXFF', '1033'])]['filename'].values):
    filename = filename.split('/')[-1]
    versions = get_bad_signal_time(filename, plot=False)
    output[filename] = versions

output = pd.DataFrame.from_dict(output, orient='index')
output.reset_index(names='filename').to_csv('../data/bad_signal_time_4_patients.csv', index=False)