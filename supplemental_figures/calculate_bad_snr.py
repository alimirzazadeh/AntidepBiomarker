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

def fill_ends_with_1(arr):
    ones = np.where(arr == 1)[0]
    if len(ones) == 0:
        return arr
    arr[:ones[0]] = 1
    arr[ones[-1] + 1:] = 1
    return arr
def get_bad_signal_time(filename, plot=False ):
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
    end_idx = start_idx + sleep_idx + 4 * 60 * 60 * 5
    v1 = (data_med[start_idx + sleep_idx + int(0 * 60 * 60 * 5):end_idx] > 0.5)
    
    data2 = (data > 0.5).astype(float)
    data2_med = medfilt(data2, kernel_size=150*4*5 + 1)
    v2 = (data2_med[start_idx + sleep_idx + int(0 * 60 * 60 * 5):end_idx] > 0.5)
    
    # v1 = fill_ends_with_1(v1)
    v2 = fill_ends_with_1(v2)
    if plot:
        fig, ax = plt.subplots(3, figsize=(10, 5))
        ax[0].plot(data[start_idx + sleep_idx + int(0.25 * 60 * 60 * 5):end_idx])
        ax[1].scatter(np.arange(start_idx + sleep_idx + int(0.25 * 60 * 60 * 5), end_idx), v1)
        ax[2].scatter(np.arange(start_idx + sleep_idx + int(0.25 * 60 * 60 * 5), end_idx), v2)
        ax[0].set_title('Original Signal')
        ax[1].set_title('Median Filtered')
        ax[2].set_title('Binary Filtered')
        # plt.savefig(f'../data/plots/snr_comparison_{filename}.png')
        plt.show()
    return (v1 == 0).sum() / (5 * 60), (v2 == 0).sum() / (5 * 60)



output = {}
for filename in tqdm(df[df['pid'] == '1007']['filename'].values):
    filename = filename.split('/')[-1]
    bad_signal_time, bad_signal_time_binary = get_bad_signal_time(filename, plot=False)
    output[filename] = [bad_signal_time, bad_signal_time_binary]

output = pd.DataFrame.from_dict(output, orient='index', columns=['bad_signal_time', 'bad_signal_time_binary'])
output = output.reset_index()
output.columns = ['filename', 'bad_signal_time', 'bad_signal_time_binary']
output.to_csv('../data/bad_signal_time_1007.csv', index=False)