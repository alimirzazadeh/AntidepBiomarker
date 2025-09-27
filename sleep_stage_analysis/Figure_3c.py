import sys 
import os 
import numpy as np
from ipdb import set_trace as bp
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
from REM_ALIGNMENT_PROD import SleepStageAlignment


df = pd.read_csv('../data/inference_v6emb_3920_all.csv')
df_other = df[df['dataset'] == 'rf']

df_other['pred'] = 1/(1 + np.exp(-df_other['pred'])) 
df_other['filename'] = df_other['filename'].apply(lambda x: x.split('/')[-1])

df_taxonomy = pd.read_csv('../data/antidep_taxonomy_all_datasets_v6.csv')
df_taxonomy = df_taxonomy[['filename','taxonomy']]

df = pd.merge(df_other, df_taxonomy, on='filename', how='inner')

# bp() 

if __name__ == "__main__":
    cmap = ListedColormap(['black', '#ff9966'])
    fig,ax = plt.subplots(2,3,figsize=(20,5), sharex=True)
    
    SOURCE_DIR = '/Volumes/T7 Shield/DATASET/PROCESSED_V5/stage/'
    SAVE_DIR = '/Volumes/T7 Shield/DATASET/PROCESSED_V5/FIGURES_REM_ALIGNMENT_V10/'
    bad_nights = np.load('/Volumes/T7 Shield/DATASET/PROCESSED_V5/bad_nights_v6.npz')
    
    SAVE_FILE = 'individual_v2'
    OVERWRITE = False 
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if not os.path.isdir(os.path.join(SAVE_DIR, SAVE_FILE)):
        os.mkdir(os.path.join(SAVE_DIR, SAVE_FILE))
        
    
    
    control_devices = ['1047','NIHXB175YAGF7 - Control', '1001'] #05
    antidep_devices = ['10105','1089_V2','1111'] 
    
    drug = {'1111':'SNRI - Venlafaxine', '1089_V2':'SSRI - Fluoxetine', '10105':'SSRI - Escitalopram'}
    notdrug = {'1047':'Control, Cycles=3', 'NIHXB175YAGF7 - Control':'Control, Cycles=4', '1001':'Control, Cycles=5'}
    for device in tqdm(os.listdir(SOURCE_DIR)):
        if device not in control_devices and device not in antidep_devices:
            continue 
        
        row = 1 if device in control_devices else 0

        print(device)
        
        device_path = os.path.join(SOURCE_DIR, device)
        
        ss1 = SleepStageAlignment(source_folder=device_path, device_name=device, bad_nights=bad_nights[device], MAX_SHIFT_PER_ITERATION=10, plot=False)
        arr = ss1.new_stages[:, int(0 * 60 * 2): int(12.5 * 60 * 2)]
        #self.new_stages
        if row == 1:
            axs = ax[1, control_devices.index(device)]
            axs.set_title(notdrug[device])
            axs.set_xlabel('Hours from Sleep Onset')
        else:
            axs = ax[0, antidep_devices.index(device)]
            axs.set_title(drug[device])
            
            
        axs.pcolormesh(arr==4, cmap=cmap)
        axs.set_ylim(0,arr.shape[0])
        axs2 = axs.twinx()
        axs2.plot(np.mean(arr==4,0), color='white',alpha=0.7)
        
        axs.set_xticks(np.arange(60, 61 + 10 * 60 * 2, 2 * 60))
        axs.set_xticklabels(np.arange(0, 11))
        
        
        axs.set_yticks([])
        axs2.set_yticks([])
        
    fig.supylabel('Consecutive Nights')
    fig.subplots_adjust(left=0.04)
    fig.savefig(os.path.join('Figure_3c.png'), dpi=300)
    