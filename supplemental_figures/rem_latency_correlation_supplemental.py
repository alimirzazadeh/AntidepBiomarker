import numpy as np
import scipy.stats
import os 
from tqdm import tqdm
import scipy
import os
import matplotlib.pyplot as plt 
import math
import pandas as pd 
from ipdb import set_trace as bp 

np.random.seed(20)

END_SIZE = 256

PRED = True 

SO_START = 0 
SO_END = 1

BETA_START = 16
BETA_END = 32

WHICH_STAGE = [1,2]
MAXVAL = 120 #90 #300 #300 #90


def show_kernel_effect(idxx, plot=True):
    if PRED:
        img = pred_shhs_allimginputs[idxx,:END_SIZE]
        img_st = pred_shhs_img_st[idxx] #((img - pred_shhs_imgmean[:128,None]) / pred_shhs_imgstd[:128,None]) + 5
        msk = np.isin(pred_shhs_all_stages_raw[idxx], WHICH_STAGE) #+ (pred_shhs_all_stages_raw[idxx] == 3) + (pred_shhs_all_stages_raw[idxx] == 1) 
        img_light = img_st[:,msk]
        # img_st = (img - imgmean[:128,None]) / imgstd[:128,None]
    else:
        img_gt = gt_shhs_allimginputs[idxx,:END_SIZE]
        img_st_gt = gt_shhs_img_st[idxx]
        msk_gt = np.isin(gt_shhs_all_stages_raw[idxx], WHICH_STAGE) #+ (gt_shhs_all_stages_raw[idxx] == 3) + (gt_shhs_all_stages_raw[idxx] == 1) 
        img_light_gt = img_st_gt[:,msk_gt]
    

    if True:
        rem_indices = np.argwhere(pred_shhs_all_stages_raw[idxx] == 4)
        slp_indices = np.argwhere(pred_shhs_all_stages_raw[idxx] > 0)
        n3_indices = np.argwhere(pred_shhs_all_stages_raw[idxx] == 3)
        n3_latency = n3_indices[0][0] if n3_indices.size > 0 else None
        
        rem_latency = rem_indices[0][0] if rem_indices.size > 0 else None
        slp_latency = slp_indices[0][0] if slp_indices.size > 0 else None
        if rem_latency is not None and slp_latency is not None:
            rem_latency_slp_latency = rem_latency - slp_latency
        else:
            rem_latency_slp_latency = np.nan
    elif False:
        rem_latency_slp_latency = np.sum((pred_shhs_all_stages_raw[idxx] == 0)[:])
    

    
    return None, rem_latency_slp_latency, None, n3_latency, img_light if PRED else img_light_gt

def process_stages(stages):
    stages[stages < 0] = 0
    stages[stages > 5] = 0
    stages = stages.astype(int)
    # mapping = np.array([0, 2, 2, 3, 3, 1, 0, 0, 0, 0], int)
    mapping = np.array([0, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0], np.int64)
    return mapping[stages]
    
    



columns = ['dataset', 'pid', 'antidep', 'dosage', 'subtype', 'pred', 'filename']
dfs = None


FOLD = {}

fold = 0 


df_shhs = pd.read_csv('../data/inference_v6emb_3920_all.csv')
df_shhs['filename'] = df_shhs['filename'].apply(lambda x: x.split('/')[-1])
df_shhs = df_shhs.groupby('filename').first().reset_index()
df_shhs.set_index('filename', inplace=True)

PRED_PATH = '/data/netmit/sleep_lab/filtered/MAGE/XXX_new/mage/cv_0/'
GT_PATH = '/data/netmit/wifall/ADetect/data/XXX_new/c4_m1_multitaper/'
pred_all_patients = []
for dataset in ['shhs1', 'shhs2','mros1','mros2','cfs','wsc']:
    pred_all_patients.extend([PRED_PATH.replace('XXX',dataset) + item for item in np.sort(os.listdir(PRED_PATH.replace('XXX',dataset)))])
gt_all_patients = []
for dataset in ['shhs1', 'shhs2','mros1','mros2','cfs','wsc']:
    gt_all_patients.extend([GT_PATH.replace('XXX',dataset) + item for item in np.sort(os.listdir(GT_PATH.replace('XXX',dataset).replace('cfs_new','cfs')))])


gt_total_files = len(gt_all_patients)
pred_total_files = len(pred_all_patients)

gt_shhs_ll = [] 
gt_shhs_age = [] 
gt_shhs_allfilepaths = [] 
gt_shhs_allimginputs = np.zeros((gt_total_files,256,150*8))
gt_shhs_all_stages = np.zeros((gt_total_files,5,150*8))
gt_shhs_all_stages_raw = np.zeros((gt_total_files,150*8))

pred_shhs_ll = [] 
pred_shhs_age = [] 
pred_shhs_allfilepaths = [] 
pred_shhs_allimginputs = np.zeros((pred_total_files,256,150*8))
pred_shhs_all_stages = np.zeros((pred_total_files,5,150*8))
pred_shhs_all_stages_raw = np.zeros((pred_total_files,150*8))


for PRED, all_patients, shhs_ll, shhs_age, shhs_allfilepaths, shhs_allimginputs, shhs_all_stages, shhs_all_stages_raw in [[False, gt_all_patients, gt_shhs_ll, gt_shhs_age, gt_shhs_allfilepaths, gt_shhs_allimginputs, gt_shhs_all_stages, gt_shhs_all_stages_raw],[True, pred_all_patients, pred_shhs_ll, pred_shhs_age, pred_shhs_allfilepaths, pred_shhs_allimginputs, pred_shhs_all_stages, pred_shhs_all_stages_raw]]:
    if not PRED:
        continue 
    unsucessful = [] 
    for i in tqdm(range(len(all_patients))):
        filepath = all_patients[i]
        file = np.load(all_patients[i])
        if PRED:
            data = file['pred']
        else:
            data = file['data']

        dataset = filepath.split('/')[-4] if PRED else filepath.split('/')[-3]
        file = np.load('/data/netmit/wifall/ADetect/data/'+dataset.replace('cfs_new','cfs')+'/stage/'+filepath.split('/')[-1])
        
        stage = process_stages(file['data'])[::int(30 * file['fs'])]            
        
        try:
            if PRED:
                shhs_ll.append(df_shhs.loc[filepath.split('/')[-1],'label'])
            else:
                if 'shhs1' in filepath:
                    shhs_ll.append(df_shhs.loc[filepath.split('/')[-1],'label'])
                else:
                    shhs_ll.append(df_shhs.loc[filepath.split('/')[-1],'label'])
        except:
            unsucessful.append(i)
            print('skipping',i)
            continue

        if data.shape[-1] < 4*60*2:
            unsucessful.append(i)
            print('skipping2',i)
            continue
        if data.shape[-1] > 10*60*2:
            data = data[:,:10*60*2]
        
        if np.all(data == 0):
            unsucessful.append(i)
            print('skipping3',i)
            continue
        
        shhs_allimginputs[i,:,:data.shape[-1]] = data
        
        if len(stage) > 150*8:
            stage = stage[:150*8]
            
            # shhs_all_stages_raw[i,:] = stage[:150*8]
        
        shhs_all_stages_raw[i,:len(stage)] = stage
        for stg in range(5):
            shhs_all_stages[i, stg, :len(stage)] = stage == stg
        
        shhs_allfilepaths.append(filepath)

    if len(unsucessful) > 0:
        print('removing unsuccessful')

        if not PRED:
            val_mask = [i not in unsucessful for i in range(gt_shhs_allimginputs.shape[0])]
        else:
            val_mask = [i not in unsucessful for i in range(pred_shhs_allimginputs.shape[0])]
        print(sum(val_mask))
        print(unsucessful)
        if not PRED:
            gt_shhs_allimginputs = gt_shhs_allimginputs[val_mask]
            gt_shhs_all_stages = gt_shhs_all_stages[val_mask]
            gt_shhs_all_stages_raw = gt_shhs_all_stages_raw[val_mask]
        else:
            pred_shhs_allimginputs = pred_shhs_allimginputs[val_mask]
            pred_shhs_all_stages = pred_shhs_all_stages[val_mask]
            pred_shhs_all_stages_raw = pred_shhs_all_stages_raw[val_mask]



if PRED:
    pred_shhs_allimginputs = np.where(pred_shhs_allimginputs != 0, pred_shhs_allimginputs, np.nan)

    pred_shhs_img_st = pred_shhs_allimginputs

else:
    gt_shhs_allimginputs = np.where(gt_shhs_allimginputs != 0, gt_shhs_allimginputs, np.nan)
    gt_shhs_allimginputs2=gt_shhs_allimginputs
    gt_shhs_img_st = gt_shhs_allimginputs2



n3_latency = [] 
rem_latencynolog = [] 
all_lights = []
all_lights_gt = [] 
for i in tqdm(range(len(pred_shhs_ll))):
    _, rl, _, n3lat,lights = show_kernel_effect(i,plot=False)

    rem_latencynolog.append(rl)

    
    if lights.shape[0] != 256 or lights.shape[1] < MAXVAL:
        backup = np.zeros((256,MAXVAL))
        backup_gt = np.zeros((256,MAXVAL))
        
        backup[:] = np.nan
        backup_gt[:] = np.nan
        
        if PRED:
            backup[:,:lights.shape[1]] = lights
            all_lights.append(backup)
        else:
            backup_gt[:,:lights.shape[1]] = lights
            all_lights_gt.append(backup_gt)
    else:
        if PRED:
            all_lights.append(lights[:,:MAXVAL])
        else:
            all_lights_gt.append(lights[:,:MAXVAL])
        
    if n3lat is not None:
        n3_latency.append(n3lat)


yraw = np.array(pred_shhs_ll)


rem_latencynolog = np.array(rem_latencynolog)
rem_latency = rem_latencynolog

if PRED:
    all_lights = np.stack(all_lights)
else:
    all_lights_gt = np.stack(all_lights_gt)

if PRED:
    del pred_shhs_allimginputs
else:
    del gt_shhs_allimginputs



def calculate_cohort_metrics(PRED=False, NORM_POWER=True, LOGNORM_LATENCY=True, CONTROL=False, TREATMENT=True):
    
    
    if PRED:
        global all_lights
        all_lights = np.array(all_lights)
        valbeta = np.nanmean(np.nansum(all_lights[:,BETA_START*8:BETA_END*8,:] ,1) ,1)
        valswa = np.nanmean(np.nansum(all_lights[:,SO_START*8:SO_END*8,:] ,1) ,1)
        nan_vals = np.all(np.isnan(all_lights),axis=(1,2))
    else:
        global all_lights_gt
        all_lights_gt = np.array(all_lights_gt)
        valbeta = np.nanmean(np.nansum(all_lights_gt[:,BETA_START*8:BETA_END*8,:] ,1) ,1)
        valswa = np.nanmean(np.nansum(all_lights_gt[:,SO_START*8:SO_END*8,:] ,1) ,1)
        nan_vals = np.all(np.isnan(all_lights_gt),axis=(1,2))

    if LOGNORM_LATENCY:
        val_latency = np.log(rem_latency)

    
    bad_vals = np.isnan(valbeta) + np.isnan(valswa) + np.isnan(val_latency) + np.isinf(val_latency)
    if CONTROL and not TREATMENT:
        bad_vals = nan_vals + bad_vals + (yraw == 1)
    elif TREATMENT and not CONTROL:
        bad_vals = nan_vals + bad_vals + (yraw == 0)
    
    
    val_latency = val_latency[~bad_vals]


    if LOGNORM_LATENCY:
        val_latency = (val_latency - np.nanmean(val_latency)) / np.nanstd(val_latency)


    valbeta = valbeta[~bad_vals]
    valswa = valswa[~bad_vals]

    if NORM_POWER:
        valbeta = (valbeta - np.mean(valbeta)) / np.std(valbeta)
        valswa = (valswa - np.mean(valswa)) / np.std(valswa)
    
    
    print('SWA Beta correlation: ', scipy.stats.pearsonr(valbeta, valswa).statistic)
    print('SWA Latency correlation: ', scipy.stats.pearsonr(val_latency, valswa).statistic)
    print('Beta Latency correlation: ', scipy.stats.pearsonr(val_latency, valbeta).statistic)
    print('SWA+Beta Latency correlation: ', scipy.stats.pearsonr(val_latency, valbeta + valswa).statistic)
    
    return valbeta, valswa, val_latency


valbeta, valswa, val_latency = calculate_cohort_metrics(PRED=True, NORM_POWER=True, LOGNORM_LATENCY=True, CONTROL=False, TREATMENT=True)

text_y = 2.0
fig,axs = plt.subplots(3,2,figsize=(8,12),sharex=True, sharey=True)
valbeta, valswa, val_latency = calculate_cohort_metrics(PRED=True, NORM_POWER=True, LOGNORM_LATENCY=True, CONTROL=False, TREATMENT=True)
axs[0,0].scatter(valswa, val_latency,alpha=0.4)
pearson_r = scipy.stats.pearsonr(valswa, val_latency)
pval = pearson_r.pvalue
pearson_r = pearson_r.statistic
axs[0,0].text(-3.4,text_y,f'Pearson r={pearson_r:.2f}\np={pval:.2e}',fontsize=10)
axs[0,0].set_title(f'Antidepressant (N={len(valswa)})\nSO ({SO_START}-{SO_END}Hz)')
axs[1,0].scatter(valbeta, val_latency,alpha=0.4)
pearson_r = scipy.stats.pearsonr(valbeta, val_latency)
pval = pearson_r.pvalue
pearson_r = pearson_r.statistic
axs[1,0].text(-3.4,text_y,f'Pearson r={pearson_r:.2f}\np={pval:.2e}',fontsize=10)
axs[1,0].set_title(f'Beta ({BETA_START}-{BETA_END}Hz)')
axs[2,0].scatter(valswa+valbeta, val_latency,alpha=0.4)
pearson_r = scipy.stats.pearsonr(valswa+valbeta, val_latency)
pval = pearson_r.pvalue
pearson_r = pearson_r.statistic
axs[2,0].text(-3.4,text_y,f'Pearson r={pearson_r:.2f}\np={pval:.2e}',fontsize=10)
axs[2,0].set_title('SO + Beta')
print(scipy.stats.pearsonr(valswa, val_latency))
axs[2,0].set_xlim(-3.5,3.5)
axs[2,0].set_ylim(-3,3)
axs[0,0].set_ylabel('REM Latency, LogNorm')
axs[1,0].set_ylabel('REM Latency, LogNorm')
axs[2,0].set_ylabel('REM Latency, LogNorm')
axs[0,1].set_ylabel('REM Latency, LogNorm')
axs[1,1].set_ylabel('REM Latency, LogNorm')
axs[2,1].set_ylabel('REM Latency, LogNorm')
axs[0,0].set_yticks([])
axs[0,0].set_xticks([])
axs[2,0].set_xlabel('Predicted Power, Norm')
axs[0,0].set_xlabel('Predicted Power, Norm')
axs[1,0].set_xlabel('Predicted Power, Norm')
axs[2,1].set_xlabel('Predicted Power, Norm')
axs[0,1].set_xlabel('Predicted Power, Norm')
axs[1,1].set_xlabel('Predicted Power, Norm')
valbeta, valswa, val_latency = calculate_cohort_metrics(PRED=True, NORM_POWER=True, LOGNORM_LATENCY=True, CONTROL=True, TREATMENT=False)
axs[0,1].scatter(valswa, val_latency,alpha=0.05)
pearson_r = scipy.stats.pearsonr(valswa, val_latency)
pval = pearson_r.pvalue
pearson_r = pearson_r.statistic
axs[0,1].text(-3.4,text_y,f'Pearson r={pearson_r:.2f}\np={pval:.2e}',fontsize=10)
axs[0,1].set_title(f'Control (N={len(valswa)})\nSO ({SO_START}-{SO_END}Hz)')
axs[1,1].scatter(valbeta, val_latency,alpha=0.05)
pearson_r = scipy.stats.pearsonr(valbeta, val_latency)
pval = pearson_r.pvalue
pearson_r = pearson_r.statistic
axs[1,1].text(-3.4,text_y,f'Pearson r={pearson_r:.2f}\np={pval:.2e}',fontsize=10)
axs[1,1].set_title(f'Beta ({BETA_START}-{BETA_END}Hz)')
axs[2,1].scatter(valswa+valbeta, val_latency,alpha=0.05)
pearson_r = scipy.stats.pearsonr(valswa+valbeta, val_latency)
pval = pearson_r.pvalue
pearson_r = pearson_r.statistic
axs[2,1].text(-3.4,text_y,f'Pearson r={pearson_r:.2f}\np={pval:.2e}',fontsize=10)
axs[2,1].set_title('SO + Beta')
print(scipy.stats.pearsonr(valswa, val_latency))
axs[2,1].set_xlim(-3.5,3.5)
axs[2,1].set_ylim(-3,3)

fig.suptitle('Predicted Powers vs REM Latency')
stage_string = '_'.join([str(i) for i in WHICH_STAGE])

plt.savefig(f'pred_powers_vs_rem_latency_{MAXVAL}epochs_{stage_string}_{SO_START}_{SO_END}hz_{BETA_START}_{BETA_END}hz.png')


print('done')
