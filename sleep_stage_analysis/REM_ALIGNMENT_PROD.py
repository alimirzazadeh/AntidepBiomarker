"""
The goal of this script is to provide callable functions for REM Alignment and REM Degradatation
Can be called on a patient, expecting a folder of nights of sleep stages, 30 sec sampling frequecy

Does not require analytics repo
"""
    
    
from typing import List
import os 
import numpy as np 
import datetime 
from ipdb import set_trace as bp 
from skimage.util.shape import view_as_windows as viewW
from copy import deepcopy
import torch.nn.functional as F
import torch 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, normalized_mutual_info_score
from tqdm import tqdm 

class SleepStageAlignment:
    """@alimirz"""
    def __init__(self, source_folder: str, rem_reward=1.0, light_reward=0.0, deep_reward=0.0, MAX_EPOCHS=10, clip = False,
                 template_threshold = None, device_name="", bad_nights=None, MAX_SHIFT_PER_ITERATION = 30, plot=False, MAX_NIGHTS=None, 
                 start_date = None, end_date = None) -> None:
        
        self.TEMPLATE_THRESHOLD = template_threshold
        self.MAX_SHIFT_PER_ITERATION = MAX_SHIFT_PER_ITERATION
        self.rewards = [0, light_reward, light_reward, deep_reward, rem_reward]
        self.device_name = device_name
        self.CLIP = clip
        
        all_files = os.listdir(source_folder)
        if bad_nights is not None:
            print('before bad nights', len(all_files))
            all_files = [item for item in all_files if item not in bad_nights]
            print('after bad nights', len(all_files))
        
        if MAX_NIGHTS is not None:
            all_files = all_files[:min(len(all_files), MAX_NIGHTS)]
        
        all_dates = [datetime.datetime.strptime(item.split('/')[-1].split('.')[0][-len('2021_12_24'):], '%Y_%m_%d') for item in all_files]
        
        if start_date is not None: 
            all_dates = [item for item in all_dates if item > start_date]
        if end_date is not None:
            all_dates = [item for item in all_dates if item < end_date]
        
        date_sort_idx = np.argsort(all_dates)
        
        sorted_files = np.array(all_files)[date_sort_idx]
        
        all_stages = np.zeros((len(sorted_files), 2 * 60 * 15))
        
        for i, file in enumerate(sorted_files):
            data = np.load(os.path.join(source_folder,file))['data']
            
            data = self.strip_wake(data)
            all_stages[i, :len(data)] = data
        
        self.all_stages = all_stages
        
        self.all_stages = self.strided_indexing_roll(self.all_stages, np.zeros(self.all_stages.shape[0],dtype=int) + 2 * 60 )
        self.shifts = np.zeros(all_stages.shape[0])

        if plot:
            fig, axs = self.prepare_e_step_visual()
            self.e_step_visual(self.all_stages,axs)
        
        self.rem_agreement_score = self.calculate_agreement_score(self.all_stages, stage=4)
        print('Original REM Agreement Score: ', self.rem_agreement_score)
        print('Original Deep Agreement Score: ', self.calculate_agreement_score(self.all_stages, stage=3))
        
        repeating = True 
        epoch = 0
        all_stages = deepcopy(self.all_stages)
        while epoch < MAX_EPOCHS and repeating:
            self.template = self.e_step(all_stages, threshold_template=self.TEMPLATE_THRESHOLD, clip=self.CLIP)
            
            all_stages = self.m_step(all_stages, self.template)
            
            rem_agreement_score = self.calculate_agreement_score(all_stages, stage=4)
            deep_agreement_score = self.calculate_agreement_score(all_stages, stage=3)

            
            if rem_agreement_score - self.rem_agreement_score < 0.002:
                repeating = False 
            
            self.rem_agreement_score = rem_agreement_score
            epoch += 1
        
        print(f'Epoch {epoch} REM Agreement Score: ', rem_agreement_score)
        print(f'Epoch {epoch} Deep Agreement Score: ', deep_agreement_score)
        print('Average Shift: ', np.mean(np.abs(self.shifts)), np.mean(self.shifts))
        if plot:
            self.e_step_visual(all_stages,axs)
            self.axs = axs 
            self.fig = fig 
            
        self.rem_auroc_score = np.round(rem_agreement_score, 3)
        self.deep_auroc_score = np.round(deep_agreement_score, 3)
        self.avg_absolute_shift = np.round(np.mean(np.abs(self.shifts)) / 2, 1)
        self.avg_shift = np.round(np.mean(self.shifts) / 2, 1)
        
        self.rem_auprc_score = self.calculate_agreement_score_auprc(all_stages, stage=4)
        self.new_stages = all_stages
        
    """ Takes an expectation step for all_stages 
        Assumes threshold_template is a list of length 5
    """
    def e_step(self, all_stages, threshold_template=None, clip=False):
        output = [] 
        for i in range(5):
            template = np.mean(all_stages == i,0)
            if threshold_template is not None and threshold_template[i] is not None:
                if clip:
                    template[template >= threshold_template[i]] = threshold_template[i]
                else:
                    template = template >= threshold_template[i]
            
            output.append(template)
        return np.stack(output,0)

    def prepare_e_step_visual(self):
        fig,axs = plt.subplots(5, sharex=True, sharey=True)
        fig.suptitle(self.device_name)
        for i in range(5):
            axs[i].set_ylim(0, 1)
            axs[i].set_xlim(0, 2 * 60 * 15)
            axs[i].set_autoscale_on(False)
            axs[i].set_title(['wake','n1','n2','n3','rem'][i])
        return fig, axs 
    
    """ Visualized Template, requires axs with 5 subplots """
    def e_step_visual(self, all_stages, axs):
        for i in range(5):
            axs[i].plot(np.mean(all_stages == i,0))
            
    
    """ Does maximization step for all stages """
    def m_step(self, all_stages, template):
        ## THRESHOLD OR NOT 
        ## MAX SHIFT PER ITERATION: minutes
        MAX_SHIFT_PER_ITERATION = self.MAX_SHIFT_PER_ITERATION
        
        stages_onehot = F.one_hot(torch.tensor(all_stages, dtype=int), num_classes=5).float() #[:,1:]
        
        template = np.array([self.rewards]).transpose() * template
        
        template_onehot = torch.tensor(template).float()
        
        
        
        stages_onehot = stages_onehot.unsqueeze(1).permute(0, 1, 3, 2)  # Shape becomes (350, 1, 5, 1800)
        
        stages_onehot = F.pad(stages_onehot, (MAX_SHIFT_PER_ITERATION * 2, MAX_SHIFT_PER_ITERATION * 2, 0, 0), mode='circular')
        
        template_onehot = template_onehot.unsqueeze(0).unsqueeze(0)
        output = F.conv2d(stages_onehot, template_onehot, padding=0)
        
        shift_idx = -1 * (np.argmax(output,axis=3).flatten() - MAX_SHIFT_PER_ITERATION * 2)
        self.shifts += shift_idx.numpy()
        new_stages = self.strided_indexing_roll(all_stages, shift_idx)
        
        return new_stages
        
    def calculate_agreement_score(self, stages, stage=4):
        CORRELATION_VS_PROPORTION = 0 
        #TEMPLATE_VS_AVERAGE
        template = self.e_step(stages, threshold_template=None)
        
        labels = (stages == stage).flatten()
        pred = np.tile(template[stage], stages.shape[0])
        
        auroc = roc_auc_score(labels, pred)
    
        
        return auroc 

    def calculate_agreement_score_auprc(self, stages, stage=4):
        #TEMPLATE_VS_AVERAGE
        template = self.e_step(stages, threshold_template=None)
        
        labels = (stages == stage).flatten()
        pred = np.tile(template[stage], stages.shape[0])
        
        auprc = average_precision_score(labels, pred)
    
        
        return auprc 
    
    def calculate_agreement_score_mask_correlation(self, stages, stage=4, threshold=None):
        
        #TEMPLATE_VS_AVERAGE
        template = self.e_step(stages, threshold_template=None)
        template = template[stage]
        if threshold is not None: 
            template = template >= threshold 
        
        labels = stages == stage
        # template = np.tile(template, stages.shape[0])
        
        correlations = np.mean([np.corrcoef(template, labels[i])[0, 1] for i in range(stages.shape[0])])
        
        return correlations 

    def calculate_agreement_score_mask_agreement(self, stages, stage=4, threshold=None):
        
        #TEMPLATE_VS_AVERAGE
        template = self.e_step(stages, threshold_template=None)
        template = template[stage]
        if threshold is not None: 
            template = template >= threshold 
        
        labels = stages == stage
        # template = np.tile(template, stages.shape[0])
        
        correlations = np.mean([template * labels[i] for i in range(stages.shape[0])])
        
        return correlations / np.mean(labels)
    
    def calculate_agreement_score_mask_agreement_2(self, stages, stage=4, threshold=None):
        
        #TEMPLATE_VS_AVERAGE
        template = self.e_step(stages, threshold_template=None)
        template = template[stage]
        if threshold is not None: 
            template = template >= threshold 
        
        labels = stages == stage
        # template = np.tile(template, stages.shape[0])
        
        correlations = np.mean([np.sum(template * labels[i]) for i in range(stages.shape[0])])
        
        return correlations
    
    def calculate_agreement_score_mask_agreement_3(self, stages, stage=4, threshold=None):
        
        #TEMPLATE_VS_AVERAGE
        template = self.e_step(stages, threshold_template=None)
        template = template[stage]
        if threshold is not None: 
            template = template >= threshold 
        
        labels = stages == stage
        # template = np.tile(template, stages.shape[0])
        
        correlations = np.mean([np.sum(template * labels[i]) for i in range(stages.shape[0])])
        
        return correlations / np.mean(np.sum(labels,1))

    def calculate_agreement_score_mask_agreement_4(self, stages, stage=4, threshold=None):
        
        #TEMPLATE_VS_AVERAGE
        template = self.e_step(stages, threshold_template=None)
        template = template[stage]
        if threshold is not None: 
            template = template >= threshold 
        
        labels = stages == stage
        # template = np.tile(template, stages.shape[0])
        
        correlations = np.mean([np.sum(template * labels[i]) for i in range(stages.shape[0])])
        
        return correlations / np.sum(template)

    # normalized_mutual_info_score
    def calculate_agreement_score_mask_nmi(self, stages, threshold, stage=4):
        
        #TEMPLATE_VS_AVERAGE
        template = self.e_step(stages, threshold_template=None)
        template = template[stage] >= threshold
        
        labels = stages == stage
        # template = np.tile(template, stages.shape[0])
        
        return np.mean([normalized_mutual_info_score(template, labels[i]) for i in range(stages.shape[0])])
        normalized_mutual_info_score

    def calculate_agreement_score_mask_iou(self, stages, stage=4, threshold=None):
        
        #TEMPLATE_VS_AVERAGE
        template = self.e_step(stages, threshold_template=None)
        template = template[stage]
        if threshold is not None: 
            template = template >= threshold 
        
        labels = stages == stage
        # template = np.tile(template, stages.shape[0])
        
        correlations = np.mean([(template * labels[i]) / (np.sum(template) + np.sum(labels[i])) for i in range(stages.shape[0])])
        
        return correlations
    
    def calculate_agreement_score_mask_recall(self, stages, threshold, stage=4):
        
        template = self.e_step(stages, threshold_template=None)
        template = template[stage] >= threshold
        
        labels = stages == stage
        # template = np.tile(template, stages.shape[0])
        
        return np.mean([recall_score(template, labels[i],zero_division=0) for i in range(stages.shape[0])])

    def calculate_agreement_score_mask_precision(self, stages, threshold, stage=4):
        
        #TEMPLATE_VS_AVERAGE
        template = self.e_step(stages, threshold_template=None)
        template = template[stage] >= threshold
        
        labels = stages == stage
        # template = np.tile(template, stages.shape[0])
        
        return np.mean([precision_score(template, labels[i], zero_division=0) for i in range(stages.shape[0])])
    
    def strip_wake(self, data):
        idx = np.argwhere(data > 0)
        return data[idx[0][0]:idx[-1][0]]
    
    def strided_indexing_roll(self, a, r):
        if a is None:
            return None
        # Concatenate with sliced to cover all rolls
        a_ext = np.concatenate((a,a[:,:-1]),axis=1)

        # Get sliding windows; use advanced-indexing to select appropriate ones
        n = a.shape[1]
        return viewW(a_ext,(1,n))[np.arange(len(r)), (n-r)%n,0]
    
if __name__ == "__main__":
    SOURCE_DIR = '/Volumes/T7 Shield/DATASET/PROCESSED_V5/stage/'
    SAVE_DIR = '/Volumes/T7 Shield/DATASET/PROCESSED_V5/FIGURES_REM_ALIGNMENT_V10/'
    SAVE_FILE = 'individual_v2'
    OVERWRITE = False 
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if not os.path.isdir(os.path.join(SAVE_DIR, SAVE_FILE)):
        os.mkdir(os.path.join(SAVE_DIR, SAVE_FILE))
        
    
    bad_nights = np.load('/Volumes/T7 Shield/DATASET/PROCESSED_V5/bad_nights_v6.npz')
    
    for device in tqdm(os.listdir(SOURCE_DIR)):
        if not OVERWRITE and device+'.pdf' in os.listdir(os.path.join(SAVE_DIR, SAVE_FILE)):
            print('skipping', device)
            continue 
        print(device)
        
        device_path = os.path.join(SOURCE_DIR, device)
        
        try:
            ss1 = SleepStageAlignment(source_folder=device_path, device_name=device, bad_nights=bad_nights[device], MAX_SHIFT_PER_ITERATION=10, plot=True)
            ss2 = SleepStageAlignment(source_folder=device_path, device_name=device, bad_nights=bad_nights[device], template_threshold=[None, None, None, None, 0.3],MAX_SHIFT_PER_ITERATION=10, plot=True)
            ss3 = SleepStageAlignment(source_folder=device_path, device_name=device, bad_nights=bad_nights[device], template_threshold=[None, None, None, None, 0.2],MAX_SHIFT_PER_ITERATION=10, plot=True)
            ss4 = SleepStageAlignment(source_folder=device_path, device_name=device, bad_nights=bad_nights[device], template_threshold=[None, None, None, None, 0.2],MAX_SHIFT_PER_ITERATION=10, plot=True, clip=True)
            ss5 = SleepStageAlignment(source_folder=device_path, device_name=device, bad_nights=bad_nights[device], template_threshold=[None, None, None, None, 0.3],MAX_SHIFT_PER_ITERATION=10, plot=True, clip=True)
        except:
            print('ERROR WITH DEVICE: ', device)
            continue 
        

        """Combining the figures from each of the above into a single plot"""
        # Create a new figure to combine both
        combined_fig, ax_combined = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(32, 12))

        titles = [
            "10 MIN SHIFT, SMOOTH TEMPLATE",
            "10 MIN SHIFT, 30% REM TEMPLATE",
            "10 MIN SHIFT, 20% REM TEMPLATE",
            "10 MIN SHIFT, 20% REM CLIP SMOOTH TEMPLATE",
            "10 MIN SHIFT, 30% REM CLIP SMOOTH TEMPLATE",
        ]

        for i,ss in enumerate([ss1,ss2,ss3,ss4,ss5]):
            for j,ax in enumerate(ss.fig.get_axes()):
                for line in ax.get_lines():
                    ax_combined[j,i].plot(line.get_xdata(), line.get_ydata())
                    ax_combined[j,i].set_xticks([])
                ax_combined[j,i].set_ylabel(ax.get_title())
                ax_combined[j,i].set_ylim(0, 1)
                ax_combined[j,i].set_xlim(0, 2 * 60 * 15)
                ax_combined[j,i].set_autoscale_on(False)
            
            ax_combined[0,i].set_title(titles[i])
            ax_combined[1,i].set_title(f"REM AUROC: {ss.rem_auroc_score}  DEEP AUROC: {ss.deep_auroc_score}")
            ax_combined[2,i].set_title(f"Avg Abs Shift (minutes): {ss.avg_absolute_shift}  Avg Shift (minutes): {ss.avg_shift}")

            plt.close(ss.fig)
        combined_fig.suptitle(device)
        combined_fig.savefig(os.path.join(SAVE_DIR,SAVE_FILE,device+'.pdf'))
        