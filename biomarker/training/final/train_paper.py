import itertools
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import MageEncodingViT, MagePredViT
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from dataset import CrossFold, DataCollector_V2,  LabelHunter
from copy import deepcopy
from tqdm import tqdm
from ipdb import set_trace as bp
#import numpy as np
import argparse
import json 
import torch.nn.functional as F
from metrics import Metrics
import numpy as np 
torch.cuda.empty_cache()
import sys 

### A more updated version of train_antidep_with_rf_consistency (with filtered dataset)

class BinaryFocalLoss(nn.Module):
    ## AM2024: loss to focus on harder case examples
    def __init__(self, alpha=0.8, gamma=1.0, reduction='mean', logits=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.logits = logits

    def forward(self, inputs, targets):
        
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss) + 1e-8  # prevents nans when probability 0
        alpha_t = self.alpha * targets + (1 - targets) * (1 - self.alpha)
        F_loss = alpha_t * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def logger(writer, metrics, phase, epoch_index):

    for key, value in metrics.items():
        ## check for 2 class multiclass
        if type(value)!= float and len(value.shape) > 0 and value.shape[0] == 2:
            value = value[1]
        elif type(value)!= float and len(value.shape) > 0 and value.shape[0] > 2:
            bp()
        writer.add_scalar("%s/%s"%(phase, key), value, epoch_index)
    writer.flush()

def transform_modality(dataset):
    return [1 if item.startswith('hchs') else 2 if item == 'rf' else 0 for item in dataset]


def contrastive_loss(y_pred, y_true , margin=1.0):
    x_normalized = F.normalize(y_pred, p=2, dim=1) 
    cosine_similarity_matrix = torch.matmul(x_normalized, x_normalized.T) 
    positive_pairs = y_true * (1 - cosine_similarity_matrix) **2
    negative_pairs = (~y_true) * F.relu(cosine_similarity_matrix - margin) ** 2
    loss = (positive_pairs + negative_pairs).mean()
    return loss 

def get_contrastive_label(y_dict):
    y_label = [item.split('/')[-2] for item in y_dict['filepath']]
    y_label = np.array(y_label)
    return torch.tensor(y_label[:,None] == y_label[None, :])


parser = argparse.ArgumentParser(description='trainingLoop w/specified hyperparams')
parser.add_argument('-lr', type=float, default=4e-4, help='learning rate')
parser.add_argument('-w', type=str, default='1.0,10.0', help='respective class weights (comma-separated)')
parser.add_argument('--task', type=str, default='multiclass', help='multiclass or regression')
parser.add_argument('--num_classes', type=int, default=2, help='for multiclass')
parser.add_argument('--dataset', type=str, default='wsc', help='which dataset to train on')
parser.add_argument('-bs', type=int, default=16, help='batch size')
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--label', type=str, default='antidep', help="dep, antidep, or benzo")
parser.add_argument('--fold', type=int, default=0, help="for cross-validation")
parser.add_argument('--num_heads', type=int, default=3, help="for attention condensation")
parser.add_argument('--add_name', type=str, default="", help="adds argument to the experiment name")
parser.add_argument('--dropout', type=float, default=0.5, help="dropout regularization")
parser.add_argument('--mage_encoding_vit_model', action='store_true', default=False, help="use simon model")
parser.add_argument('--hidden_size', type=int, default=8, help="for SimonModel")
parser.add_argument('--fc2_size', type=int, default=32, help="for SimonModel")
parser.add_argument('--fc1_size', type=int, default=8, help="for SimonModel")
## for conv model
parser.add_argument('--feature_dim', type=int, default=16, help="for MageModel")
parser.add_argument('--num_token_heads', type=int, default=4, help="for MageModel")
parser.add_argument('--svd_reduce', type=int, default=0, help="for svd reduction")
parser.add_argument('--save_model', action='store_true', default=False, help="use simon model")
parser.add_argument('--training_resample', action='store_true', default=False, help="use simon model")
parser.add_argument('--log_every_n_step', type=int, default=100, help= '')
parser.add_argument('--tuning', action='store_true', default=False, help="tuning mode ")
parser.add_argument('--mage_pred_input', action='store_true', default=False, help="uses the mage image rather than the latent space ")
parser.add_argument('--use_gt_as_mage_pred', action='store_true', default=False, help="uses the eeg multitaper image rather than the mage pred ")
parser.add_argument('--mage_pred_vit_model', action='store_true', default=False, help="Mage Pred Based VIT MODEL ")
parser.add_argument('--mage_pred_kaiwen_model', action='store_true', default=False, help="Mage Pred Based Conv Model from Kaiwen MODEL ")
parser.add_argument('--pos_class_oversample', type=int, default=0, help="scaling factor for how many times to oversample the positive class")

parser.add_argument('--simul_separate_optimizers', action='store_true', default=False, help="for simul training, separates the losses into different optimizers")
parser.add_argument('--add_ssri_loss', action='store_true', default=False, help="for simul training, whether to add the ssri loss ")
parser.add_argument('--add_subtype_loss', action='store_true', default=False, help="for simul training, whether to add the subtype loss ")

parser.add_argument('--CLEANED_DATA', action='store_true', default=False, help=" use cleaned version of RF data, also with new mage version for all datasets ")

parser.add_argument('--separate_head', action='store_true', default=False, help=" adds a separate head for the simul task")
parser.add_argument('--num_layers_vit', type=int, default=4, help="for vit ")

parser.add_argument('--NOISE_PADDING', action='store_true', default=False, help=" instead of padding with 0s uses random noise, also can randomly add to beginning  ")

parser.add_argument('--tail_length_vit', type=int, default=-1, help="separates initial layers for VIT, -1 has no separation, 0 separates the embedding, >1 is how many layers of vit to separate ")
parser.add_argument('--num_tokens', type=int, default=1, help='how many classifier tokens to append to the transformer')
                    
parser.add_argument('--weight_decay', type=float, default=1e-4, help="l2 regularization")
parser.add_argument('--focal_loss', action='store_true', default=False)

parser.add_argument('--MAGE_INPUT_SIZE', type=int, default=150, help='either 150 or 270, specifies the mage input size limit')

parser.add_argument('--gamma',type=float, default=1.0) ## for focal loss
parser.add_argument('--alpha', type=float, default=0.8 ) ## for focal loss

parser.add_argument('--margin', type=float, default=1.0, help='contrastive loss margin')
parser.add_argument('--beta', type=float, default=0.5, help='ratio between losses, higher beta more contrastive loss')

parser.add_argument('--modality_input', action='store_true', default=False)
parser.add_argument('--age_input', action='store_true', default=False)
parser.add_argument('--sex_input', action='store_true', default=False)

parser.add_argument('--conditional_mask_ratio', type=float, default=0.5, help='ratio be')

parser.add_argument('--no_conv_proj', action='store_true', default=False)

parser.add_argument('--t5_demographics', action='store_true', default=False)
parser.add_argument('--t5_demographics_nomean', action='store_true', default=False)

parser.add_argument('--natural_reweight', action='store_true', default=False)

parser.add_argument('--black_oversample', type=int, default=0, help="for svd reduction")
parser.add_argument('--minority_pos_oversample', type=int, default=0, help="for svd reduction")

parser.add_argument('--balanced_medications_per_fold', action='store_true', default=False)
parser.add_argument('--training_resample_number', type=int, default=60, help="number of rf nights per patient in epoch")


torch.manual_seed(20)
np.random.seed(20)

folder_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD_2024/"
model_folder = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD_2024"

args = parser.parse_args()
lr = args.lr
task = args.task
dataset_name = args.dataset
label = args.label
num_classes = args.num_classes
num_class_name = f"class_{num_classes}"
batch_size = args.bs
debug = args.debug
num_steps = args.num_steps
#num_folds = args.num_folds
fold = args.fold
add_name = args.add_name

dpt = args.dropout
dpt_str = f"_{dpt}"
print("Label: ", label)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
args.device = device



if args.focal_loss:
    loss_fn = BinaryFocalLoss(alpha=args.alpha, gamma=args.gamma, logits=True)
else:
    if args.natural_reweight:
        loss_fn = nn.BCEWithLogitsLoss(reduce=False)
    else:
        loss_fn = nn.BCEWithLogitsLoss(reduce=True)

lr_warmup_prop = 0.1


### Going to create two sets of datasets and loaders, one for contrastive task and the other for antidep task


#### ANTIDEP TASK
aa = dataset_name.split("__")
train_datasets = [] 
external_val_dataset = []
for item in aa[0].split('_'):
    if item == '':
        continue
    assert item in ['hchs','mros1', 'mros2','rf', 'wsc', 'shhs1', 'shhs2', 'cfs', 'stages']
    train_datasets.append(item)
for item in aa[1].split('_'):
    if item == '':
        continue
    assert item in ['hchs','mros1', 'mros2','rf', 'wsc', 'shhs1', 'shhs2', 'cfs', 'stages']
    assert item not in train_datasets
    external_val_dataset.append(item)


dc = DataCollector_V2('mage',train_datasets + external_val_dataset, args.fold)
lh = LabelHunter(args.label)
cf = CrossFold(4,args, use_kaiwen_split= False, use_balanced_med_split=args.balanced_medications_per_fold, use_paper_split=True)


args.num_datasets = 1 

for dataset in dc.all_files:
    print(dataset)
    lbls = lh.get(dc.all_files[dataset], dataset)
    gndr = lh.get_gender(dc.all_files[dataset],dataset)
    age = lh.get_age(dc.all_files[dataset],dataset)
    race = lh.get_race(dc.all_files[dataset],dataset)
    
    cf.add(lbls, dc.get_filepath(dataset), is_rf=(dataset == 'rf'), is_external=(dataset in external_val_dataset), gender=gndr, age=age, race=race)

trainset, testset = cf.split(fold=args.fold, return_datasets=True)

print(trainset[0])

print('Length of trainset: ', len(trainset))
print('Length of testset: ', len(testset))

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True )
test_loader = DataLoader(testset, batch_size=batch_size*4, shuffle=False, num_workers=20)


### CONTRASTIVE TASK

train_datasets = ['rf'] 
external_val_dataset = []
dc2 = DataCollector_V2('mage' if not args.use_gt_as_mage_pred else 'mage_gt',train_datasets + external_val_dataset, args.fold)
lh2 = LabelHunter(args.label)
args2 = deepcopy(args)
args2.training_resample = False
cf2 = CrossFold(4,args2, use_kaiwen_split= not args.balanced_medications_per_fold, use_balanced_med_split=args.balanced_medications_per_fold)
args.num_datasets = 1 

for dataset in dc2.all_files:
    lbls = lh2.get(dc2.all_files[dataset], dataset)
    cf2.add(lbls, dc2.get_filepath(dataset), is_rf=(dataset == 'rf'), is_external=(dataset in external_val_dataset))

pids_with_multiple_labels = cf2.dataset.groupby('pid').filter(lambda x: x[args.label].nunique() > 1)['pid'].unique()
cf2.dataset = cf2.dataset[~cf2.dataset['pid'].isin(pids_with_multiple_labels)]
trainset2 = cf2.split_from_arr(cf.all_train_ids,val=False)
testset2 = cf2.split_from_arr(cf.all_test_ids,val=True)



if len(trainset2) > 0:
    train_loader2 = DataLoader(trainset2, batch_size=256, shuffle=True, num_workers=20, drop_last=False )
else:
    train_loader2 = []
if len(testset2) > 0:
    test_loader2 = DataLoader(testset2, batch_size=256, shuffle=True, num_workers=20)
else:
    test_loader2 = []




if (args.mage_encoding_vit_model):
    model = MageEncodingViT(args).to(device)
    print('Mage Encoding VIT Model')
elif (args.mage_pred_vit_model):
    assert args.mage_pred_input
    model = MagePredViT(args).to(device)
    print('Mage Encoding VIT Model')
elif (args.mage_pred_kaiwen_model):
    assert args.mage_pred_input
    model = resnet18_kaiwen(args).to(device)
    print('Mage Pred Kaiwen Model')
else:
    assert False 
    

n_model = model 
if not args.debug:
    n_model = torch.nn.DataParallel(n_model)

n_model = n_model.to(device)

contrastiveHead = nn.Sequential(nn.Linear(args.feature_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 32)).to(device)

print("Total Trainable Parameters: {:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

pretrained = '' 
exp_name = f"{args.label}_{dataset_name}_lr_{lr}_bs_{batch_size}_steps_{num_steps}_dpt_{args.dropout}_fold{fold}{pretrained}_heads{args.num_heads}_{add_name}_featuredim_{args.feature_dim}_numtokenheads_{args.num_token_heads}_{'trn_resmp' if args.training_resample else ''}_{'NOISE' if args.NOISE_PADDING else ''}_{'TAIL'+str(args.tail_length_vit) if args.tail_length_vit >=0 else ''}_wd_{str(round(args.weight_decay,4))}_{'focal'+str(round(args.gamma,2)) + str(round(args.alpha)) if args.focal_loss else 'bce'}"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder path '{folder_path}' created successfully.")
exp_event_path = os.path.join(folder_path, exp_name)

writer = SummaryWriter(log_dir=exp_event_path)

argstosave = deepcopy(vars(args))
argstosave.pop('device')
json.dump(argstosave, open(os.path.join(exp_event_path,'args.json'),'w'), indent=4)
with open(os.path.join(exp_event_path,'command.txt'),'w') as file: file.write(' '.join(sys.argv))


if not args.debug:
    train_pids = set()
    if type(trainset) == torch.utils.data.dataset.ConcatDataset:
        for item in trainset.datasets:
            train_pids.update(list(item.data))
    else:
        for train_batch in tqdm(train_loader):
            data, y_dict = train_batch
            train_pids.update(y_dict['filepath'])
    test_pids = set()
    for test_batch in tqdm(test_loader):
        data, y_dict = test_batch
        test_pids.update(y_dict['filepath'])
    train_pids2 = set()
    for train_batch in tqdm(train_loader2):
        data, y_dict = train_batch
        train_pids2.update(y_dict['filepath'])
    test_pids2 = set()
    for test_batch in tqdm(test_loader2):
        data, y_dict = test_batch
        test_pids2.update(y_dict['filepath'])
    
    if 'rf' in args.dataset:
        assert len(np.intersect1d(train_pids, test_pids2)) == 0
        assert len(np.intersect1d(train_pids2, test_pids)) == 0
        assert len(np.intersect1d(train_pids, test_pids)) == 0
        assert len(np.intersect1d(train_pids2, test_pids2)) == 0
    json.dump(list(train_pids), open(os.path.join(exp_event_path,'train.json'),'w'), indent=4)
    json.dump(list(test_pids), open(os.path.join(exp_event_path,'val.json'),'w'), indent=4)




assert task == 'binary' 


optimizer = optim.AdamW(
    itertools.chain(n_model.parameters(), contrastiveHead.parameters()),
    lr=0,
    weight_decay=args.weight_decay
)

scheduler = None

metrics = Metrics(args)

max_f1 = -1.0

val_losses = [] 




def get_loss_weights(y_batch):
    race = y_batch['mit_race']
    age = y_batch['mit_age'].to(args.device)
    age_bins = torch.tensor([20, 30, 40, 50, 60, 70, 80, 100], device=args.device)
    y_age_bin = torch.clip(torch.bucketize(age, age_bins, right=True),0, 7)
    gender = y_batch['mit_gender'] - 1
    weights = (gender_weights[gender.long()]) + (age_weights[y_age_bin.long()]) + (race_weights[race.long()])
    ## add noise to weights
    weights = weights + torch.randn_like(weights) * 0.1
    return weights.to(args.device)


def run_train_step(n_model, trainloader, trainloader2):
    try:
        X_batch, y_batch = next(trainloader)
    except:
        trainloader = iter(train_loader)
        X_batch, y_batch = next(trainloader)
    
    X_batch = X_batch.to(device)
        
    if 'rf' in args.dataset:
        try:    
            X_batch2, y_batch2 = next(trainloader2)
        except:
            trainloader2 = iter(train_loader2)
            X_batch2, y_batch2 = next(trainloader2)
        X_batch2 = X_batch2.to(device)
    
    
    
    
    
    n_model.train()
    with torch.enable_grad():
        y_label = y_batch['label']
        y_label = y_label.to(device)
        y_dataset = y_batch['dataset']
        y_dataset_emb = torch.tensor(transform_modality(y_dataset), dtype=torch.int64, device=device) if args.modality_input else None

        t5_emb = None 
        
        if 'rf' in args.dataset:
            y_label2 = get_contrastive_label(y_batch2)
            y_label2 = y_label2.to(device)
            if args.modality_input:
                y_dataset2 = torch.zeros(len(y_label2), dtype=torch.int64, device=device) + 2 # assuming rf is dataset 2
            else:
                y_dataset2 = None
            y_pred2 = n_model(x=X_batch2, return_embeddings=True, modality=y_dataset2)[0]
            
            y_pred2 = contrastiveHead(y_pred2)
            loss2 = args.beta * contrastive_loss(y_pred2, y_label2, args.margin)
        else:
            loss2 = torch.tensor(0.0, device=device)
            
        y_pred = n_model(x=X_batch, t5_emb=t5_emb, modality=y_dataset_emb) 

        loss = loss_fn(y_pred.squeeze(1), y_label.float())
        if args.natural_reweight:
            loss = loss * get_loss_weights(y_batch)
            loss = loss.mean() 
        
        loss = (1-args.beta) * loss
        
        
        ## Loss 1 is contrastive, so higher beta means more contrastive loss
        optimizer.zero_grad()
        all_loss = loss + loss2 
        all_loss.backward()
        torch.nn.utils.clip_grad_norm_(itertools.chain(n_model.parameters(), contrastiveHead.parameters()), max_norm=1.0)
        optimizer.step()
        metrics.fill_metrics(y_pred, y_label, y_dataset = y_dataset)
        
    return loss.item() , loss2.item() 

def run_val_epoch(n_model, epoch):
    
    n_model.eval()

    with torch.no_grad():
        running_loss = 0
        running_loss2 = 0
          
        for X_batch, y_batch in tqdm(test_loader):
            y_label = y_batch['label']
            y_gender = y_batch['mit_gender'].to(device)
            y_age = y_batch['mit_age'].to(device) 
            y_dataset = y_batch['dataset']
            y_dataset_emb = torch.tensor(transform_modality(y_dataset), dtype=torch.int64, device=device) if args.modality_input else None
            
            t5_emb = None 
            
            X_batch = X_batch.to(device)
            y_label = y_label.to(device)
            y_pred = n_model(x=X_batch, t5_emb=t5_emb, modality=y_dataset_emb) #gender=y_gender, age=y_age)
            loss = loss_fn(y_pred.squeeze(1), y_label.float())
            
            if args.natural_reweight:
                loss = loss * get_loss_weights(y_batch)
                loss = loss.mean() 
            
            running_loss += loss.item()
            metrics.fill_metrics(y_pred, y_label, y_dataset = y_dataset) # feed the raw scores, not thresh'd

        for X_batch, y_batch in tqdm(test_loader2):   
            y_label = get_contrastive_label(y_batch)

            X_batch = X_batch.to(device)
            y_label = y_label.to(device)
            if args.modality_input:
                y_dataset2 = torch.zeros(len(y_label), dtype=torch.int64, device=device) + 2 # assuming rf is dataset 2
            else:
                y_dataset2 = None
            
            y_pred = n_model(x=X_batch, return_embeddings=True, modality=y_dataset2)[0]
            y_pred = contrastiveHead(y_pred)
            loss = contrastive_loss(y_pred, y_label, args.margin)
                
            
            running_loss2 += loss.item()


        
        epoch_loss = running_loss / len(test_loader)
        if len(test_loader2) > 0:
            contr_loss = running_loss2 / len(test_loader2)
        else:
            contr_loss = 0.0
        
        
        computed_metrics = metrics.compute_and_log_metrics(loss=epoch_loss, hy_loss=contr_loss) #, loss3=gender_loss)
            
        logger(writer, computed_metrics, 'val', epoch)
        
        metrics.clear_metrics()   
        if args.save_model: 
            print('saving at epoch: ', epoch)
            model_name = exp_name + str(epoch) + ".pt"
            model_save_path = os.path.join(exp_event_path, model_name)
            torch.save(n_model.state_dict(), model_save_path)
        
        

step = 0


run_val_epoch(n_model, epoch=step)

trainloader = iter(train_loader)
trainloader2 = iter(train_loader2)

running_loss = 0
running_loss2 = 0
running_loss3 = 0 

for step in tqdm(range(num_steps)):
    step += 1 
    
    
    rl1, rl2 = run_train_step(n_model, trainloader, trainloader2)
    

    if step < args.num_steps * lr_warmup_prop:
        lr = args.lr * (step / (args.num_steps * lr_warmup_prop))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        if scheduler is None:
            if not args.tuning:
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, int(args.num_steps * (1-lr_warmup_prop)), eta_min=0)
            else:
                scheduler = None

        else:
            scheduler.step() 
        
    
    running_loss += rl1
    running_loss2 += rl2
    
    
    if step % args.log_every_n_step == 0:
        epoch_loss = running_loss / (args.log_every_n_step) #len(train_loader)
        epoch_loss_d = running_loss2/ (args.log_every_n_step) #len(train_loader)
        
        computed_metrics = metrics.compute_and_log_metrics(epoch_loss, hy_loss=epoch_loss_d) #loss3=epoch_loss3)
        logger(writer, computed_metrics, 'train', step )
        metrics.clear_metrics()
        
        running_loss = 0
        running_loss2 = 0
        
        run_val_epoch(n_model, epoch=step)