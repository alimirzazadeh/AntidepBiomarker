import torchmetrics
import torchmetrics.classification
import torch
from ipdb import set_trace as bp
import numpy as np 

def subtracted_list(list1, list2):
    subtracted = list()
    for item1, item2 in zip(list1, list2):
        item = item1 - item2
        subtracted.append(item)
    return subtracted

class AvgDepScore(torchmetrics.Metric): 
    def __init__(self, dist_sync_on_step=False): 
        super().__init__()
        self.add_state("zung_scores", default=[], dist_reduce_fx="cat")

    def update(self, raw_predictions, raw_labels: torch.tensor):
        self.pred_classes = torch.argmax(raw_predictions, dim=1)
        self.zung_scores.extend(self.pred_classes * raw_labels) # raw labels of those predicted positive

    def compute(self):
        #'positive_scores' name is misleading: again, it's the raw zung scores of the patients for whom the model predicted positive, regardless of whether they really are or not
        positive_scores = [zung_score.item() for zung_score in self.zung_scores if zung_score != 0]
        if len(positive_scores) == 0:
            return 0
        else:
            return sum(positive_scores) / len(positive_scores)

    def reset(self):
        self.zung_scores = [] # reset to empty list

class CustomMAE(torchmetrics.Metric):
    def __init__(self, threshold, dist_sync_on_step=False):
        super().__init__()
        self.threshold = threshold
        self.add_state("preds_above_threshold", default=[], dist_reduce_fx="cat") # note: predictions of the guys with actual labels >=5, regardless of prediction itself
        self.add_state("labels_above_threshold", default=[], dist_reduce_fx="cat")
    
    def update(self, raw_predictions, raw_labels: torch.tensor):
        idx_above_threshold = [idx for idx, label in enumerate(raw_labels) if label >= self.threshold]
        self.preds_above_threshold.extend([raw_predictions[idx] for idx in idx_above_threshold])
        self.labels_above_threshold.extend([raw_labels[idx] for idx in idx_above_threshold])
        #bp()
    
    def compute(self):
        raw_errors = subtracted_list(self.preds_above_threshold, self.labels_above_threshold)
        abs_errors = [abs(error) for error in raw_errors]
        return sum(abs_errors) / len(abs_errors)

    def reset(self):
        self.preds_above_threshold = []
        self.labels_above_threshold = []

class Metrics():
    def __init__(self, args):
        self.args = args 
        
        self.used_keys = {}
        # if self.args.label == 'antidep':
        #      self.num_classes = 4
        # elif self.args.task == 'multiclass':
        #     self.num_classes = 2
        # elif self.args.target == 'hy':
        #     self.num_classes = 8
        # elif self.args.target == 'age':
        #     self.num_classes = 2
        self.num_classes = args.num_classes
        
        
        dataset_names = [] 
        for word in args.dataset.split('_'):
            if word == "train" or word == "val" or word == "all" or word == '':
                continue 
            if word[-1].isdigit():
                word = word[:-1]
            
            # if word[-1].isdigit():
            #     word = word[:-1]
            
            if word not in dataset_names:
                if word.startswith('rf'):
                    dataset_names.append('rf')
                else:
                    dataset_names.append(word)
        
        self.dataset_names = dataset_names
        self.num_datasets = len(dataset_names)

            
        self.init_metrics()
    
    def multiclass_prediction(self, predictions, labels, threshold=0):
        predictions = torch.argmax(predictions,dim=1)
        # if self.args.label=="dep":
        #     labels = (labels >= threshold).int()
        return predictions, labels
    def regression_prediction(self, predictions, labels):
        threshold = 36.0 # or change to 50...but otherwise v few positive labels
        predictions = torch.where(predictions < threshold, torch.tensor(0), torch.tensor(1))
        labels = torch.where(labels < threshold, torch.tensor(0), torch.tensor(1))
        return predictions, labels
    def age_prediction(self, predictions, labels, threshold=5):
        output = torch.abs(predictions - labels) < threshold
        return output.int(), torch.zeros_like(output).to(self.args.device) + 1
    def nearest_hy_score(self, predictions, labels):
        hy_scores = torch.tensor([0, 1, 1.5, 2, 2.5, 3, 4, 5])
        output = []
        for score in predictions:
            output.append(torch.argmin(torch.abs(hy_scores - score.item())).item())
        return torch.tensor(output).to(self.args.device), labels
        
    def init_metrics(self):
        
        if self.args.task == 'multiclass':
            if self.num_datasets > 1:
                self.classifier_metrics_dict = {
                    "acc": torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes, average = None).to(self.args.device),
                    "kappa": torchmetrics.CohenKappa(task='multiclass',num_classes=self.num_classes).to(self.args.device),
                    "f1": torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average = None).to(self.args.device),
                    "auroc": torchmetrics.AUROC(task="multiclass", num_classes=self.num_classes, average = None).to(self.args.device),
                    "prec": torchmetrics.Precision(task = "multiclass",num_classes=self.num_classes, average = None).to(self.args.device),
                    "recall": torchmetrics.Recall(task = "multiclass",num_classes=self.num_classes, average = None).to(self.args.device),
                }
                for name in self.dataset_names:
                    self.classifier_metrics_dict.update(
                        {
                            "acc_{0}".format(name): torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes, average = None).to(self.args.device),
                            "kappa_{0}".format(name): torchmetrics.CohenKappa(task='multiclass',num_classes=self.num_classes).to(self.args.device),
                            "f1_{0}".format(name): torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average = None).to(self.args.device),
                            "auroc_{0}".format(name): torchmetrics.AUROC(task="multiclass", num_classes=self.num_classes, average=None).to(self.args.device),
                            "prec_{0}".format(name): torchmetrics.Precision(task = "multiclass",num_classes=self.num_classes, average = None).to(self.args.device),
                            "recall_{0}".format(name): torchmetrics.Recall(task = "multiclass",num_classes=self.num_classes, average = None).to(self.args.device),
                            # "prec_{0}".format(name): torchmetrics.Precision(task = "binary").to(self.args.device),
                            # "recall_{0}".format(name): torchmetrics.Recall(task = "binary").to(self.args.device)
                        } 
                    )
                return 
            else:
                self.classifier_metrics_dict = {
                    "acc": torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes, average = None).to(self.args.device),
                    "kappa": torchmetrics.CohenKappa(task='multiclass',num_classes=self.num_classes).to(self.args.device),
                    "prec": torchmetrics.Precision(task = "multiclass",num_classes=self.num_classes, average = None).to(self.args.device),
                    "recall": torchmetrics.Recall(task = "multiclass",num_classes=self.num_classes, average = None).to(self.args.device),
                    "f1": torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average = None).to(self.args.device), 
                    "auroc": torchmetrics.AUROC(task="multiclass", num_classes=self.num_classes, average=None).to(self.args.device),
                } 
                return 
            

                
                
                # "confusion": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes).to(self.args.device)
            # if self.args.label=="dep":
            #     classifier_metrics_dict["avg_dep_score"] = AvgDepScore().to(self.args.device)
        elif self.args.task == 'binary':
            if self.num_datasets > 1:
                self.classifier_metrics_dict = {
                    "acc": torchmetrics.Accuracy(task='binary').to(self.args.device),
                    "kappa": torchmetrics.CohenKappa(task='binary').to(self.args.device),
                    "f1": torchmetrics.F1Score(task="binary").to(self.args.device),
                    "auroc": torchmetrics.AUROC(task="binary").to(self.args.device),
                    "prec": torchmetrics.Precision(task = "binary").to(self.args.device),
                    "recall": torchmetrics.Recall(task = "binary").to(self.args.device)
                }
                for name in self.dataset_names:
                    self.classifier_metrics_dict.update(
                        {
                            "acc_{0}".format(name): torchmetrics.Accuracy(task='binary').to(self.args.device),
                            "kappa_{0}".format(name): torchmetrics.CohenKappa(task='binary').to(self.args.device),
                            "f1_{0}".format(name): torchmetrics.F1Score(task="binary").to(self.args.device),
                            "auroc_{0}".format(name): torchmetrics.AUROC(task="binary").to(self.args.device),
                            "prec_{0}".format(name): torchmetrics.Precision(task = "binary").to(self.args.device),
                            "recall_{0}".format(name): torchmetrics.Recall(task = "binary").to(self.args.device)
                        } 
                    )
                if self.args.label == 'simul':
                    arr = ['subtype','ssri'] if self.args.add_ssri_loss else ['subtype']
                    for which in arr:
                        NUM_SUBTYPES = 2
                        self.classifier_metrics_dict.update({
                        "f1_{0}".format(which): torchmetrics.F1Score(task="multilabel", num_labels=NUM_SUBTYPES if which == 'subtype' else 5, average=None, ignore_index=-1).to(self.args.device),
                        "auroc_{0}".format(which): torchmetrics.AUROC(task="multilabel", num_labels=NUM_SUBTYPES if which == 'subtype' else 5, average=None, ignore_index=-1).to(self.args.device),
                        })
                return 
            else:
                self.classifier_metrics_dict = {
                    "acc": torchmetrics.Accuracy(task='binary').to(self.args.device),
                    "kappa": torchmetrics.CohenKappa(task='binary').to(self.args.device),
                    "prec": torchmetrics.Precision(task = "binary").to(self.args.device),
                    "recall": torchmetrics.Recall(task = "binary").to(self.args.device),
                    "f1": torchmetrics.F1Score(task="binary").to(self.args.device),
                    "auroc": torchmetrics.AUROC(task="binary").to(self.args.device)
                } 
                return 
        elif self.args.task == 'multilabel':
            ## For the sake of simplicity, we will just show rn and non-rf for each label 
            assert self.args.label in ['antidep_6_class', 'antidep_14_class']
            if self.args.label == 'antidep_6_class':
                self.label_names = ['SSRI', 'SNRI', 'TCA', 'SARI', 'Mirtazapine', 'Buproprion']
            elif self.args.label == 'antidep_14_class':
                self.label_names = ['ESCITALOPRAM', 'CITALOPRAM', 'FLUOXETINE', 'SERTRALINE', 'PAROXETINE', 'VENLAFAXINE', 'DESVENLAFAXINE', 'IMIPRAMINE', 'DULOXETINE', 'VORTIOXETINE', 'MIRTAZAPINE', 'BUPROPRION', 'NORTRYPTILINE', 'TRAZODONE']
            
            classifier_metrics_dict = {
                "acc": torchmetrics.Accuracy(task='binary').to(self.args.device),
                "kappa": torchmetrics.CohenKappa(task='binary').to(self.args.device),
                "prec": torchmetrics.Precision(task = "binary").to(self.args.device),
                "recall": torchmetrics.Recall(task = "binary").to(self.args.device),
                "f1": torchmetrics.F1Score(task="binary").to(self.args.device),
                "auroc": torchmetrics.AUROC(task="binary").to(self.args.device)
            } 
            for dataset in ['rf','belt']:
                for i, label in enumerate(self.label_names):
                    classifier_metrics_dict[f"{dataset}_prec_{label}"] = torchmetrics.Precision(task = "binary").to(self.args.device)
                    classifier_metrics_dict[f"{dataset}_recall_{label}"] = torchmetrics.Recall(task = "binary").to(self.args.device)
                    classifier_metrics_dict[f"{dataset}_f1_{label}"] = torchmetrics.F1Score(task="binary").to(self.args.device)
                    classifier_metrics_dict[f"{dataset}_auroc_{label}"] = torchmetrics.AUROC(task="binary").to(self.args.device)
                
            

            
        
        elif self.args.task == 'regression':
            classifier_metrics_dict = {

                "mae": torchmetrics.MeanAbsoluteError().to(self.args.device),

                #"expvar": torchmetrics.ExplainedVariance().cuda()#.to(self.args.device),

                "r2": torchmetrics.R2Score().to(self.args.device),

                "pearson": torchmetrics.PearsonCorrCoef().to(self.args.device),

                "spearman": torchmetrics.SpearmanCorrCoef().to(self.args.device)

            }
            if self.args.label=="dep" and self.args.dataset=="shhs2":
                classifier_metrics_dict["5_mae"] = CustomMAE(threshold=5).to(self.args.device)
                classifier_metrics_dict["7_mae"] = CustomMAE(threshold=7).to(self.args.device)
            
            for name in self.dataset_names:
                classifier_metrics_dict.update({
                    f"r2_{name}": torchmetrics.R2Score().to(self.args.device),
                    f"pearson_{name}": torchmetrics.PearsonCorrCoef().to(self.args.device),
                    f"spearman_{name}": torchmetrics.SpearmanCorrCoef().to(self.args.device)
                })
                                                    
        else:
            classifier_metrics_dict = {}
        self.classifier_metrics_dict = classifier_metrics_dict
        
    def fill_metrics(self, raw_predictions, raw_labels, y_dataset=None):
        
        if self.args.task == 'regression':
            self.classifier_metrics_dict["r2"].update(raw_predictions, raw_labels)
            self.classifier_metrics_dict["mae"].update(raw_predictions, raw_labels)
            self.classifier_metrics_dict["pearson"].update(raw_predictions, raw_labels.float())
            self.classifier_metrics_dict["spearman"].update(raw_predictions, raw_labels.float())
            #self.classifier_metrics_dict["expvar"].update(raw_predictions, raw_labels)
            if self.args.label == "dep" and self.args.dataset == "shhs2":
                self.classifier_metrics_dict["5_mae"].update(raw_predictions, raw_labels)
                self.classifier_metrics_dict["7_mae"].update(raw_predictions, raw_labels)
            
            for i,name in enumerate(self.dataset_names):
                if type(y_dataset[0]) != str:
                    batch_mask = y_dataset == i
                elif type(y_dataset[0]) == str:
                    batch_mask = np.array(y_dataset) == name 
                else:
                    raise NotImplementedError
                
                labels = raw_labels[batch_mask]
                if len(labels) == 0:
                    continue 
                raw_preds = raw_predictions[batch_mask]
                self.classifier_metrics_dict[f"r2_{name}"].update(raw_preds, labels)
                self.classifier_metrics_dict[f"pearson_{name}"].update(raw_preds, labels.float())
                self.classifier_metrics_dict[f"spearman_{name}"].update(raw_preds, labels.float())
        elif self.args.task == 'multiclass' and self.args.num_classes > 2:
            #bp()
            predictions, labels = self.multiclass_prediction(raw_predictions, raw_labels)
            self.classifier_metrics_dict["acc"].update(predictions, labels)
            self.classifier_metrics_dict["kappa"].update(predictions, labels)
            self.classifier_metrics_dict["prec"].update(predictions, labels)
            self.classifier_metrics_dict["recall"].update(predictions, labels)
            self.classifier_metrics_dict["f1"].update(predictions, labels)
            self.classifier_metrics_dict["auroc"].update(raw_predictions, labels) # need raw_preds for threshold, but need the binary labels

            self.used_keys["acc"] = True 
            self.used_keys["kappa"] = True 
            self.used_keys["f1"] = True 
            self.used_keys["auroc"] = True 


            if self.num_datasets > 1 and y_dataset is not None:
                for i,name in enumerate(self.dataset_names):
                    if type(y_dataset[0]) != str:
                        batch_mask = y_dataset == i
                    elif type(y_dataset[0]) == str:
                        batch_mask = np.array(y_dataset) == name 
                    else:
                        raise NotImplementedError
                    
                    labels = raw_labels[batch_mask]
                    if len(labels) == 0:
                        continue
                    
                    if name not in self.used_keys:
                        self.used_keys["acc_{0}".format(name)] = True 
                        self.used_keys["kappa_{0}".format(name)] = True 
                        self.used_keys["f1_{0}".format(name)] = True 
                        self.used_keys["auroc_{0}".format(name)] = True 
                        self.used_keys["prec_{0}".format(name)] = True 
                        self.used_keys["recall_{0}".format(name)] = True 
                    
                    predictions, _ = self.multiclass_prediction(raw_predictions, raw_labels)
                    
                    preds = predictions[batch_mask]
                    # raw_pred = torch.sigmoid(raw_predictions)[batch_mask]
                    # preds = (raw_pred > 0.5).squeeze(1)
                    
                    self.classifier_metrics_dict["acc_{0}".format(name)].update(preds, labels)
                    self.classifier_metrics_dict["kappa_{0}".format(name)].update(preds, labels)
                    self.classifier_metrics_dict["f1_{0}".format(name)].update(preds, labels)
                    self.classifier_metrics_dict["prec_{0}".format(name)].update(preds, labels)
                    self.classifier_metrics_dict["recall_{0}".format(name)].update(preds, labels)
                    self.classifier_metrics_dict["auroc_{0}".format(name)].update(raw_predictions[batch_mask], labels) 


            # self.classifier_metrics_dict["f1_c"].update(predictions, labels)
            # self.classifier_metrics_dict["confusion"].update(predictions, labels)
            #bp()
            # if False and self.args.label == "dep":
                #could also just feed the following the argmax'd predictions
                # self.classifier_metrics_dict["avg_dep_score"].update(raw_predictions, raw_labels) # needs the raw zung scores to do the calculation
        elif self.args.task == 'binary' or (self.args.task == 'multiclass' and self.args.num_classes == 2):

            
            if self.args.label == 'simul': ## 3 different tasks, so need to account
                
                self.classifier_metrics_dict["f1_subtype"].update(torch.sigmoid(raw_predictions[1]) > 0.5, raw_labels[1])
                self.classifier_metrics_dict["auroc_subtype"].update(raw_predictions[1], raw_labels[1])
                if self.args.add_ssri_loss:
                    self.classifier_metrics_dict["f1_ssri"].update(torch.sigmoid(raw_predictions[2]) > 0.5, raw_labels[2])
                    self.classifier_metrics_dict["auroc_ssri"].update(raw_predictions[2], raw_labels[2])
                raw_predictions = raw_predictions[0]
                raw_labels = raw_labels[0]
            
            if self.args.num_classes == 2:
                bp()
                raw_pred = torch.softmax(raw_predictions)[:,1]
            else:
                raw_pred = torch.sigmoid(raw_predictions)
            predictions = (raw_pred > 0.5).squeeze(1)
            labels = raw_labels
            self.classifier_metrics_dict["acc"].update(predictions, labels)
            self.classifier_metrics_dict["kappa"].update(predictions, labels)
            self.classifier_metrics_dict["prec"].update(predictions, labels)
            self.classifier_metrics_dict["recall"].update(predictions, labels)
            #self.classifier_metrics_dict["prec"].update(predictions, labels)
            #self.classifier_metrics_dict["recall"].update(predictions, labels)
            self.classifier_metrics_dict["f1"].update(predictions, labels)
            self.classifier_metrics_dict["auroc"].update(raw_pred, labels) # need raw_preds for threshold, but need the binary labels
            
           
            
            self.used_keys["acc"] = True 
            self.used_keys["kappa"] = True 
            self.used_keys["f1"] = True 
            self.used_keys["auroc"] = True 
            
            if self.num_datasets > 1 and y_dataset is not None:
                for i,name in enumerate(self.dataset_names):

                    if type(y_dataset[0]) != str:
                        batch_mask = y_dataset == i
                    elif type(y_dataset[0]) == str:
                        batch_mask = np.array(y_dataset) == name 
                    else:
                        raise NotImplementedError
                    
                    labels = raw_labels[batch_mask]
                    if len(labels) == 0:
                        continue 

                    if name not in self.used_keys:
                        self.used_keys["acc_{0}".format(name)] = True 
                        self.used_keys["kappa_{0}".format(name)] = True 
                        self.used_keys["f1_{0}".format(name)] = True 
                        self.used_keys["auroc_{0}".format(name)] = True 
                        self.used_keys["prec_{0}".format(name)] = True 
                        self.used_keys["recall_{0}".format(name)] = True 
                        
                    raw_pred = torch.sigmoid(raw_predictions)[batch_mask]
                    preds = (raw_pred > 0.5).squeeze(1)
                    
                    self.classifier_metrics_dict["acc_{0}".format(name)].update(preds, labels)
                    self.classifier_metrics_dict["kappa_{0}".format(name)].update(preds, labels)
                    self.classifier_metrics_dict["f1_{0}".format(name)].update(preds, labels)
                    self.classifier_metrics_dict["prec_{0}".format(name)].update(preds, labels)
                    self.classifier_metrics_dict["recall_{0}".format(name)].update(preds, labels)
                    self.classifier_metrics_dict["auroc_{0}".format(name)].update(raw_pred, labels) 
        elif self.args.task == 'multilabel':
            raw_pred = torch.sigmoid(raw_predictions)
            predictions = (raw_pred > 0.5)
            labels = raw_labels
            
            # classifier_metrics_dict = {
            #     "acc": torchmetrics.Accuracy(task='binary').to(self.args.device),
            #     "kappa": torchmetrics.CohenKappa(task='binary').to(self.args.device),
            #     "prec": torchmetrics.Precision(task = "binary").to(self.args.device),
            #     "recall": torchmetrics.Recall(task = "binary").to(self.args.device),
            #     "f1": torchmetrics.F1Score(task="binary").to(self.args.device),
            #     "auroc": torchmetrics.AUROC(task="binary").to(self.args.device)
            # } 
            # for dataset in ['rf','belt']:
            #     for i, label in enumerate(self.label_names):
            #         classifier_metrics_dict[f"{dataset}_prec_{label}"] = torchmetrics.Precision(task = "binary").to(self.args.device)
            #         classifier_metrics_dict[f"{dataset}_recall_{label}"] = torchmetrics.Recall(task = "binary").to(self.args.device)
            #         classifier_metrics_dict[f"{dataset}_f1_{label}"] = torchmetrics.F1Score(task="binary").to(self.args.device)
            #         classifier_metrics_dict[f"{dataset}_auroc_{label}"] = torchmetrics.AUROC(task="binary").to(self.args.device)
                
            
            self.classifier_metrics_dict["acc"].update(predictions, labels)
            self.classifier_metrics_dict["kappa"].update(predictions, labels)
            self.classifier_metrics_dict["prec"].update(predictions, labels)
            self.classifier_metrics_dict["recall"].update(predictions, labels)
            self.classifier_metrics_dict["f1"].update(predictions, labels)
            self.classifier_metrics_dict["auroc"].update(raw_pred, labels) # need raw_preds for threshold, but need the binary labels
            
            
            if y_dataset is not None:
                batch_mask = np.array(y_dataset) == 'rf' 
                labels1 = raw_labels[batch_mask]
                labels2 = raw_labels[~batch_mask]
                raw_pred1 = raw_pred[batch_mask]
                raw_pred2 = raw_pred[~batch_mask]
                predictions1 = predictions[batch_mask]
                predictions2 = predictions[~batch_mask]
                
                for i, label in enumerate(self.label_names):
                    if labels1.sum() > 0:
                        self.classifier_metrics_dict[f"rf_prec_{label}"].update(predictions1[:,i], labels1[:,i])
                        self.classifier_metrics_dict[f"rf_recall_{label}"].update(predictions1[:,i], labels1[:,i])
                        self.classifier_metrics_dict[f"rf_f1_{label}"].update(predictions1[:,i], labels1[:,i])
                        self.classifier_metrics_dict[f"rf_auroc_{label}"].update(raw_pred1[:,i], labels1[:,i])
                    if labels2.sum() > 0:
                        self.classifier_metrics_dict[f"belt_prec_{label}"].update(predictions2[:,i], labels2[:,i])
                        self.classifier_metrics_dict[f"belt_recall_{label}"].update(predictions2[:,i], labels2[:,i])
                        self.classifier_metrics_dict[f"belt_f1_{label}"].update(predictions2[:,i], labels2[:,i])
                        self.classifier_metrics_dict[f"belt_auroc_{label}"].update(raw_pred2[:,i], labels2[:,i])
                
        
    def compute_and_log_metrics(self, loss=0, hy_loss=0, loss3=0, classwise_prec_recall=True, classwise_f1=True):
        if self.args.task == 'multiclass':
            #prec = self.classifier_metrics_dict["prec"].compute()
            #rec = self.classifier_metrics_dict["recall"].compute()
            #f1 = self.classifier_metrics_dict["f1_macro"].compute()
            metrics = {}
            for item in self.used_keys:
                metrics[item] = self.classifier_metrics_dict[item].compute() 
            
            metrics['auroc_overall'] = np.nanmean([metrics[item] for item in self.used_keys if item.startswith('auroc_')])
            
            if loss3 != 0:
                metrics["loss_three"] = loss3
            if hy_loss != 0:
                metrics["loss_disc"] = hy_loss
            if loss != 0:  
                if type(loss) == np.ndarray:
                    if self.args.label == 'simul':
                        metrics["loss_bce"] = loss[0]
                        metrics["loss_subtype"] = loss[1]
                        if self.args.add_ssri_loss:
                            metrics["loss_ssri"] = loss[2]
                    else:
                        raise Exception("add correct loss labels for multiloss task")
                else:
                    metrics["loss_bce"] = loss
            
            # "hy_loss": hy_loss,
            # "mae": self.classifier_metrics_dict["mae"].compute(),
            # "expvar": self.classifier_metrics_dict["expvar"].compute(),
            # "r2": self.classifier_metrics_dict["r2"].compute(),
            # metrics = {
                
            #     "total_loss": loss, 
            #     "acc": self.classifier_metrics_dict["acc"].compute(),
            #     #"r2": self.classifier_metrics_dict["r2"].compute(),
            #     #"mae": self.classifier_metrics_dict["mae"].compute(),
            #     # "kappa": self.classifier_metrics_dict["kappa"].compute(),
            #     # "neg_precision": neg_prec,
            #     # "pos_precision": pos_prec,
            #     # "neg_recall": neg_rec,
            #     # "pos_recall": pos_rec,
            #     "confusion": self.classifier_metrics_dict["confusion"].compute(),
            #     "f1_macro": self.classifier_metrics_dict["f1_macro"].compute(),
            #     "auroc": self.classifier_metrics_dict["auroc"].compute()
                
            #     }
            # if classwise_f1:
            #         f1_c = self.classifier_metrics_dict["f1_c"].compute()
            #         for i in range(self.num_classes):
            #             metrics[str(i) + "_f1"] = f1_c[i]
            # # if classwise_prec_recall:
            # #     for i in range(self.num_classes):
            # #         metrics[str(i) + "_precision"] = prec[i]
            # #         metrics[str(i) + "_recall"] = rec[i]

            # if False and self.args.label == "dep":
            #     metrics["avg_dep_score"] = self.classifier_metrics_dict["avg_dep_score"].compute()
        elif self.args.task == 'binary':
            metrics = {}
            for item in self.used_keys:
                metrics[item] = self.classifier_metrics_dict[item].compute() 
            
            metrics['auroc_overall'] = np.mean([metrics[item].cpu().item() for item in self.used_keys if (item.startswith('auroc_') and 'subtype' not in item and 'ssri' not in item)])
            
            if hy_loss != 0:
                metrics["loss_disc"] = hy_loss
            
            if loss3 != 0:
                metrics["loss_three"] = loss3
            
            if type(loss) == np.ndarray:
                if self.args.label == 'simul':
                    metrics["loss_bce"] = loss[0]
                    metrics["loss_subtype"] = loss[1]
                    if self.args.add_ssri_loss:
                        metrics["loss_ssri"] = loss[2]
                else:
                    raise Exception("add correct loss labels for multiloss task")
            elif loss != 0:  
                metrics["loss_bce"] = loss
            
            if self.args.label == 'simul':
                subtype_f1_results = self.classifier_metrics_dict['f1_subtype'].compute() 
                subtype_auroc_results = self.classifier_metrics_dict['auroc_subtype'].compute() 
                for i in range(2):
                    metrics['f1_subtype_{0}'.format(str(i))] = subtype_f1_results[i]
                    metrics['auroc_subtype_{0}'.format(str(i))] = subtype_auroc_results[i]
                
                
                if self.args.add_ssri_loss:
                    ssri_f1_results = self.classifier_metrics_dict['f1_ssri'].compute() 
                    ssri_auroc_results = self.classifier_metrics_dict['auroc_ssri'].compute() 
                    for i in range(5):
                        metrics['f1_ssri_{0}'.format(str(i))] = ssri_f1_results[i]
                        metrics['auroc_ssri_{0}'.format(str(i))] = ssri_auroc_results[i]


        elif self.args.task == 'multilabel':
            metrics = {}
            for item in self.classifier_metrics_dict.keys():
                metrics[item] = self.classifier_metrics_dict[item].compute() 
            
            # metrics['auroc_overall'] = np.mean([metrics[item].cpu().item() for item in self.used_keys if (item.startswith('auroc_')])
            
            metrics["loss_bce"] = loss
            
 
        elif self.args.task == 'regression':
            metrics = {
                "total_loss": loss,
                "r2": self.classifier_metrics_dict["r2"].compute(),
                "mae": self.classifier_metrics_dict["mae"].compute(),
                "pearson": self.classifier_metrics_dict["pearson"].compute(),
                "spearman": self.classifier_metrics_dict["spearman"].compute()
            }
            if self.args.label=="dep" and self.args.dataset=="shhs2":
                metrics["5_mae"] = self.classifier_metrics_dict["5_mae"].compute()
                metrics["7_mae"] = self.classifier_metrics_dict["7_mae"].compute()
            for i,name in enumerate(self.dataset_names):
                metrics[f"r2_{name}"] = self.classifier_metrics_dict[f"r2_{name}"].compute()
                metrics[f"pearson_{name}"] = self.classifier_metrics_dict[f"pearson_{name}"].compute()
                metrics[f"spearman_{name}"] = self.classifier_metrics_dict[f"spearman_{name}"].compute()
        else:
            metrics = {} 
            metrics["loss"] = loss
        # self.logger(writer, metrics, phase, epoch)
        return metrics 
    
    def clear_metrics(self):
            for _, val in self.classifier_metrics_dict.items():
                val.reset()
            self.used_keys = {}

# include mae above 5 and mae above 7


class FairMetrics():
    def __init__(self, args):
        self.args = args 
        self.init_metrics()
    def init_metrics(self):
        self.age_bins = torch.tensor(np.array([20, 30, 40, 50, 60, 70, 80, 100])).to(self.args.device)
        self.race_bins = np.arange(6)
        self.gender_bins = np.arange(2)
        
        self.fair_metrics_dict = {
            f"auroc_age_{str(i)}": torchmetrics.AUROC(task="binary").to(self.args.device) for i in range(len(self.age_bins))
        }
        for i in self.race_bins:
            self.fair_metrics_dict[f"auroc_race_{str(i)}"] = torchmetrics.AUROC(task="binary").to(self.args.device)
        for i in self.gender_bins:
            self.fair_metrics_dict[f"auroc_gender_{str(i+1)}"] = torchmetrics.AUROC(task="binary").to(self.args.device)

    def fill_metrics(self, raw_predictions, y_batch):
        # mit_age = np.digitize(y_batch['mit_age'].cpu().numpy(), bins=self.age_bins, right=True)
        ## check if age > 7 exists, if so then do the bucketizing
        if y_batch['mit_age'].max() > 8:
            mit_age = torch.clip(torch.bucketize(y_batch['mit_age'].to(self.args.device), self.age_bins, right=True),0, 7)
        else:
            mit_age = y_batch['mit_age']
        
        for i in torch.unique(y_batch['mit_race']):
            self.fair_metrics_dict[f"auroc_race_{str(int(i.item()))}"].update(raw_predictions[y_batch['mit_race']==i], y_batch['label'][y_batch['mit_race']==i].to(self.args.device))
        for i in torch.unique(mit_age):
            self.fair_metrics_dict[f"auroc_age_{str(int(i.item()))}"].update(raw_predictions[mit_age==i], y_batch['label'].to(self.args.device)[mit_age==i])
        for i in torch.unique(y_batch['mit_gender']):
            self.fair_metrics_dict[f"auroc_gender_{str(int(i.item()))}"].update(raw_predictions[y_batch['mit_gender']==i], y_batch['label'][y_batch['mit_gender']==i].to(self.args.device))
        
    def compute_and_log_metrics(self, loss=0. , loss2=0.):
        metrics = {}
        for item in self.fair_metrics_dict:
            if len(self.fair_metrics_dict[item].target) < 1 or torch.cat(self.fair_metrics_dict[item].target).unique().shape[0] < 2:
                print('caught and skipping')
                continue
            metrics[item] = self.fair_metrics_dict[item].compute() 
            
            
        if loss != 0:  
            metrics["loss_bce"] = loss
        if loss2 != 0:  
            metrics["loss_fairness"] = loss2
            
        return metrics 
    
    def clear_metrics(self):
        for _, val in self.fair_metrics_dict.items():
            val.reset()
        self.used_keys = {}