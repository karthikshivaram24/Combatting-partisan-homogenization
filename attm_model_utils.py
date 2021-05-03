import torch
import numpy as np
from Scripts.utils.general_utils import timer
from attm_metrics import calculate_scores, calculate_scores_single

@timer
def evaluate_mt(model,dataloader,device=torch.device('cuda:1')):
    """
    """
    model.eval()
    y1_preds = []
    y2_preds = []
    y1_true =[]
    y2_true = []
    which_cluster = []
    with torch.no_grad():
        for bid, (x1,x2,y1,y2,wc) in enumerate(dataloader):
            x1,x2 = x1.to(device), x2.to(device)
            y_1,y_2,att_c,att_w = model(x1,x2)
            y_1 = y_1.cpu().numpy()
            y_2 = y_2.cpu().numpy()
            y1_preds.append(y_1.flatten())
            y2_preds.append(y_2.flatten())
            y1_true.append(y1.cpu().numpy().flatten())
            y2_true.append(y2.cpu().numpy().flatten())
            which_cluster.append(wc.cpu().numpy().flatten())
    
    scores = calculate_scores(preds_1= np.concatenate(y1_preds,axis=0),
                              preds_2=np.concatenate(y2_preds,axis=0),
                              true_1=np.concatenate(y1_true,axis=0),
                              true_2=np.concatenate(y2_true,axis=0),
                              which_cluster = np.concatenate(which_cluster,axis=0))
    return scores

@timer
def evaluate_st(model,dataloader,device=torch.device('cuda:1')):
    """
    """
    model.eval()
    y1_preds = []
    y2_preds = []
    y1_true =[]
    y2_true = []
    which_cluster = []
    with torch.no_grad():
        for bid, (x1,y1,wc) in enumerate(dataloader):
            x1,y1 = x1.to(device), y1.to(device)
            y_1,att_c,att_w = model(x1)
            y_1 = y_1.cpu().numpy()
            y1_preds.append(y_1.flatten())
            y1_true.append(y1.cpu().numpy().flatten())
            which_cluster.append(wc.cpu().numpy().flatten())
    
    scores = calculate_scores_single(preds_1= np.concatenate(y1_preds,axis=0),
                              true_1=np.concatenate(y1_true,axis=0),
                              which_cluster = np.concatenate(which_cluster,axis=0))
    return scores

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=2, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True