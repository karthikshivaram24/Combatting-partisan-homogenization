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
        for bid, (x1,am1,x2,y1,y2,wc) in enumerate(dataloader):
            x1,am1,x2 = x1.to(device),am1.to(device), x2.to(device)
            y_1,y_2,att_c,att_w = model(x1,am1,x2)
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
        for bid, (x1,am1,y1,wc) in enumerate(dataloader):
            x1,am1,y1 = x1.to(device),am1.to(device), y1.to(device)
            y_1,att_c,att_w = model(x1,am1)
            y_1 = y_1.cpu().numpy()
            y1_preds.append(y_1.flatten())
            y1_true.append(y1.cpu().numpy().flatten())
            which_cluster.append(wc.cpu().numpy().flatten())
    
    scores = calculate_scores_single(preds_1= np.concatenate(y1_preds,axis=0),
                              true_1=np.concatenate(y1_true,axis=0),
                              which_cluster = np.concatenate(which_cluster,axis=0))
    return scores

