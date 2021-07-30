
import numpy as np
import torch
from torch_utils import timer
from model_metrics import calculate_scores, calculate_scores_single

@timer
def evaluate_mt(model,dataloader,device=torch.device('cuda:1'),avg_type="binary"):
    """
    """
    model.eval()
    y1_preds = []
    y2_preds = []
    y1_true =[]
    y2_true = []
    which_cluster = []
    with torch.no_grad():
        for bid, (x1v, x2v, y1v, t1v, wcv) in enumerate(dataloader):
            x1v,y1v = x1v.to(device), y1v.to(device)
            y_1,y_2 ,attcvec= model(x1v,None)
            y_1 = y_1.cpu().numpy()
            y1_preds.append(y_1.flatten())
            y1_true.append(y1v.cpu().numpy().flatten())
            which_cluster.append(wcv.cpu().numpy().flatten())
    
    scores = calculate_scores(preds_1= np.concatenate(y1_preds,axis=0),
                              preds_2=np.zeros(np.concatenate(y1_preds,axis=0).shape),
                              true_1=np.concatenate(y1_true,axis=0),
                              true_2=np.zeros(np.concatenate(y1_true,axis=0).shape),
                              which_cluster = np.concatenate(which_cluster,axis=0),avg_type=avg_type)
    return scores

@timer
def evaluate_st(model,dataloader,device=torch.device('cuda:1'),avg_type="binary"):
    """
    """
    model.eval()
    y1_preds = []
    y2_preds = []
    y1_true =[]
    y2_true = []
    which_cluster = []
    with torch.no_grad():
        for bid, (x1,y1,t1,wc) in enumerate(dataloader):
            x1,y1 = x1.to(device), y1.to(device)
            y_1 = model(x1)
            y_1 = y_1.cpu().numpy()
            y1_preds.append(y_1.flatten())
            y1_true.append(y1.cpu().numpy().flatten())
            which_cluster.append(wc.cpu().numpy().flatten())
    
    scores = calculate_scores_single(preds_1= np.concatenate(y1_preds,axis=0),
                              true_1=np.concatenate(y1_true,axis=0),
                              which_cluster = np.concatenate(which_cluster,axis=0),avg_type=avg_type)
    return scores