import numpy as np
from collections import defaultdict
from sklearn import metrics
import pandas as pd

def print_res(scores_):
    """
    """
    settings = []
    score_types = []
    output_type = []
    f1_scores = []
    precis_scores = []
    recall_scores = []
    auc_roc_scores = []
    accuracy_scores = []
    for setting in scores_.keys():
        for score_type in scores_[setting].keys():
            for score_output in scores_[setting][score_type].keys():
                settings.append(setting)
                score_types.append(score_type)
                output_type.append(score_output)
                f1_scores.append(scores_[setting][score_type][score_output]["f1"])
                recall_scores.append(scores_[setting][score_type][score_output]["recall"])
                precis_scores.append(scores_[setting][score_type][score_output]["precision"])
                auc_roc_scores.append(scores_[setting][score_type][score_output]["roc_auc"])
                accuracy_scores.append(scores_[setting][score_type][score_output]["accuracy"])
        
    df = pd.DataFrame()
    df["Settings"] = settings
    df["Score_type"] = score_types
    df["Output_type"] = output_type
    df["F1"] = f1_scores
    df["Precision"] = precis_scores
    df["Recall"] = recall_scores
    df["ROC_AUC"] = auc_roc_scores
    df["Accuracy"] = accuracy_scores
    
    return df



def mask_arrays(yp_1,yp_2,y1,y2,which_cluster,cluster_2_mask=1):
    """
    """
    yp_1_ma = []
    yp_2_ma = []
    y1_ma = []
    y2_ma = []
    
    for wc_i , cluster in enumerate(which_cluster):
        if cluster != cluster_2_mask:
            yp_1_ma.append(yp_1[wc_i])
            yp_2_ma.append(yp_2[wc_i])
            y1_ma.append(y1[wc_i])
            y2_ma.append(y2[wc_i])
    
    return yp_1_ma, yp_2_ma, y1_ma, y2_ma

def mask_arrays_single(yp_1,y1,which_cluster,cluster_2_mask=1):
    """
    """
    yp_1_ma = []
    y1_ma = []
    
    for wc_i , cluster in enumerate(which_cluster):
        if cluster != cluster_2_mask:
            yp_1_ma.append(yp_1[wc_i])
            y1_ma.append(y1[wc_i])
    
    return yp_1_ma, y1_ma
            

def calc_score_(yp_1,yp_2,y1,y2,scores_,key="overall"):
    """
    """
    yp_1 = np.array(yp_1)
    print("Predicted Label Shape : %s"%str(yp_1.shape))
    yp_2 = np.array(yp_2)
    y1 = np.array(y1)
    print("True Label Shape : %s"%str(y1.shape))
    y2 = np.array(y2)
     
    
    yp_1[yp_1 > 0.5] = 1.0
    yp_1[yp_1 <= 0.5] = 0.0
    yp_2[yp_2 > 0.5] = 1.0
    yp_2[yp_2 <=0.5] = 0.0
    
    if 0.0 not in yp_1 or 1.0 not in yp_1:
        print("One class predicitions for class labels")
    
    if 0.0 not in yp_2 or 1.0 not in yp_2: 
        print("One class predictions for word labels")
    
    f1 = metrics.f1_score(y1,yp_1)
    prec = metrics.precision_score(y1,yp_1)
    recall = metrics.recall_score(y1,yp_1)
    try:
        roc_auc = metrics.roc_auc_score(y1,yp_1)
    except:
        print("ROC Error")
        roc_auc = 0.0
    accuracy = metrics.accuracy_score(y1,yp_1)
    
    scores_[key]["class_scores"]["f1"] = f1
    scores_[key]["class_scores"]["precision"] = prec
    scores_[key]["class_scores"]["recall"] = recall
    scores_[key]["class_scores"]["accuracy"] = roc_auc
    scores_[key]["class_scores"]["roc_auc"] = accuracy
    
    f1 = metrics.f1_score(y2,yp_2)
    prec = metrics.precision_score(y2,yp_2)
    recall = metrics.recall_score(y2,yp_2)
    try:
        roc_auc = metrics.roc_auc_score(y2,yp_2)
    except:
        print("ROC Error")
        roc_auc = 0.0
    accuracy = metrics.accuracy_score(y2,yp_2)
    
    scores_[key]["word_scores"]["f1"] = f1
    scores_[key]["word_scores"]["precision"] = prec
    scores_[key]["word_scores"]["recall"] = recall
    scores_[key]["word_scores"]["accuracy"] = roc_auc
    scores_[key]["word_scores"]["roc_auc"] = accuracy
    
    return scores_

def calc_score_single(yp_1,y1,scores_,key="overall"):
    """
    """
    yp_1 = np.array(yp_1)
    print("Predicted Label Shape : %s"%str(yp_1.shape))
    y1 = np.array(y1)
    print("True Label Shape : %s"%str(y1.shape))
     
    
    yp_1[yp_1 > 0.5] = 1.0
    yp_1[yp_1 <= 0.5] = 0.0
    
    if 0.0 not in yp_1 or 1.0 not in yp_1:
        print("One class predicitions for class labels")
    
    
    f1 = metrics.f1_score(y1,yp_1)
    prec = metrics.precision_score(y1,yp_1)
    recall = metrics.recall_score(y1,yp_1)
    try:
        roc_auc = metrics.roc_auc_score(y1,yp_1)
    except:
        print("ROC Error")
        roc_auc = 0.0
    accuracy = metrics.accuracy_score(y1,yp_1)
    
    scores_[key]["class_scores"]["f1"] = f1
    scores_[key]["class_scores"]["precision"] = prec
    scores_[key]["class_scores"]["recall"] = recall
    scores_[key]["class_scores"]["accuracy"] = roc_auc
    scores_[key]["class_scores"]["roc_auc"] = accuracy
    
    return scores_
    

def calculate_scores(preds_1,preds_2,true_1,true_2,which_cluster):
    """
    """
    print("Y1 Pred Dist : ")
    print("1 : %s" %str(np.sum(np.array(preds_1>0.5,dtype=int))))
    print("0 : %s" %str(preds_1.shape[0] - (np.sum(preds_1>0.5))))
    
    print("Y2 Pred Dist : ")
    print("1 : %s" %str(np.sum(np.array(preds_2>0.5,dtype=int))))
    print("0 : %s" %str(preds_2.shape[0] - (np.sum(preds_2>0.5))))
    
    scores_ = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
    # overall_metrics
    scores_ = calc_score_(preds_1,preds_2,true_1,true_2,scores_,key="overall")
    
    # cluster 1 metrics
    yp_1_ma, yp_2_ma, y1_ma, y2_ma =  mask_arrays(preds_1,preds_2,true_1,true_2,which_cluster,cluster_2_mask=2)
    scores_ = calc_score_(yp_1_ma,yp_2_ma,y1_ma,y2_ma,scores_,key="cluster1")
    
    # cluster 2 metrics
    yp_1_ma, yp_2_ma, y1_ma, y2_ma =  mask_arrays(preds_1,preds_2,true_1,true_2,which_cluster,cluster_2_mask=1)
    scores_ = calc_score_(yp_1_ma,yp_2_ma,y1_ma,y2_ma,scores_,key="cluster2")
    
    return scores_

def calculate_scores_single(preds_1,true_1,which_cluster):
    """
    """
    print("Y1 Pred Dist : ")
    print("1 : %s" %str(np.sum(np.array(preds_1>0.5,dtype=int))))
    print("0 : %s" %str(preds_1.shape[0] - (np.sum(preds_1>0.5))))
    
    scores_ = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
    # overall_metrics
    scores_ = calc_score_single(preds_1,true_1,scores_,key="overall")
    
    # cluster 1 metrics
    yp_1_ma, y1_ma =  mask_arrays_single(preds_1,true_1,which_cluster,cluster_2_mask=2)
    scores_ = calc_score_single(yp_1_ma,y1_ma,scores_,key="cluster1")
    
    # cluster 2 metrics
    yp_1_ma, y1_ma =  mask_arrays_single(preds_1,true_1,which_cluster,cluster_2_mask=1)
    scores_ = calc_score_single(yp_1_ma,y1_ma,scores_,key="cluster2")
    
    return scores_

def get_accuracy_from_logits(probs,labels):
    """
    """
#     preds = (probs > 0.5).long()
#     print(preds)
#     acc = (preds.squeeze() == labels.long()).float().mean()
    
    probs = probs.detach().cpu().numpy()
    probs[probs > 0.5] = 1
    probs[probs <= 0.5] = 0
    print(probs)
    acc = metrics.accuracy_score(labels.cpu().numpy(),probs)
    return acc