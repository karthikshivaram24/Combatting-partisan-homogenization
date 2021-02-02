import numpy as np
from sklearn import metrics
from config import RANDOM_SEED
from collections import defaultdict

def calculate_avg_precision_param_variation(scores_,params,mode="single"):
    """
    """
    param_results =defaultdict(dict)
    for param in params:
        avg_prescision = []
        c1_avg_precision = []
        c2_avg_precision = []
        for cp in scores_:
            avg_prescision.append(np.mean(scores_[cp][str(param)]["precision"]))
            if mode == "mixed":
                c1_avg_precision.append(np.mean(scores_[cp][str(param)]["precision_c1"]))
                c2_avg_precision.append(np.mean(scores_[cp][str(param)]["precision_c2"]))
        
        param_results[param]["avg_precision"] = avg_prescision
        
        if mode == "mixed":
            param_results[param]["c1_avg_precision"] = c1_avg_precision
            param_results[param]["c2_avg_precision"] = c2_avg_precision
    
    return param_results

def calculate_map_param_variation(param_results,mode="single"):
    """
    """
    for param in param_results:
        print("\nParam : %s" %str(param))
        print(np.mean(param_results[param]["avg_precision"]))
        if mode == "mixed":
            print("c1 MAP : \n%s"%str(np.mean(param_results[param]["c1_avg_precision"])))
            print("c2 MAP : \n%s"%str(np.mean(param_results[param]["c2_avg_precision"])))
            

def calculate_avg_precision(scores_,mode="single"):
    """
    Calculates Average Precision for a given Cluster Pair, here each 
    cluster pair represents a user against a system trained to learn his
    preferences
    """
    avg_prescision = []
    c1_avg_precision = []
    c2_avg_precision = []
    for cp in scores_:
        avg_prescision.append(np.mean(scores_[cp]["logistic_regression"]["precision"]))
        if mode == "mixed":
            c1_avg_precision.append(np.mean(scores_[cp]["logistic_regression"]["precision_c1"]))
            c2_avg_precision.append(np.mean(scores_[cp]["logistic_regression"]["precision_c2"]))
            
    if mode == "single":
        return avg_prescision
    else:
        return avg_prescision, c1_avg_precision, c2_avg_precision
        
    
def calculate_map(scores_,mode="single"):
    """
    Calculates the mean average precision over all cluster pairs
    """
    all_avgp = []
    c1_avgp = []
    c2_avgp = []
    for cp in scores_:
        all_avgp.append(scores_[cp]["avg_precision"])
        if mode == "mixed":
            c1_avgp.append(scores_[cp]["c1_avg_precision"])
            c2_avgp.append(scores_[cp]["c2_avg_precision"])
    
    print("Mean Avg Precision : %s" %str(round(np.mean(all_avgp),3)))
    if mode == "mixed":
        print("C1 Mean Avg Precision : %s" %str(round(np.mean(c1_avgp),3)))
        print("C2 Mean Avg Precision : %s" %str(round(np.mean(c2_avgp),3)))
    

def get_scores(y_test,predictions,threshold,using_thresh=True):
    """
    """
    if using_thresh:
        predicted_probas = predictions[:,1]
        predictions = np.where(predicted_probas>=threshold,1,0).flatten()
    f1 = metrics.f1_score(y_test,predictions,zero_division=0,average="macro")
    precision = metrics.precision_score(y_test,predictions,zero_division=0,average="macro")
    recall = metrics.recall_score(y_test,predictions,zero_division=0,average="macro")
    accuracy = metrics.accuracy_score(y_test,predictions)
    
    return f1,precision,recall,accuracy

def get_scores_wot(y_test,predictions):
    """
    """
    f1 = metrics.f1_score(y_test,predictions,zero_division=0,average="macro")
    precision = metrics.precision_score(y_test,predictions,average="macro")
    recall = metrics.recall_score(y_test,predictions,average="macro")
    accuracy = metrics.accuracy_score(y_test,predictions)
    
    return f1,precision,recall,accuracy