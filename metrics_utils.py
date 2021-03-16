import numpy as np
from sklearn import metrics
from config import RANDOM_SEED
from collections import defaultdict,Counter

def entropy(arr, max_one=True):
    """
    Normalized entropy , we divide by the log_2(num of states/ classes)
    """
    e = 0
    for p in arr:
        if p > 0:
            e += p*np.log2(p)

    if max_one:
        e /= np.log2(len(arr))

    return -1*e   

def calc_entropy_and_dist(preds):
    """
    Calculates entropy btwe topic, ps and new's source distribution
    """
    
    prob_counter = Counter(preds)
    prob_vec = [prob_counter[stance]/len(preds) for stance in sorted(prob_counter.keys())]

    assert round(sum(prob_vec)) == 1.0

    return entropy(prob_vec),prob_vec

def calc_entropy_and_dist_sep(preds,which_cluster):
    """
    """
    # Probs - p_c1, p_c2
    preds_c1 =[]
    preds_c2 = []
    
    for cind,c in enumerate(which_cluster):
        if c == 1:
            preds_c1.append(preds[cind])
        else:
            preds_c2.append(preds[cind])
    
    prob_count = Counter(preds)
    prob_C1 = Counter(preds_c1)
    prob_C2 = Counter(preds_c2)
    
    prob_vec = [prob_count[stance]/len(preds) for stance in sorted(prob_count.keys())]
    prob_vec_c1 = [prob_C1[stance]/len(preds_c1) for stance in sorted(prob_C1.keys())]
    prob_vec_c2 = [prob_C2[stance]/len(preds_c2) for stance in sorted(prob_C2.keys())]
    # entropy - e_c1, e_c2
    
    ent = entropy(prob_vec)
    ent_c1 = entropy(prob_vec_c1)
    ent_c2 = entropy(prob_vec_c2)
    
    return ent, ent_c1, ent_c2, prob_vec, prob_vec_c1, prob_vec_c2
    pass

def calculate_masked_avg(act_arr,mask_array,cluster_=1):
    """
    """
    mask_const = None
    if cluster_ == 1:
        mask_const = 2
    else:
        mask_const = 1
        
    masked_actual = np.ma.masked_where(mask_array==mask_const,act_arr)
    return np.ma.mean(masked_actual)

def calculate_avg_precision_param_variation(scores_,params,mode="single"):
    """
    """
    param_results =defaultdict(dict)
    for param in params:
        avg_prescision = []
        c1_avg_precision = []
        c2_avg_precision = []
        avg_entropy = []
        c1_avg_entropy = []
        c2_avg_entropy = []
        
        avg_dist = []
        c1_avg_dist = []
        c2_avg_dist = []
        for cp in scores_:
            avg_prescision.append(np.mean(scores_[cp][str(param)]["precision"]))
            avg_entropy.append(scores_[cp][str(param)]["entropy"])
            avg_dist.append(scores_[cp][str(param)]["stance_dist"])
            if mode == "mixed":
#                 c1_avg_precision.append(np.mean(scores_[cp][str(param)]["precision_c1"]))
                c1_avg_precision.append(calculate_masked_avg(act_arr = scores_[cp][str(param)]["precision_c1"], 
                                                             mask_array=scores_[cp][str(param)]["which_cluster"],cluster_=1))
                c1_avg_entropy.append(scores_[cp][str(param)]["entropy_c1"])
#                 c1_avg_dist.append(scores_[cp][str(param)]["stance_dist_c1"])
#                 c2_avg_precision.append(np.mean(scores_[cp][str(param)]["precision_c2"]))
                c2_avg_precision.append(calculate_masked_avg(act_arr = scores_[cp][str(param)]["precision_c2"], 
                                                             mask_array=scores_[cp][str(param)]["which_cluster"],cluster_=2))
                c2_avg_entropy.append(scores_[cp][str(param)]["entropy_c2"])
#                 c2_avg_dist.append(scores_[cp][str(param)]["stance_dist_c2"])
    
        
        param_results[param]["avg_precision"] = avg_prescision
        param_results[param]["avg_entropy"] = avg_entropy
        param_results[param]["stance_dist"] = avg_dist
        
        if mode == "mixed":
            param_results[param]["c1_avg_precision"] = c1_avg_precision
            param_results[param]["c2_avg_precision"] = c2_avg_precision
            param_results[param]["c1_avg_entropy"] = c1_avg_entropy
            param_results[param]["c2_avg_entropy"] = c2_avg_entropy
#             param_results[param]["c1_avg_dist"] = c1_avg_dist
#             param_results[param]["c2_avg_dist"] = c2_avg_dist
    
    return param_results

def calculate_map_param_variation(param_results,mode="single"):
    """
    """
    results_dict = defaultdict(lambda : defaultdict(float))
    for param in param_results:
#         print("\nParam : %s" %str(param))
#         print(np.mean(param_results[param]["avg_precision"]))
        results_dict[param]["avg_precision"] = np.mean(param_results[param]["avg_precision"])
        results_dict[param]["avg_entropy"] = np.mean(param_results[param]["avg_entropy"])
        
#         avg_stance_0 = np.mean([i[0] for i in param_results[param]["stance_dist"]])
#         avg_stance_1 = np.mean([i[1] for i in param_results[param]["stance_dist"]])
        
        results_dict[param]["avg_stance"] = np.mean(param_results[param]["stance_dist"],axis=0)
        
        if mode == "mixed":
            results_dict[param]["c1_avg_precision"] = np.mean(param_results[param]["c1_avg_precision"])
            results_dict[param]["c2_avg_precision"] = np.mean(param_results[param]["c2_avg_precision"])
            results_dict[param]["c1_avg_entropy"] = np.mean(param_results[param]["c1_avg_entropy"])
            results_dict[param]["c2_avg_entropy"] = np.mean(param_results[param]["c2_avg_entropy"])
            
            
#             avg_stance_0 = np.mean(param_results[param]["c1_avg_dist"],axis=0)
#             avg_stance_1 = np.mean(param_results[param]["c1_avg_dist"],axis=0)
#             results_dict[param]["c1_avg_dist"] = np.mean(param_results[param]["c1_avg_dist"],axis=0)
            
#             avg_stance_0 = np.mean([i[0] for i in param_results[param]["c2_avg_dist"]])
#             avg_stance_1 = np.mean([i[1] for i in param_results[param]["c2_avg_dist"]])
#             results_dict[param]["c2_avg_dist"] = np.mean(param_results[param]["c2_avg_dist"],axis=0)
            
            
        else:
            results_dict[param]["c1_avg_precision"] = np.nan
            results_dict[param]["c2_avg_precision"] = np.nan
            results_dict[param]["c1_avg_entropy"] = np.nan
            results_dict[param]["c2_avg_entropy"] = np.nan
#             results_dict[param]["c1_avg_dist"] = [np.nan, np.nan]
#             results_dict[param]["c2_avg_dist"] = [np.nan, np.nan]
            
            
    return results_dict
    
def calculate_avg_precision(scores_,mode="single"):
    """
    Calculates Average Precision for a given Cluster Pair, here each 
    cluster pair represents a user against a system trained to learn his
    preferences
    """    
    avg_prescision = []
    c1_avg_precision = []
    c2_avg_precision = []
    avg_entropy = []
    c1_avg_entropy = []
    c2_avg_entropy = []   
    avg_dist = []
    c1_avg_dist = []
    c2_avg_dist = []
    
    for cp in scores_:
        avg_prescision.append(np.mean(scores_[cp]["logistic_regression"]["precision"]))
        avg_entropy.append(np.mean(scores_[cp]["logistic_regression"]["entropy"]))
        avg_dist.append(scores_[cp]["logistic_regression"]["stance_dist"])
        
        if mode == "mixed":
#             c1_avg_precision.append(np.mean(scores_[cp]["logistic_regression"]["precision_c1"]))
            c1_avg_precision.append(calculate_masked_avg(act_arr = scores_[cp]["logistic_regression"]["precision_c1"], 
                                                             mask_array=scores_[cp]["logistic_regression"]["which_cluster"],cluster_=1))
            c1_avg_entropy.append(scores_[cp]["logistic_regression"]["entropy_c1"])
#             c1_avg_dist.append(scores_[cp]["logistic_regression"]["stance_dist_c1"])
#             c2_avg_precision.append(np.mean(scores_[cp]["logistic_regression"]["precision_c2"]))
            c2_avg_precision.append(calculate_masked_avg(act_arr = scores_[cp]["logistic_regression"]["precision_c2"], 
                                                             mask_array=scores_[cp]["logistic_regression"]["which_cluster"],cluster_=2))
            c2_avg_entropy.append(scores_[cp]["logistic_regression"]["entropy_c2"])
#             c2_avg_dist.append(scores_[cp]["logistic_regression"]["stance_dist_c2"])
            
    if mode == "single":
        return avg_prescision,avg_entropy,avg_dist
    else:
        return avg_prescision, c1_avg_precision, c2_avg_precision, avg_entropy, c1_avg_entropy, c2_avg_entropy, avg_dist, c1_avg_dist, c2_avg_dist
        
    
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