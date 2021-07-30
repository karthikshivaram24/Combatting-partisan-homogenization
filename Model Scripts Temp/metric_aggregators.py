import pandas as pd
from model_metrics import print_res

def print_settings(settings,single=False,loss3=False):
    """
    (0.001, 8, 16, 0.3, 0.3, 0.05)
    """
    order = None
    if not single:
        order = ["lr","epochs","batch_size","dropout","loss_w","l2-regc"]
        if loss3:
            order = ["lr","epochs","batch_size","dropout","loss_w1","loss_w2","l2-regc"]
    
    if single:
        order = ["lr","epochs","batch_size","dropout","weight_decay"]
        
    for i,k in enumerate(settings):
        print("%s :: %s" %(order[i],str(k)))

def avg_scores(score_tup_list):
    """
    """
    dfs = []
    for st in score_tup_list:
        score_df = st[1]
        dfs.append(score_df)
    
    overall_df = pd.concat(dfs,axis=0)
    return overall_df.groupby(['Score_type']).mean()

def agg_multiscores(scores_tup_list):
    """
    """
    dfs = []
    for stp in scores_tup_list:
        cp = stp[0]
        df = stp[1]
        
        df["Loss Weight"] = df["Settings"].apply(lambda x: x[4])
        dfs.append(df)
    
    overall_df = pd.concat(dfs,axis=0)
    return overall_df.groupby(['Score_type','Loss Weight']).mean()

def agg_multiscores_dual_loss(scores_tup_list):
    """
    """
    dfs = []
    for stp in scores_tup_list:
        cp = stp[0]
        df = stp[1]
        
        df["Loss Weight 1"] = df["Settings"].apply(lambda x: x[4])
        df["Loss Weight 2"] = df["Settings"].apply(lambda x: x[5])
        dfs.append(df)
    
    overall_df = pd.concat(dfs,axis=0)
    print(overall_df.columns)
    return overall_df.groupby(['Score_type',"Loss Weight 1","Loss Weight 2"]).mean()

def select_top_settings(scores,metric_type="F1"):
    """
    """
    score_df = print_res(scores)
    
    # Select the setting with best overall F1 and best overall Acc
    
    overall_df = score_df.loc[score_df.Score_type == "overall"]
    overall_df = overall_df.reset_index(drop=True)
    best_setting = overall_df.iloc[overall_df[metric_type].idxmax()].tolist()[0]
    score_df_best = score_df.loc[score_df.Settings == best_setting]
    return score_df_best,best_setting

def select_best_setting_multi(scores,metric_type="F1"):
    """
    """
    score_df = print_res(scores)
    
    # Select the setting with best overall F1 and best overall Acc
    
    overall_df = score_df.loc[score_df.Score_type == "overall"]
    overall_df = overall_df.reset_index(drop=True)
    best_setting = overall_df.iloc[overall_df[metric_type].idxmax()].tolist()[0]
    
    
    
    best_setting_withlossweights = []
    
    for loss_weight in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        
        temp = list(best_setting)
        temp[4] = loss_weight
        best_setting_withlossweights.append(temp)
    
    score_df_best = score_df.loc[score_df.Settings.isin(best_setting_withlossweights)]
    
    return score_df_best, best_setting