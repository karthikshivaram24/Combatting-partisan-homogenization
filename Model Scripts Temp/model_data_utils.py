'''
This script contains utility functions used to create our train and test set from given topic pairs

@Author : Karthik Shivaram

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch_seeder import RANDOM_SEED



def balance_classes(df):
    """
    This function undersamples the majority class label for a dataset (pandas df) to balance out the class labels
    """
    min_partisan = df.binary_ps.value_counts().min()
    
    df_0 = df[df.binary_ps == 0].sample(min_partisan,random_state=42)
    df_1 = df[df.binary_ps == 1].sample(min_partisan,random_state=42)
    
    df = pd.concat((df_0,df_1)).sample(frac=1,random_state=42)
    
    return df

def get_label_dist(labels):
    """
    This function prints the class label distribution given a list of labels
    """
    print("1 : %s" %str(sum(labels)))
    print("0 : %s" %str(len(labels) - sum(labels)))

    
def split_data_mixed(c1s,c2,cluster_2_doc_map,df,c2_train_perc=0.10, verbose=True):
    """
    Train - 90 % from c1 mix + 10% from c2
    Test - 50% from c1 mix and 50% from c2
    """
    c1_sizes = [len(cluster_2_doc_map[c]) for c in c1s]
    c2_size = len(cluster_2_doc_map[c2])
    
    data_2_split_size = sum(sorted(c1_sizes+[c2_size])[:len(c1s+[c2])-1])
    
    train_size = 0.65 * data_2_split_size
    test_size = 0.2 * data_2_split_size
    val_size = 0.15 * data_2_split_size
    
    c1_mix_train_size = (1-c2_train_perc) * train_size
    c2_train_size = c2_train_perc * train_size
    
    if verbose:
    
        print("Selected data sizes : ")
        print("Train Size : %s" %str(train_size))
        print("Test Size : %s" %str(test_size))
        print("Val Size : %s" %str(val_size))

        print("Percentage of train size for c in c1mix : %s"%str(c1_mix_train_size/len(c1s)))
        print("Train Size C1 : %s"%str(c1_mix_train_size))
        print("Train Size C2 : %s"%str(c2_train_size))
    
    
    c1_mix_val_size = (1-c2_train_perc) * val_size
    c2_val_size = c2_train_perc * val_size
    
    c1_mix_test_size = 0.5 * test_size
    c2_test_size = 0.5 * test_size
    
    if verbose:
        
        print("Percentage of Val size for c in c1mix : %s"%str(c1_mix_val_size/len(c1s)))
        print("Percentage of test size for c in c1mix : %s"%str(c1_mix_test_size/len(c1s)))
    
    cmix_train = []
    cmix_test = []
    cmix_val = []
    
    strata_columns = "binary_ps"
    
    # Sample c1mix for train,val,test
    for c in c1s:
        c_df = df.iloc[cluster_2_doc_map[c]]
        c_df["which_cluster"] = [1]*c_df.shape[0]
        c_df["context_pos_words_size"] = c_df["context_pos_words"].apply(lambda x: len(x))
        c_df = c_df[c_df["context_pos_words_size"] > 0]
        c_df = balance_classes(c_df)
        
        c_train, c_test = train_test_split(c_df,test_size=int(c1_mix_test_size/len(c1s)), stratify=c_df[strata_columns], random_state=RANDOM_SEED)
        c_train, c_val = train_test_split(c_train,test_size=int(c1_mix_val_size/len(c1s)), stratify=c_train[strata_columns], random_state=RANDOM_SEED)
        
        if c_train.shape[0] > int(c1_mix_train_size/len(c1s)):
            c_train,_ = train_test_split(c_train,train_size=int(c1_mix_train_size/len(c1s)), stratify=c_train[strata_columns], random_state=RANDOM_SEED)
        
        
        cmix_train.append(c_train)
        cmix_test.append(c_test)
        cmix_val.append(c_val)
        
     # Merge the cluster dfs
    
    cmix_train = pd.concat(cmix_train,axis=0)
    cmix_test = pd.concat(cmix_test,axis=0)
    cmix_val = pd.concat(cmix_val,axis=0)
    
    # Subsample to balance the sizes
    
    if cmix_train.shape[0] > c1_mix_train_size:
        cmix_train,_ = train_test_split(cmix_train,train_size=int(c1_mix_train_size), stratify=cmix_train[strata_columns], random_state=RANDOM_SEED)
    
    if cmix_test.shape[0] > c1_mix_test_size:
        cmix_test,_ = train_test_split(cmix_test,train_size=int(c1_mix_test_size), stratify=cmix_test[strata_columns], random_state=RANDOM_SEED)
    
    if cmix_val.shape[0] > c1_mix_val_size:
        cmix_val,_ = train_test_split(cmix_val,train_size=int(c1_mix_val_size), stratify=cmix_val[strata_columns], random_state=RANDOM_SEED)
        
        
    
    # Sample from c2
    c2_df = df.iloc[cluster_2_doc_map[c2]]
    c2_df["which_cluster"] = [2]*c2_df.shape[0]
    c2_df["binary_ps"] = c2_df["binary_ps"].apply(lambda x: np.abs(x+(-1)))
    c2_df["context_pos_words_size"] = c2_df["context_pos_words"].apply(lambda x: len(x))
    c2_df = c2_df[c2_df["context_pos_words_size"] > 0]
    c2_df = balance_classes(c2_df)
    
    c2_train,c2_test = train_test_split(c2_df,test_size=int(c2_test_size), stratify=c2_df[strata_columns], random_state=RANDOM_SEED)
    c2_train,c2_val = train_test_split(c2_train,test_size=int(c2_val_size), stratify=c2_train[strata_columns], random_state=RANDOM_SEED)
    
    # Subsample to balance the sizes
    
    if c2_train.shape[0] > c2_train_size:
        c2_train,_ = train_test_split(c2_train,train_size=int(c2_train_size), stratify=c2_train[strata_columns], random_state=RANDOM_SEED)
    
    if c2_test.shape[0] > c2_test_size:
        c2_test,_ = train_test_split(c2_test,train_size=int(c2_test_size), stratify=c2_test[strata_columns], random_state=RANDOM_SEED)
    
    if c2_val.shape[0] > c2_val_size:
        c2_val,_ = train_test_split(c2_val,train_size=int(c2_val_size), stratify=c2_val[strata_columns], random_state=RANDOM_SEED)
    
    
    # Combine c1mix with c2
    train = pd.concat([cmix_train,c2_train],axis=0).sample(frac=1,random_state=42)
    test = pd.concat([cmix_test,c2_test],axis=0).sample(frac=1,random_state=42)
    val = pd.concat([cmix_val,c2_val],axis=0).sample(frac=1,random_state=42)
    
    if verbose:
        print("\nSampled Data Sizes: ")
        print("Train Size : %s" %str(train.shape))
        get_label_dist(train["binary_ps"])
        print("Test Size : %s" %str(test.shape))
        get_label_dist(test["binary_ps"])
        print("Val Size : %s" %str(val.shape))
        get_label_dist(val["binary_ps"])
    
    return train,test,val
    
def load_loss3_weights():
    """
    """
    loss_w_df = pd.read_csv("word_loss3_weights.csv",skiprows=1)
    print(loss_w_df.columns)
    w1s = list(set(loss_w_df["Alpha 1"].tolist()))
    w2s = list(set(loss_w_df["Alpha 2"].tolist()))
    return w1s,w2s

def filter_4_pure(data,cluster=2):
    """
    """
    print("Data Shape before Purifying : %s" %str(data.shape))
    data = data.loc[data["which_cluster"] ==cluster]
    print("Data Shape after Purifying : %s" %str(data.shape))
    return data