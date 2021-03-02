import numpy as np
import pandas as pd
from general_utils import timer
from config import RANDOM_SEED
from collections import defaultdict,Counter

@timer
def load_data(path):
    """
    Loads the article csv file and processes it to remove articles with missing content,
    removes duplicates, removes articles with neutral stance (0), converts the partisan scores
    to a binary format and returns a pandas dataframe with the data loaded.
    
    Parameters :
    ----------
    * path -> path of the csv file to load
    
    Returns:
    --------
    Pandas Dataframe
    """
    df = pd.read_csv(path)
    print(df.columns)
    print("Df original shape : %s" %str(df.shape))
    # drop rows with text as nan
    df = df[df['text'].notna()]
    print("Df shape after dropping nan text : %s" %str(df.shape))
    #drop duplicates based on title
    df = df.drop_duplicates(subset=['title'], keep="first")
    print("Df shape after dropping duplicate articles based on title : %s" %str(df.shape))
    # drop articles that have stance = 0
    df = df[df["source_partisan_score"] != 0]
    print("Df shape after dropping 0 stance articles : %s" %str(df.shape))
    # convert articles of stance -1,+1,-2,+2
    df["binary_ps"] = df["source_partisan_score"].apply(lambda x: 1 if x>=1 else 0)
    
    return df

@timer
def sample_data(df,sample_size=100000,seed=RANDOM_SEED):
    """
    Samples a given pandas dataframe
    
    Parameters :
    ----------
    * df -> pandas dataframe to sample from
    * sample_size -> float, proportion to sample from the given dataframe
    * seed -> the random seed to set for random sampling
    
    Returns :
    ---------
    Pandas Dataframe
    """
    df_out = df.sample(n=sample_size, replace=False, random_state=seed)
    df_out.reset_index(drop=True,inplace=True)
    return df_out

def balanced_sampling(labels):
    """
    takes in a array of labels, undersamples the major class and returns the indices to keep
    """    
    labels = labels.astype(int)
    freqs = Counter(labels)
    pos_size = freqs[1]
    neg_size = freqs[0]
    
    min_class_size = min([pos_size,neg_size])
    
    pos_indices = []
    neg_indices = []

    for index in range(labels.shape[0]):
        if labels[index] == 1:
            pos_indices.append(index)
        else:
            neg_indices.append(index)
    
    
    pos_indices = np.random.choice(np.array(pos_indices),min_class_size,replace=False)
    neg_indices = np.random.choice(np.array(neg_indices),min_class_size,replace=False)
    
    res = np.concatenate([pos_indices,neg_indices])
    np.random.shuffle(res)
    
    return res

def create_train_test(cluster_pair,cluster2doc,X_feats,df,user_type="Heterogeneous"):
    """
    Labels are based on conservative when homogenous, or conservative on cluster 1 and liberal on cluster 2 if heterogeneous
    """
    c1 = cluster_pair[0]
    c2 = cluster_pair[1]
    
    cluster_1_doc_indices = cluster2doc[c1]
    cluster_2_doc_indices = cluster2doc[c2]
    
    x_train = X_feats[cluster_1_doc_indices]
    x_test = X_feats[cluster_2_doc_indices]
    
    ps_train = df["binary_ps"].values[cluster_1_doc_indices]
    ps_test = df["binary_ps"].values[cluster_2_doc_indices]
    
    ps_train_indices_sample = balanced_sampling(ps_train)
    ps_test_indices_sample = balanced_sampling(ps_test)
    
    x_train = np.take(x_train, ps_train_indices_sample, axis=0)
    ps_train = np.take(ps_train,ps_train_indices_sample,axis=0)
    
    x_test = np.take(x_test, ps_test_indices_sample, axis=0)
    ps_test = np.take(ps_test,ps_test_indices_sample,axis=0) 
    
    if user_type == "Heterogeneous":
        y_train = ps_train
        y_test = list(map(lambda x: 0.0 if x==1.0 else 1.0,ps_test))
    
    if user_type == "Homogeneous":
        y_train = ps_train
        y_test = ps_test
    
    u_train,c_train = np.unique(y_train, return_counts=True)
    u_test,c_test = np.unique(y_test, return_counts=True)
    
    assert x_train.shape[0] == len(y_train)
    assert x_test.shape[0] == len(y_test)
    
    return x_train,x_test,y_train,y_test,cluster_1_doc_indices,cluster_2_doc_indices