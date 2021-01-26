import os
from collections import Counter,defaultdict
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pickle
import re
import seaborn as sns
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn import metrics
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV,SGDClassifier,PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import joblib
from joblib import Parallel, delayed
from functools import wraps
from time import time
import itertools
from functools import partial
import functools
import time
import warnings
import copy

from .settings import RANDOM_SEED

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
    
    for ind,val in np.ndenumerate(labels):
        if val == 1:
            pos_indices.append(ind[0])
        else:
            neg_indices.append(ind[0])
    
    
    pos_indices = np.random.choice(np.array(pos_indices),min_class_size,replace=False)
    neg_indices = np.random.choice(np.array(neg_indices),min_class_size,replace=False)
    
    return np.concatenate([pos_indices,neg_indices])


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
    #print("Train Label Dist :\n %s : %s\n %s:%s" %(str(u_train[0]),str(c_train[0]),str(u_train[1]),str(c_train[1])))
    #print("Test Label Dist :\n %s : %s\n %s:%s" %(str(u_test[0]),str(c_test[0]),str(u_test[1]),str(c_test[1])))
    
    assert x_train.shape[0] == len(y_train)
    assert x_test.shape[0] == len(y_test)
    
    return x_train,x_test,y_train,y_test

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

