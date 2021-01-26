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

from .general_utils import timer
from .settings import RANDOM_SEED, NUM_CLUSTERS, CLUSTER_MIN_SIZE, CLUSTER_MAX_SIZE, MIN_PARTISAN_SIZE


@timer
def run_clustering(vectors,seed=RANDOM_SEED,num_clusters=NUM_CLUSTERS,clus_type="kmeans"):
    """
    """
    if clus_type == "kmeans":
        print("\nRunning KMEANS Clustering with k=%s" %str(num_clusters))
        km = MiniBatchKMeans(n_clusters=num_clusters, random_state=seed, n_init=3, max_iter=200, batch_size=1000)
        clusters = km.fit_predict(vectors)
        return clusters,km
    
    if clus_type == "spectral":
        return None
    
    if clus_type == "dbscan":
        return None

@timer
def get_cluster_sizes(cluster_clf):
    """
    """
    cluster_sizes = Counter(cluster_clf.labels_)
    return cluster_sizes

@timer
def score_cluster(vectors,cluster_clf,score_type="sil_score"):
    """
    """
    if score_type == "sil_score":
        sil_score = metrics.silhouette_score(vectors, cluster_clf.labels_, metric='euclidean')
        print("\nSilhouetter Score : %s" %str(sil_score))
        return sil_score
    
    return None

@timer
def get_cluster_pairs(num_clusters):
    """
    """
    cluster_pairs = list(itertools.combinations(range(num_clusters), 2))
    print("\nNumber of Cluster Pairs : %s" %str(len(cluster_pairs)))
    return cluster_pairs

@timer
def get_pairwise_dist(cluster_clf,dist_type="cosine"):
    """
    """
    pairwise_dist = None
    if dist_type == "cosine":
        pairwise_dist = cosine_similarity(cluster_clf.cluster_centers_)
    return pairwise_dist

@timer
def cluster2doc(num_texts,cluster_labels):
    """
    """
    cluster_2_doc = defaultdict(list)
    for index in range(num_texts):
        cluster = cluster_labels[index]
        cluster_2_doc[cluster].append(index)
    return cluster_2_doc


@timer
def filter_clusters(cluster_pairs,
                    doc_2_cluster_map,
                    cluster_sizes,
                    partisan_scores,
                    min_size=CLUSTER_MIN_SIZE,
                    max_size=CLUSTER_MAX_SIZE,
                    min_partisan_size=MIN_PARTISAN_SIZE):
    """
    min_partisan_size : percentage of docs in the cluster that must have partisan score of 0 (and similarly 1), this removes pure clusters as well
    """

    def get_cluster_partisan_map(doc_2_cluster_map,partisan_scores):
        """
        """
        cluster_partisan_map = defaultdict(int)
        for cluster in doc_2_cluster_map:
            ps_scores = []
            for doc_id in doc_2_cluster_map[cluster]:
                ps_scores.append(partisan_scores[doc_id])
            cluster_partisan_map[cluster]=ps_scores
        
        return cluster_partisan_map
    
    cluster_partisan_map = get_cluster_partisan_map(doc_2_cluster_map,partisan_scores)
    
    def filter_min_max(cluster_pair,cluster_sizes):
        """
        Boolean Func
        """
        print(min_size)
        print(max_size)
        print(cluster_pair)
        verdict = True
        cond1 = cluster_sizes[cluster_pair[0]] >= min_size and cluster_sizes[cluster_pair[0]] <= max_size 
        cond2 = cluster_sizes[cluster_pair[1]] >= min_size and cluster_sizes[cluster_pair[1]] <= max_size 
        if cond1 == True and cond2 == True:
            verdict = False
        
        return verdict
    
    partial_filter_min_max = partial(filter_min_max,cluster_sizes=cluster_sizes)
    
    def filter_partisan_size(cluster_pair,min_partisan_size,cluster_sizes,cluster_partisan_map):
        """
        Boolean Func
        takes a cluster pair and checks the partisan distribution compaired to its cluster size
        
        """
        conds = [True,True]
        for i,c in enumerate(cluster_pair):
            cluster_partisan = cluster_partisan_map[c]
            partisan_size = Counter(cluster_partisan)
            if partisan_size[0] >= int(cluster_sizes[c]*min_partisan_size) and partisan_size[1] >= int(cluster_sizes[c]*min_partisan_size):
                conds[i] = False
        
        if conds[0] == conds[1] == False:
            return False
        else:
            return True

    
    partial_filter_partisan_size = partial(filter_partisan_size,
                                           min_partisan_size=min_partisan_size,
                                           cluster_sizes=cluster_sizes,
                                           cluster_partisan_map=cluster_partisan_map)
    
    filter_verdicts_min_max = Parallel(n_jobs=-1)(delayed(partial_filter_min_max)(c_p) for c_p in cluster_pairs)
    filter_verdicts_partisan_size = Parallel(n_jobs=-1)(delayed(partial_filter_partisan_size)(c_p) for c_p in cluster_pairs)
    
    filtered_cps = []
    for index,cp in enumerate(cluster_pairs):
        if not filter_verdicts_min_max[index] and not filter_verdicts_partisan_size[index]:
            filtered_cps.append(cp)
    
    return filtered_cps

@timer
def get_top_100_clusterpairs(cluster_pairs,dist_matrix,reverse=True):
    """
    """
    sorted_cps = sorted(cluster_pairs,key=lambda x: dist_matrix[x[0],x[1]],reverse=reverse)[:100]
    return sorted_cps