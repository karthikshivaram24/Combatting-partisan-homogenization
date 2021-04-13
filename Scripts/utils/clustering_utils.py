from Scripts.utils.general_utils import timer
from Scripts.utils.config import RANDOM_SEED
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from joblib import Parallel, delayed
from functools import partial
import itertools

@timer
def run_clustering(vectors,seed=RANDOM_SEED,num_clusters=1000,clus_type="kmeans"):
    """
    Performs clustering on a given set of vectors
    
    Parameters:
    -----------
    * vectors -> numpy matrix to cluster
    * seed -> Random seed to use to initialize the clustering models
    * num_clusters -> number of clusters (used for Kmeans)
    * clus_type -> str, the type of clustering to use
    
    Returns:
    -------
    * clusters -> list of ints, the cluster labels for the vectors
    * km -> clustering object
    
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
    calculates the cluster sizes given a list of cluster labels
    
    Parameters:
    -----------
    * cluster_clf -> clustering object
    
    Returns:
    --------
    * cluster_sizes -> A counter object containing the cluster sizes
    """
    cluster_sizes = Counter(cluster_clf.labels_)
    return cluster_sizes

@timer
def score_cluster(vectors,cluster_clf,score_type="sil_score"):
    """
    Scores a clustering output based on the type of metric used
    
    Parameters:
    -----------
    * vectors -> the vectors used in clustering
    * cluster_clf -> the clustering object
    * score_type -> str, a string depicting the type of metric to use to score the clustering
    
    Returns:
    --------
    * Score -> float
    """
    if score_type == "sil_score":
        sil_score = metrics.silhouette_score(vectors, cluster_clf.labels_, metric='euclidean')
        print("\nSilhouetter Score : %s" %str(sil_score))
        return sil_score
    
    return None

@timer
def get_cluster_pairs(num_clusters):
    """
    Calculates all possible cluster pair combinations given a set of clusters
    
    Parameters:
    -----------
    * num_clusters -> int, the number of cluster centers found using the clustering model
    
    Returns:
    --------
    * cluster_pairs -> list of tuples
    """
    cluster_pairs = list(itertools.combinations(range(num_clusters), 2))
    print("\nNumber of Cluster Pairs : %s" %str(len(cluster_pairs)))
    return cluster_pairs

@timer
def get_pairwise_dist(cluster_clf,dist_type="cosine"):
    """
    calcualtes pairwise distance btw cluster centers of cluster pairs
    
    Parameters:
    ----------
    * cluster_clf -> clustering object
    * dist_type -> str, type of distance to calculate pairwise distance
    
    Returns:
    --------
    * pairwise_dist -> matrix of floats
    """
    pairwise_dist = None
    if dist_type == "cosine":
        pairwise_dist = cosine_similarity(cluster_clf.cluster_centers_)
    return pairwise_dist

@timer
def cluster2doc(num_texts,cluster_labels):
    """
    creates a dictionary of cluster label to documents that belong to that respective cluster label
    
    Parameters:
    ----------
    * num_texts ->
    * cluster_labels ->
    
    Returns:
    --------
    * cluster_2_doc -> a dictionary, where keys are cluster labels, values are list of documents that belong to that given cluster label
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
                    min_size=300,
                    max_size=5000,
                    min_partisan_size=0.3):
    """
    Filtering function that removes clusters based on size limits and the label distribution present in the cluster
    
    Parameters:
    -----------
    * cluster pairs -> list of cluster pair tuples
    * doc_2_cluster_map -> dictionary containing cluster label to documents mapping
    * cluster_sizes -> dictionary with cluster sizes
    * partisan_scores -> a list of partisan scores for each document
    * min_size -> minimum size a cluster needs to have
    * max_size -> maximum size a cluster needs to have
    
    Returns:
    --------
    * filtered_cps -> a list of filtered cluster pairs
    """

    def get_cluster_partisan_map(doc_2_cluster_map,partisan_scores):
        """
        Inner Function that creates a dictionary of cluster to partisan scores
        """
        cluster_partisan_map = {}
        for cluster in doc_2_cluster_map:
            ps_scores = []
            for doc_id in doc_2_cluster_map[cluster]:
                ps_scores.append(partisan_scores[doc_id])
            cluster_partisan_map[cluster]=ps_scores
        
        return cluster_partisan_map
    
    cluster_partisan_map = get_cluster_partisan_map(doc_2_cluster_map,partisan_scores)
    
    def filter_min_max(cluster_pair,cluster_sizes):
        """
        Function that filters a set of cluster pairs based on the size limits
        """
        verdict = True
        cond1 = cluster_sizes[cluster_pair[0]] >= min_size and cluster_sizes[cluster_pair[0]] <= max_size 
        cond2 = cluster_sizes[cluster_pair[1]] >= min_size and cluster_sizes[cluster_pair[1]] <= max_size 
        if cond1 == True and cond2 == True:
            verdict = False
        
        return verdict
    
    partial_filter_min_max = partial(filter_min_max,cluster_sizes=cluster_sizes)
    
    def filter_partisan_size(cluster_pair,min_partisan_size,cluster_sizes,cluster_partisan_map):
        """
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

    
    partial_filter_partisan_size = partial(filter_partisan_size,min_partisan_size=min_partisan_size,
                                                                cluster_sizes=cluster_sizes,
                                                                cluster_partisan_map=cluster_partisan_map)
    
    # filter for size
    filter_verdicts_min_max = Parallel(n_jobs=-1)(delayed(partial_filter_min_max)(c_p) for c_p in cluster_pairs)
    cp_filtered_min__max = [c_p for i,c_p in enumerate(cluster_pairs) if filter_verdicts_min_max[i] == False ]
    
    # filter for partisan dist
    filter_verdicts_partisan_size = Parallel(n_jobs=-1)(delayed(partial_filter_partisan_size)(c_p) for c_p in cp_filtered_min__max)
    filtered_cps = [c_p for i,c_p in enumerate(cp_filtered_min__max) if filter_verdicts_partisan_size[i]==False ]
    
    return filtered_cps



@timer
def get_top_100_clusterpairs(cluster_pairs,dist_matrix,reverse=True):
    """
    Sorts given set of cluster pairs 
    """
    sorted_cps = sorted(cluster_pairs,key=lambda x: dist_matrix[x[0],x[1]],reverse=reverse)[:100]
    return sorted_cps