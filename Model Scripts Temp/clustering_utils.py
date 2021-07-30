import tqdm
import pandas as pd
import numpy as np
import itertools
import string
import re
from joblib import Parallel, delayed
from nltk.tokenize import sent_tokenize
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from context_word_sampler import vectorize_text, sample_context_words,get_tokens
from torch_utils import save_obj,timer
from torch_seeder import RANDOM_SEED
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

def plot_size_dist(cluster_sizes):
    """
    Plots the size distribution of all the clusters
    """
    plt.figure(figsize=(20,10))
    plt.bar(cluster_sizes.keys(), cluster_sizes.values(),width=2)
    plt.xlabel("Cluster-Number")
    plt.ylabel("Documents in Cluster")
    
    plt.title("Cluster Size Distribution")
    plt.savefig("Graphs/cluster_size_dist.pdf")
    plt.show()

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
def preprocess_texts(text_lists):
    """
    Cleans a list of texts by:
    * Selecting the first 10 sentences
    * converting it to lowercase
    * removing punctuations
    * removing small words of size 1 and 2
    * replacing multiple spaces with a single space
    
    Parameters :
    ----------
    * text_lists -> A list of strings
    
    Returns:
    --------
    A cleaned list of text
    """
    
    def select_first10(x):
        return " ".join(sent_tokenize(x)[:10])
    
    def to_lower(x):
        return x.lower()
    
    def remove_punc(x):
        return x.translate(str.maketrans('', '', string.punctuation))
    
    def remove_non_alpha_numeric(x):
        return re.sub("[^0-9a-zA-Z]+", " ", x)
    
    def remove_small_words(x):
        return re.sub(r'\b\w{1}\b', '', x)
    
    def remove_spaces(x):
        return re.sub(' +', ' ', x)
    
    preprocess_pipe = [select_first10,to_lower,remove_punc,remove_small_words,remove_spaces,remove_non_alpha_numeric]
    
    processed_texts = text_lists
    for preprocess_func in preprocess_pipe:
        print("Running : %s" %str(preprocess_func.__name__))
        processed_texts = Parallel(n_jobs=-1)(delayed(preprocess_func)(x) for x in processed_texts)

    return processed_texts

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
        km = MiniBatchKMeans(n_clusters=num_clusters, random_state=seed, n_init=3, max_iter=200, batch_size=100)
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
def tfidf_vectorization(df,min_df=30,max_df=0.75,max_features=8000,seed=RANDOM_SEED,combine_text = True):
    """
    Vectorizes a given list of text (in the form of a dataframe column) using TF-IDF measures
    
    Parameters:
    ----------
    * df -> dataframe that contains the text to vectorize
    * min_df -> integer or float representing the min count or percentage a word needs to appear in the corpus 
    * max_df -> integer of float representing the max count of percentage a word needs to appear in the corpus
    * seed -> Not used
    
    Returns:
    --------
    * vectors -> csr matrix of the converted vectors
    * vocab -> vocabulary of the tfidf vectors
    * tf_idf_vectorizer -> the vectorizer object 
    """
    if combine_text:
        df["all_text"] = df["title"] + " " + df["processed_text"]
    tfidf_vectorizer = TfidfVectorizer(min_df=min_df, binary=False, max_df=max_df, stop_words='english',max_features=max_features)
    vectors = tfidf_vectorizer.fit_transform(df["all_text"])
    vocab = tfidf_vectorizer.vocabulary_
    print("vocab_size : %s"%str(len(vocab)))
    return vectors,vocab,tfidf_vectorizer

def get_df(path,extreme=False):
    """
    """
    rel_df = load_data(path)
    
    if extreme:
        rel_df = rel_df.loc[rel_df["source_partisan_score"] != -1 ]
        rel_df = rel_df.loc[rel_df["source_partisan_score"] != 1 ]
    
    rel_df["processed_title"] = preprocess_texts(text_lists=rel_df["title"])
    rel_df["all_text"] = rel_df["processed_title"] + " " +rel_df["processed_text"]
    
    print("Binary Partisan Distribution :\n%s"%str(rel_df["binary_ps"].value_counts().tolist()))
    
    rel_df["Num_Tokens"] = rel_df["all_text"].apply(lambda x: len(x.split(" ")))
    print("Avg Number of Tokens in Articles : %s"%str(rel_df["Num_Tokens"].mean()))
    print("Max Number of Tokens in Articles : %s"%str(rel_df["Num_Tokens"].max()))
    print("Min Number of Tokens in Articles : %s"%str(rel_df["Num_Tokens"].min()))
    
    print("Num Tokens Dist :\n")
    rel_df["Num_Tokens"].plot(kind="hist")
    
    return rel_df



def cluster_articles(df,num_clusters):
    """
    """
    vectors,vocab,tfidf_vectorizer = tfidf_vectorization(df=df,min_df=50,max_df=0.75,seed=RANDOM_SEED,max_features=20000,combine_text=False)
    clusters,cluster_clf = run_clustering(vectors=vectors,seed=RANDOM_SEED,num_clusters=num_clusters,clus_type="kmeans")
    cluster_sizes = get_cluster_sizes(cluster_clf)
    plot_size_dist(cluster_sizes)
    
    return clusters, cluster_clf,cluster_sizes

def map_art_2_clus(df,cluster_clf):
    """
    """
    doc_2_cluster_map = cluster2doc(num_texts=df.shape[0],cluster_labels=cluster_clf.labels_)
    return doc_2_cluster_map

def filter_clusters_size(cluster_sizes,min_size=300,max_size=3000):
    """
    """
    filtered_clusters = []
    for cluster in cluster_sizes.keys():
        if cluster_sizes[cluster] >= min_size and cluster_sizes[cluster] <= max_size:
            filtered_clusters.append(cluster)
    return filtered_clusters

def filter_by_partisan_dist(clusters,cluster_2_doc_map,df,partisan_dist_diff=0.20):
    """
    """
    filtered_clusters = []
    for c in tqdm(clusters,total=len(clusters)):
        c_docs_indices = cluster_2_doc_map[c]
        total_size = len(c_docs_indices)
        c_df = df["binary_ps"].iloc[c_docs_indices]
        ps_dist = c_df.value_counts().tolist()
        pos_dist = ps_dist[-1]/total_size
        neg_dist = ps_dist[0]/total_size
        dist_diff = np.abs(pos_dist - neg_dist)
        
        if dist_diff <= partisan_dist_diff:
            filtered_clusters.append(c)
    
    return filtered_clusters

def filter_clusters(df,cluster_sizes,doc_2_cluster_map):
    """
    """
    fil_clusters = filter_clusters_size(cluster_sizes,min_size=300,max_size=3000)
    fil_ps_clusters = filter_by_partisan_dist(clusters=fil_clusters,
                                          cluster_2_doc_map=doc_2_cluster_map,
                                          df=df,
                                          partisan_dist_diff=0.30)
    return fil_ps_clusters

def create_cluster_pairs(clusters):
    cluster_pairs = []
    for pair in itertools.combinations(clusters,2):
        
        cluster_pairs.append(pair)
    
    return cluster_pairs

def select_top_n_similar_cps(cps,cluster_clf,n=10):
    """
    """
    dist_matrix = cosine_similarity(cluster_clf.cluster_centers_)
    sorted_cps = sorted(cps,key=lambda x: dist_matrix[x[0],x[1]],reverse=True)[:n]
    return sorted_cps

def select_top_cluster_with_keywords(df,clusters,doc_2_cluster_map,keywords=["democrat","republican"]):
    """
    """
    cluster_scores = {}
    for c in tqdm(clusters,total=len(clusters)):
        c_docs = df["all_text"].iloc[doc_2_cluster_map[c]].tolist()
        keyword_doc_counts = defaultdict(int)
        for doc in c_docs:
            for k in keywords:
                if k in doc:
                    keyword_doc_counts[k]+=1
        cluster_scores[c] = sum([keyword_doc_counts[k] for k in keywords])/len(c_docs)
    
    cluster_score_tuples = sorted(list(cluster_scores.items()),key=lambda x:x[1],reverse=True)
    return cluster_score_tuples

def prepare_data(path,bad_terms,num_clusters=100,min_c_size=300,max_c_size=3000,partisan_dist_diff=0.3,extreme=False,return_data=True):
    """
    """
    print("********* 1. Loading Data **************")
    rel_df = get_df(path,extreme=extreme)
    
    rel_df = rel_df.reset_index(drop=True)
    
    print("***************** 2. Sampling Context Words *****************")
    tf_idf_vectors, tfidf_vectorizer = vectorize_text(text_list=rel_df["processed_title"])
    
    all_tokens = get_tokens(text_list=rel_df["processed_title"],batch_size=100,max_length=15)
    
    sampled_pos, sampled_neg = sample_context_words(rel_df,tf_idf_vectors,all_tokens,tfidf_vectorizer,bad_terms,sample_size=3,search_range=20)
    
    rel_df["context_pos_words"] = sampled_pos
    rel_df["context_neg_words"] = sampled_neg
    
    rel_df["context_pos_word_len"] = rel_df["context_pos_words"].apply(lambda x: len(x))
    
    print("Before Dropping Samples due to missing context_pos_words : %s" %str(rel_df.shape))
    
    rel_df = rel_df.loc[rel_df["context_pos_word_len"]>=1]
    
    print("After Dropping Samples due to missing context_pos_words : %s" %str(rel_df.shape))
    
    rel_df = rel_df.reset_index(drop=True)
    
    
    if extreme:
        save_obj(path="Data_4_AttM/rel_df_extreme.pkl",obj=rel_df)
    
    if not extreme:
        save_obj(path="Data_4_AttM/rel_df.pkl",obj=rel_df)
    
    
    
    print("*********** 3. Clustering Articles ************")
    clusters, cluster_clf,cluster_sizes = cluster_articles(df=rel_df,num_clusters=num_clusters)
    
    print("********** 4. Creating Doc 2 cluster Map ************")
    doc_2_cluster_map = map_art_2_clus(df=rel_df,cluster_clf=cluster_clf)
    
    if extreme:
        save_obj(path="Data_4_AttM/doc_2_cluster_map_extreme.pkl",obj=doc_2_cluster_map)
    
    if not extreme:
        save_obj(path="Data_4_AttM/doc_2_cluster_map.pkl",obj=doc_2_cluster_map)
    
    print("*********** 5. Filtering Clusters ****************")
    
    filtered_clusters = filter_clusters_size(cluster_sizes=cluster_sizes,min_size=min_c_size,max_size=max_c_size)
    
    filtered_clusters = filter_by_partisan_dist(filtered_clusters,doc_2_cluster_map,df=rel_df,partisan_dist_diff=partisan_dist_diff)
    
    print("Number of Filtered Clusters : %s"%str(len(filtered_clusters)))
    
    print("*********** 6. Scoring Clusters based on Keyword Precsence *************")
    cluster_score_tuples = select_top_cluster_with_keywords(df=rel_df,
                                                            clusters=filtered_clusters,
                                                            doc_2_cluster_map=doc_2_cluster_map,
                                                            keywords=["democrat","republican"])
    
    if extreme:
        save_obj(path="Data_4_AttM/cp_scored_extreme.pkl",obj=cluster_score_tuples)
    
    if not extreme:
        save_obj(path="Data_4_AttM/cp_scored.pkl",obj=cluster_score_tuples)
    
    if return_data:
        
        return doc_2_cluster_map, cluster_score_tuples, rel_df