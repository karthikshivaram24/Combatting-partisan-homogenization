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
from pymagnitude import *

from .general_utils import timer
from .settings import MIN_DF, MAX_DF, GLOVE_PATH, W2V_PATH, ELMO_PATH, BERT_PATH, FASTTEXT_PATH, RANDOM_SEED
 


@timer
def preprocess_texts(text_lists):
    """
    Runs the text preprocessing pipeline for our news articles before vectorization
    """
    
    def select_first10(x):
        return " ".join(sent_tokenize(x)[:10])
    
    def to_lower(x):
        return x.lower()
    
    def remove_punc(x):
        return re.sub(r'[^\w\s]', '  ', x)
    
    def remove_small_words(x):
        return re.sub(r'\b\w{1,2}\b', '', x)
    
    def remove_spaces(x):
        return re.sub(' +', ' ', x)
    
    preprocess_pipe = [select_first10,to_lower,remove_punc,remove_small_words,remove_spaces]
    
    processed_texts = text_lists
    for preprocess_func in preprocess_pipe:
        print("Running : %s" %str(preprocess_func.__name__))
        processed_texts = Parallel(n_jobs=-1)(delayed(preprocess_func)(x) for x in processed_texts)

    return processed_texts


@timer
def vectorization(df,min_df=30,max_df=0.75,seed=RANDOM_SEED):
    """
    Performs Vectorization given a set of text articles
    """
    df["all_text"] = df["title"] + " " + df["processed_text"]
    tfidf_vectorizer = TfidfVectorizer(min_df=min_df, binary=False, max_df=max_df, stop_words='english')
    vectors = tfidf_vectorizer.fit_transform(df["all_text"])
    vocab = tfidf_vectorizer.vocabulary_
    print("vocab_size : %s"%str(len(vocab)))
    return vectors,vocab,tfidf_vectorizer

def load_glove_lookup(filepath=GLOVE_PATH):
    """
    """
    print("Loading Glove Model")
    f = open(filepath,'r',encoding='utf-8')
    glove_lookup = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        glove_lookup[word] = wordEmbedding
    print("Finished Loading Glove : %s words exist" %str(len(glove_lookup)))
    return glove_lookup

def text_2_tfidf(text,min_df=MIN_DF,max_df=MAX_DF):
    """
    """
    tfidf_vectorizer = TfidfVectorizer(min_df=min_df, binary=False, max_df=max_df, stop_words='english')
    vectors = tfidf_vectorizer.fit_transform(text)
    vocab = tfidf_vectorizer.vocabulary_
    print("vocab_size : %s"%str(len(vocab)))
    return vectors,vocab,tfidf_vectorizer


def get_embeddings(texts,path,merge_strategy):
    """
    """
    vectors = Magnitude(path)
    text_vectors = []
    for ind,text in enumerate(texts):
        tokens = word_tokenize(text)
        sent_2d = vectors.query(tokens)
        if merge_strategy == "mean":
            sent_2d = np.mean(sent_2d,axis=0).reshape(1,-1)
        
        if merge_strategy == "max":
            sent_2d = np.amax(sent_2d,axis=0).reshape(1,-1)
        
        text_vectors.append(sent_2d)
        if ind >= 10000 and ind%10000 == 0:
            print("Completed - %s" %str(ind))
    
    print("Number of Vectors : %s"%str(len(text_vectors)))
    print("Dimension of Single Vector: %s" %str(text_vectors[0].shape))
    return np.concatenate(text_vectors,axis=0)


@timer
def text_2_w2v(texts,merge_strategy="mean"):
    """
    """
    return get_embeddings(texts=texts,
                          path=W2V_PATH,
                          merge_strategy=merge_strategy)
            
            
@timer
def text_2_glove(texts,merge_strategy="mean"):
    """
    """
    return get_embeddings(texts=texts,
                          path=GLOVE_PATH,
                          merge_strategy=merge_strategy)

@timer
def text_2_fasttext(texts,merge_strategy="mean"):
    """
    """
    return get_embeddings(texts=texts,
                          path=FASTTEXT_PATH,
                          merge_strategy=merge_strategy)

@timer
def text_2_bert(texts,merge_strategy="mean"):
    """
    """
    return get_embeddings(texts=texts,
                          path=BERT_PATH,
                          merge_strategy=merge_strategy)

@timer
def text_2_elmo(texts,merge_strategy="mean"):
    """
    """
    return get_embeddings(texts=texts,
                          path=ELMO_PATH,
                          merge_strategy=merge_strategy)



@timer
def vectorize_text(df,mode="tf-idf"):
    """
    """
    df["all_text"] = df["title"] + " " + df["processed_text"]
    
    if mode == "tf-idf":
        return text_2_tfidf(texts=df["all_text"])
    if mode == "w2v":
        return text_2_w2v(texts=df["all_text"].tolist())
    if mode == "glove":
        return text_2_glove(texts=df["all_text"].tolist())
    if mode == "fasttext":
        return text_2_fasttext(texts=df["all_text"].tolist())
    if mode == "bert":
        return text_2_bert(texts=df["all_text"].tolist())
    if mode == "elmo":
        return text_2_elmo(texts=df["all_text"].tolist())


@timer
def dimensionality_reduction(vectors,mode="PCA",dim=500,seed=RANDOM_SEED):
    """
    """
    print("\nShape Before DIM REDUC : %s" %str(vectors.shape))
    if mode == "PCA":
        pca = PCA(n_components=dim,svd_solver="arpack",random_state=seed)
        vectors = pca.fit_transform(vectors.todense())
        print("Shape After DIM REDUC : %s" %str(vectors.shape))
        return vectors
    if mode == "SVD_LSA":
        tsvd = TruncatedSVD(n_components=dim,algorithm="arpack",random_state=seed)
        vectors = tsvd.fit_transform(vectors)
        print("Shape After DIM REDUC : %s" %str(vectors.shape))
        return vectors
    
    if mode == "UMAP":
        return None
    
    if mode == "None":
        return None