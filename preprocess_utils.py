from general_utils import timer
from functools import partial
from joblib import Parallel, delayed
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from config import RANDOM_SEED, min_df, max_df, GLOVE_PATH, W2V_PATH, ELMO_PATH, FASTTEXT_PATH
import re
import pandas as pd
from pymagnitude import *

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
def tfidf_vectorization(df,min_df=30,max_df=0.75,seed=RANDOM_SEED):
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
    df["all_text"] = df["title"] + " " + df["processed_text"]
    tfidf_vectorizer = TfidfVectorizer(min_df=min_df, binary=False, max_df=max_df, stop_words='english')
    vectors = tfidf_vectorizer.fit_transform(df["all_text"])
    vocab = tfidf_vectorizer.vocabulary_
    print("vocab_size : %s"%str(len(vocab)))
    return vectors,vocab,tfidf_vectorizer

@timer
def dimensionality_reduction(vectors,mode="PCA",dim=500,seed=RANDOM_SEED):
    """
    function that reduces the dimension of the given vectors using PCA,SVD or UMAP
    
    Parameters:
    -----------
    * vectors -> CSR or DENSE numpy matrix
    * dim -> int, The new dimensions of the reduced vectors
    * mode -> String, Type of dimensionality reduction technique to use
    * seed -> int, random seed used to initialize the reduction models
    
    Returns:
    -------
    * vectors -> numpy vectors
    """
    print("\nShape Before DIM REDUC : %s" %str(vectors.shape))
    if mode == "PCA":
        pca = PCA(n_components=dim,svd_solver="arpack",random_state=seed)
        reduce_vectors = pca.fit_transform(vectors.todense())
        print("Shape After DIM REDUC : %s" %str(vectors.shape))
        return reduce_vectors
    if mode == "SVD_LSA":
        tsvd = TruncatedSVD(n_components=dim,algorithm="arpack",random_state=seed)
        reduce_vectors = tsvd.fit_transform(vectors)
        print("Shape After DIM REDUC : %s" %str(vectors.shape))
        return reduce_vectors
    
    if mode == "UMAP":
        return None
    
    if mode == "None":
        return None

# def load_glove_lookup(filepath=GLOVE_PATH):
#     """
#     """
#     print("Loading Glove Model")
#     f = open(filepath,'r',encoding='utf-8')
#     glove_lookup = {}
#     for line in f:
#         splitLines = line.split()
#         word = splitLines[0]
#         wordEmbedding = np.array([float(value) for value in splitLines[1:]])
#         glove_lookup[word] = wordEmbedding
#     print("Finished Loading Glove : %s words exist" %str(len(glove_lookup)))
#     return glove_lookup


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
    if mode == "elmo":
        return text_2_elmo(texts=df["all_text"].tolist())
