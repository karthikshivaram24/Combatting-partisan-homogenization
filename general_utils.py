
from nltk.tokenize import sent_tokenize, word_tokenize
import functools
import time
import joblib
from joblib import Parallel, delayed
from functools import wraps
from time import time
import itertools
from functools import partial
import functools
import time
import warnings
import re

def timer(func):
    """
    Decorator to time a given function

    Parameters
    ----------
    func : generic
        The function to time

    Raises
    ------
    No Exceptions

    Returns
    -------
    value : generic
        The return value from func

    """
    @functools.wraps(func)
    def wrapper_timer(*args,**kwargs):
        start = time.perf_counter()
        value = func(*args,**kwargs)
        stop = time.perf_counter()
        run_time = stop - start
        print(f"\nFinished running {func.__name__!r} in {run_time/60.0:.4f} mins\n")
        return value
    return wrapper_timer

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
def load_data(path):
    """
    Loads our news articles into a dataframe, filters out articles with neutral partisan score
    and creates a binary partisan score for liberal and conservative articles
    """
    df = pd.read_csv(path)
    print(df.columns)
    print("Df original shape : %s" %str(df.shape))
    # drop rows with text as nan
    df = df[df['text'].notna()]
    print("Df shape after dropping nan text : %s" %str(df.shape))
    # drop articles that have stance = 0
    df = df[df["source_partisan_score"] != 0]
    print("Df shape after dropping 0 stance articles : %s" %str(df.shape))
    # convert articles of stance -1,+1,-2,+2
    df["binary_ps"] = df["source_partisan_score"].apply(lambda x: 1 if x>=1 else 0)
    
    return df

@timer
def sample_data(df,sample_size=0.2,seed=RANDOM_SEED):
    """
    Samples a part of the entire news article collection
    """
    return df.sample(frac=sample_size, replace=False, random_state=seed)