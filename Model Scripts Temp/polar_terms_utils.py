"""
Needs to be updated with the Chi-square identification process instead of using logistic regression to identify general polar terms
"""
import pandas as pd
import numpy as np
import torch
import gc
import itertools
import tqdm
from bert_utils import load_tokenizer, load_model
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.tokenize import word_tokenize
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import plotly.express as px
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ENGLISH_STOP_WORDS = set( stopwords.words('english') ).union( set(ENGLISH_STOP_WORDS) )

def tokenize(text):
    """
    """
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
    return tokens

def count_words(X):
    XX = X.copy()
    XX.data /= XX.data
    return XX.sum(axis=0).A1 / X.shape[0]

def get_bad_terms(X, labels, vocab):
    """
    bad terms correlate with class label across all documents
    and are relatively frequent.
    We fit a simple logistic regression on all documents, 
    find top 100 terms per class, then select 100 most frequent from those.
    """
    clf = LogisticRegression(C=1, random_state=42, max_iter=1000)
    clf.fit(X, labels)
    c = clf.coef_[0]
    bad_terms = []
    feats = np.array(vocab.get_feature_names())
    for i in np.argsort(c)[::-1][:200]:
        bad_terms.append(feats[i])
    for i in np.argsort(c)[:200]:
        bad_terms.append(feats[i])
    bad_terms = set(bad_terms)    
    word_count = count_words(X)
    f = sorted([(word_count[vocab.vocabulary_[w]], w)  for w in bad_terms])[::-1][:200]
    return set(i[1] for i in f)

def match_bad_terms(tokens_train, tokens_test, bad_terms):    
    all_toks = set()
    for t in tokens_train:
        all_toks.update(t)
    for t in tokens_test:
        all_toks.update(t)
    return sorted(list(bad_terms & all_toks))

def get_bad_term_embeddings(bad_terms):
    """
    """
    tokenizer = load_tokenizer()
    model= load_model()
    model.to(torch.device('cuda:0'))
    bad_term_embeds = []
    bad_term_tokens = []
    for b in bad_terms:
        bert_tok = tokenizer.encode_plus(b, add_special_tokens=False, max_length = 1, padding='max_length', return_attention_mask = False,truncation=True, return_tensors = 'pt')
        
        token_id = bert_tok["input_ids"]
        token = tokenizer.convert_ids_to_tokens(token_id)
        bad_term_tokens.append(token)
        token_tensors = token_id.to(torch.device('cuda:0'))
        batch_out = model(token_tensors)
        batch_hidden_states = batch_out[2]
        batch_12_layer_tensor = batch_hidden_states[-1]
        token_embed = batch_12_layer_tensor.cpu()
        bad_term_embeds.append(token_embed)
    
    return bad_term_embeds,bad_term_tokens

def balance_sampling(df,source_min=2000):
    """
    """
    df_s_list = []
    # balance sampling by source
    for grp,df_grp in df.groupby("source"):
        min_size = min([source_min,df_grp.shape[0]])
        df_grp = df_grp.sample(n=min_size,random_state=42)
        df_s_list.append(df_grp)
    
    df_o = pd.concat(df_s_list,axis=0).reset_index(drop=True)
    # balanced sampling by partisan score
    df_lib = df_o.loc[df_o["binary_ps"] == 0]
    df_cons = df_o.loc[df_o["binary_ps"] == 1]
    
    min_ps = min([df_lib.shape[0],df_cons.shape[0]])
    
    df_lib = df_lib.sample(n=min_ps,random_state=42)
    df_cons = df_cons.sample(n=min_ps,random_state=42)
    
    df_balanced = pd.concat([df_lib,df_cons]).reset_index(drop=True)
    df_balanced = df_balanced.sample(frac=1.0,random_state=42)
    
    return df_balanced

def select_polar_terms_chi2(df_balanced,tokenizer):
    """
    """
    tfidf_vectorizer = TfidfVectorizer(min_df=500, binary=True, max_df=0.80, stop_words=ENGLISH_STOP_WORDS,max_features=None,tokenizer=tokenizer)
    X = tfidf_vectorizer.fit_transform(df_balanced["all_text"])

    vocab = tfidf_vectorizer.get_feature_names()

    chi2_feats = SelectKBest(chi2, k=200)
    feats = chi2_feats.fit_transform(X, df_balanced["binary_ps"])
    feat_indices = chi2_feats.get_support(indices=True)

    selected_words = np.array(vocab)[feat_indices]
    
    return selected_words

def get_doc_map(terms,tokenized_text):
    """
    """
    term_map = defaultdict(list)
    for term in terms:
        for text_id,text in enumerate(tokenized_text):
            if term in text:
                term_map[term].append(text_id)
    
    return term_map

def get_avg_cosine_sim(terms,df,doc_map,sample_size=100):
    """
    """
    # first get bert embed of the term in text after using text as input 
    # second get pairs of docs to find cosine sim betweem
    # third average the cosine sim scores
    # return the averaged cosine sim scores
    avg_cosine_sims = []
    final_terms = []
    tokenizer = load_tokenizer()
    model= load_model()
    model.to(torch.device('cuda:1'))
    
    for term in tqdm(terms,total=len(terms)):
        texts = df["all_text"].iloc[doc_map[term]].sample(n=sample_size,random_state=42).tolist()
        token_embeds = []

        for t in texts:
            bert_tokens = tokenizer.encode_plus(t, add_special_tokens=True, max_length = 300, padding='max_length', return_attention_mask = False,truncation=True, return_tensors = 'pt')
            token_ids = bert_tokens["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(token_ids.flatten())
            term_index = -1
            try:
                term_index = tokens.index(term)
            except:
                continue

            if term_index != -1:

                token_tensors = token_ids.to(torch.device('cuda:1'))
                batch_out = model(token_tensors)
                batch_hidden_states = batch_out[2]
                batch_12_layer_tensor = batch_hidden_states[-1][:,term_index,:]
                token_embed = batch_12_layer_tensor.detach().cpu().numpy()
                token_embeds.append(token_embed)


        doc_pairs = [pair for pair in itertools.combinations([i for i in range(len(token_embeds))],2)]
        cosine_sims = [cosine_similarity(token_embeds[d[0]],token_embeds[d[1]]) for d in doc_pairs]
        if np.mean(cosine_sims) != np.nan:
            avg_cosine_sims.append(np.mean(cosine_sims))
            final_terms.append(term)
    
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    return avg_cosine_sims,final_terms

def filter_lists(a,b):
    """
    """
    k,p = [],[]
    for i,j in zip(a,b):
        if not math.isnan(j):
            k.append(i)
            p.append(j)
    
    return k,p

def plot_cosine_sim_vs_selec_scores(terms,cosine_sims,scores):
    """
    """
    scores_ = []
    for t in terms:
        for s in scores:
            if s[0] == t:
                scores_.append(s[1])
    
    df_ = pd.DataFrame()
    df_["Terms"] = terms
    df_["Average Cosine Similarity"] = cosine_sims
    df_["Score"] = scores_

    fig = px.scatter(df_, x="Score", y="Average Cosine Similarity", log_x=False,
                     hover_name="Terms",width=1000, height=900)
    
    fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    
    fig.show()