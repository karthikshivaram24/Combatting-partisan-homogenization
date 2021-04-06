from general_utils import timer

from config import RANDOM_SEED

from preprocess_utils import preprocess_texts, tfidf_vectorization, dimensionality_reduction

from clustering_utils import run_clustering, get_cluster_sizes, score_cluster, get_cluster_pairs, get_pairwise_dist, cluster2doc, filter_clusters, get_top_100_clusterpairs

from data_utils import load_data, sample_data, balanced_sampling, create_train_test

from bert_embeddings import get_tokens_from_bert_tokenizer

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
import itertools
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
import pickle
import random

import torch
import pickle
from transformers import BertModel, BertTokenizer
from general_utils import timer
import os
import numpy as np
from bert_utils import load_model, load_tokenizer, batch_text_gen
import torch
from collections import defaultdict
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


def get_kw_bert_rep(kw_list,layer_map,bert_tokenizer,bert_model,bs=10):
    """
    """
    batch_count = 0
    for kw_batch in batch_text_gen(kw_list,batch_size = bs):
        print("Batch No : %s" %str(batch_count))
        tokenized_kw_batch, encoded_batch_ids = tokenize_bert_single_token(kw_batch,bert_tokenizer)
        batch_out = bert_model(input_ids=tokenized_kw_batch)
        batch_hidden_states = batch_out[2] # 13 layers as layer(0) is input to the transformer 
        
        for layer_ind in range(13):
            for t_id, token in enumerate(kw_batch):
                token_hidden_state = batch_hidden_states[layer_ind][t_id,:,:].cpu().detach().numpy()
                # index here to drop the pad tokens and only consider the actual word representation (if subwords then avg the subwords to get word representation) (PAD has token id of 0)
                token_hidden_state = combine_subwords(token_hidden_state,encoded_batch_ids[t_id])
                layer_map[token]["layer_%s"%str(layer_ind)] = token_hidden_state
        batch_count+=1 
        
def tokenize_bert_single_token(token_batch,tokenizer,debug_flag=False):
    """
    """
    encoded_batch = []
    for token in token_batch:
        encoded_token = tokenizer.encode(token,add_special_tokens=False,padding="max_length",max_length=10) # padding is longest because subwords exist in the vocab so the tokenizer splits a given word into subwords
        if debug_flag:
            print(token)
            print(tokenizer.convert_ids_to_tokens(encoded_token))
        encoded_batch.append(encoded_token)
    
    tokenized_batch = torch.LongTensor(encoded_batch)
    tokenized_tensor = tokenized_batch.to('cuda')
    return tokenized_tensor, encoded_batch

def compare_bert_rep_token(kw_topic_specific,kw_topic_indep,plot=True):
    """
    """
    bert_mod = load_model()
    bert_tok = load_tokenizer()
    
    bert_map = defaultdict(lambda :defaultdict())
    
    # topic specific keywords
    print("Getting bert representation for Topic Specific Keywords ... \n")
    batch_size = get_batch_size(kw_topic_specific)
    
    get_kw_bert_rep(kw_list=kw_topic_specific,
                    layer_map=bert_map,
                    bert_tokenizer=bert_tok,
                    bert_model=bert_mod,
                    bs=batch_size)
    
    # topic indep keywords
    print("Getting bert representation for Topic Independent Keywords ... \n")
    batch_size = get_batch_size(kw_topic_indep)
    
    get_kw_bert_rep(kw_list=kw_topic_indep,
                    layer_map=bert_map,
                    bert_tokenizer=bert_tok,
                    bert_model=bert_mod,
                    bs=batch_size)
    
    # cosin_sim
    cosine_sims_layer_map = defaultdict(list)
    pair_count=0
    for wp in itertools.product(kw_topic_indep,kw_topic_specific):
        for layer_id in range(13):
            cos_sim = cosine_similarity(bert_map[wp[0]]["layer_%s"%str(layer_id)],bert_map[wp[1]]["layer_%s"%str(layer_id)])
            cosine_sims_layer_map[layer_id].append(cos_sim)
        pair_count+=1
    
    print("No of Pairs : %s" %str(pair_count))
    
    # avg
    avg_scores = []
    for layer in sorted(cosine_sims_layer_map.keys()):
        avg_scores.append(np.mean(cosine_sims_layer_map[layer]))
    
    # plot
    if plot:
        fig,ax = plt.subplots(1,1,figsize=(10,8))
        ax.scatter(range(13),avg_scores,s=100)
        ax.set_xlabel("Bert Layers")
        ax.set_ylabel("Avg Cosine Similarity")
        ax.set_xticks(range(-1,13,1))
        ax.set_xlim(-1,13)
        ax.set_ylim(0.0,1.0)
        sns.regplot(x=[i for i in range(13)], y=avg_scores,ax=ax,color="cornflowerblue")
        plt.title("Average Cosine Similarity between Topic Independant Keywords and Topic Specific Keywords\n for Representations from different Layers of BERT")
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.show()
    return bert_map,avg_scores
        
def get_batch_size(kw_list,ratio=10):
    """
    """
    batch_size = int(len(kw_list)/ratio)
    
    if len(kw_list) <=25:
        batch_size = len(kw_list)
    
    return batch_size

def combine_subwords(hidden_state,token_ids):
    """
    hidden_state_shape = (maxlength_padding,768)
    
    choose token_ids that are not 0
    
    """
    subword_arrs = []
    for t_i,t in enumerate(token_ids):
        if t != 0:
            subword_arrs.append(hidden_state[t_i,:])
    
    return np.mean(subword_arrs,axis=0).reshape(1,-1)

@timer
def get_all_clustered_docs(keywords,clusters,doc_2_cluster_map,sample_df,partisan_score=0):
    """
    """
    relv_docs_cluster_map = defaultdict(lambda : defaultdict(list))
    keyword_index_cluster_map = defaultdict(lambda : defaultdict(list))
    total_docs_per_cluser = defaultdict(lambda : defaultdict(int))
    tokenizer = load_tokenizer()
    for c in clusters:
        print("\nCluster : %s" %str(c))
        docs_indexes = doc_2_cluster_map[c]
        print(len(docs_indexes))
        texts = sample_df["all_text"].iloc[docs_indexes].tolist()
        print(len(texts))
        partisan_scores = sample_df["binary_ps"].iloc[docs_indexes].tolist()
        print(len(partisan_scores))
        for k in keywords:
            print("Keyword : %s" %str(k))
            for ind_t,t in enumerate(texts):
                if k in t.lower() and partisan_scores[ind_t]==partisan_score:
                    for ind,w in enumerate(get_tokens_from_bert_tokenizer(t.lower(),tokenizer)):
                        if  w == k or k in w:
                            if ind < 500:
                                keyword_index_cluster_map[k][c].append(ind)
                                relv_docs_cluster_map[k][c].append(t)
        
            total_docs_per_cluser[k][c] = len(relv_docs_cluster_map[k][c])
    
    print(total_docs_per_cluser)
    
    return relv_docs_cluster_map, keyword_index_cluster_map, total_docs_per_cluser
        
def batch_text_gen_ind(text_list,index_list,batch_size=2):
    """
    """
    for ndx in range(0,len(text_list),batch_size):
        yield (text_list[ndx:min(ndx+batch_size,len(text_list))], index_list[ndx:min(ndx+batch_size,len(text_list))])

def tokenize_for_bert(text_batch,tokenizer):
    """
    """
    tokenized_tensor = torch.LongTensor([tokenizer.encode(text,
                                                          truncation=True,
                                                          padding="max_length",
                                                          max_length=500, 
                                                          add_special_tokens=False)  # Add [CLS] and [SEP],) 
                                                          for text in text_batch])
    tokenized_tensor = tokenized_tensor.to('cuda')
    return tokenized_tensor


@timer
def get_embeddings(keywords,relv_docs_cluster_map,keyword_index_cluster_map):
    """
    here i am only considering the last occurence of a keyword in a given document
    """
    bert_tokenizer = load_tokenizer()
    bert_model = load_model()
    layer_keyword_cluster_map = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    
    for keyword in keywords:
        print("\nKeyword : %s" %keyword)
        for cluster in relv_docs_cluster_map[keyword].keys():
            print("Cluster : %s" %str(cluster))
            cluster_docs = relv_docs_cluster_map[keyword][cluster]
            if len(cluster_docs) > 0:
                keyword_indices = keyword_index_cluster_map[keyword][cluster]
                for cluster_doc_batch, keyword_indices_batch in batch_text_gen_ind(cluster_docs,keyword_indices,batch_size=2):
                    try:
                        tokenized_batch_text = tokenize_for_bert(text_batch=cluster_doc_batch,tokenizer=bert_tokenizer)
                        batch_out = bert_model(input_ids=tokenized_batch_text)
                        batch_hidden_states = batch_out[2] # 13 layers as layer(0) is input to the transformer 

                        for layer in range(13):
                            temp_reps = []
                            for ind,k_ind in enumerate(keyword_indices_batch):
                                temp_reps.append(batch_hidden_states[layer][ind,k_ind,:].cpu().detach().numpy())
                            layer_keyword_cluster_map[keyword][cluster][layer]+=temp_reps
                
                    except Exception as e:
                        print("Error Raised : \n")
                        print(e)
                    

    return layer_keyword_cluster_map


def take_mean(layer_keyword_cluster_map):
    """
    """
    for keyword in layer_keyword_cluster_map.keys():
        for cluster in layer_keyword_cluster_map[keyword].keys():
            for layer in layer_keyword_cluster_map[keyword][cluster].keys():
                list_batch_outs = layer_keyword_cluster_map[keyword][cluster][layer]
                layer_keyword_cluster_map[keyword][cluster][layer] = np.mean(list_batch_outs,axis=0).reshape(1,-1)

def plot_keyword_cosine_sim_single_layer(layer_keyword_cluster_map,keyword,avg_rp_word_sim):
    """
    """
    clusters = layer_keyword_cluster_map[keyword].keys()
    
    cluster_pairs = list(itertools.combinations(clusters,2))
    fig,ax = plt.subplots(1,1,figsize=(30,12))
    colors_map = sns.color_palette("tab20", n_colors=13)
    
    cosine_sims_map ={}
    
    for layer in range(1,13,1):
        
        cosine_sims = []
        x = [i for i in range(len(cluster_pairs))]
        for cp in cluster_pairs:
            cosine_sim = cosine_similarity(layer_keyword_cluster_map[keyword][cp[0]][layer],layer_keyword_cluster_map[keyword][cp[1]][layer])
            temp = (cosine_sim/avg_rp_word_sim[layer])[0]
            cosine_sims.append(temp)
        
        temp_cos = np.concatenate(cosine_sims,axis=0)
        indices = np.argsort(temp_cos)
        cosine_sims_map[layer]=[cluster_pairs[i] for i in indices]
        
        
        ax.scatter(x,cosine_sims,color=colors_map[layer],s=50,label="layer_%s"%str(layer))
#         sns.regplot(x=x, y=cosine_sims,ax=ax,color=colors_map[layer])
    
    ax.set_xlabel("Cluster-Pairs")
    ax.set_ylabel("Cosine Similarity")
#     ax.set_ylim(0.0,1.0)
#     ax.set_xticks([i for i in range(len(cluster_pairs))])
#     ax.set_xticklabels([ str(cp) for cp in cluster_pairs],rotation = 90)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Average Cosine Similarity between Bert Representations of **** %s **** for each layer"%str(keyword))
    plt.grid(False, linestyle='--')
    plt.tight_layout()
    plt.show()
    
    # get the lowest scoring cluster pairs
    return cosine_sims_map

def analyze_cluster_pairs(cps,relv_docs_cluster_map,keyword):
    """
    """
    tokenizer = load_tokenizer()
    for cp in cps:
        print("\n********* Analyzing Cluster Pair : %s************\n" %str(cp))
        cluster_docs_with_keyword_1 = relv_docs_cluster_map[keyword][cp[0]]
        cluster_docs_with_keyword_2 = relv_docs_cluster_map[keyword][cp[1]]
        
        print("Cluster %s has %s docs that contain %s keyword"%(str(cp[0]),len(cluster_docs_with_keyword_1),keyword))
        print("Cluster %s has %s docs that contain %s keyword"%(str(cp[1]),len(cluster_docs_with_keyword_2),keyword))
        
        print("\nRandom Docs from Cluster : %s"%str(cp[0]))
        sample_size = 5 
        if len(cluster_docs_with_keyword_1) <= sample_size:
            sample_size = len(cluster_docs_with_keyword_1)
        sample_cdwk1 = random.sample(cluster_docs_with_keyword_1,sample_size)
        for doc in sample_cdwk1:
            print("\n")
            print(doc)
            tokens,tok_inds = get_tokens_from_bert_tokenizer(doc,tokenizer)
            occurences = 0
            for tok_index,tok in enumerate(tokens):
                if tok == keyword:
                    print("\n5 Words Before %s :"%str(keyword))
                    print(tokens[tok_index-6:tok_index])
                    print("\n5 Words After %s :"%str(keyword))
                    print(tokens[tok_index:tok_index+6])
                    occurences +=1
            
            print("\n%s occurred %s times in the above document"%(keyword,str(occurences)))
        
        print("\nRandom Docs from Cluster : %s"%str(cp[1]))
        sample_size = 5 
        if len(cluster_docs_with_keyword_2) <= sample_size:
            sample_size = len(cluster_docs_with_keyword_2)
        sample_cdwk1 = random.sample(cluster_docs_with_keyword_2,sample_size)
        for doc in sample_cdwk1:
            print("\n")
            print(doc)
            tokens,tok_inds = get_tokens_from_bert_tokenizer(doc,tokenizer)
            occurences = 0
            for tok_index,tok in enumerate(tokens):
                if tok == keyword:
                    print("\n5 Words Before %s :"%str(keyword))
                    print(tokens[tok_index-5:tok_index+1])
                    print("\n5 Words After %s :"%str(keyword))
                    print(tokens[tok_index-1:tok_index+5])
                    occurences +=1
            
            print("\n%s occurred %s times in the above document"%(keyword,str(occurences)))
        

def plot_keyword_avg_cos_sim_all_layers(layer_keyword_cluster_map,keyword,avg_rp_word_sim):
    """
    """
    clusters = layer_keyword_cluster_map[keyword].keys()
    
    cluster_pairs = list(itertools.combinations(clusters,2))
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    colors_map = sns.color_palette("tab20", n_colors=13)
    
    for layer in range(1,13,1):
        
        cosine_sims = []
        x = [i for i in range(len(cluster_pairs))]
        for cp in cluster_pairs:
            cosine_sim = cosine_similarity(layer_keyword_cluster_map[keyword][cp[0]][layer],layer_keyword_cluster_map[keyword][cp[1]][layer])
            cosine_sims.append(cosine_sim/avg_rp_word_sim[layer])
    
        avg_cosine_sim = np.mean(cosine_sims)
        
        ax.scatter(layer,avg_cosine_sim,color=colors_map[layer],s=100,label="layer_%s"%str(layer))
    
    ax.set_xlabel("Bert Layer Representations")
    ax.set_ylabel("Average Cosine Similarity of a Keyword Across all Cluster Pairs")
#     ax.set_ylim(0.0,1.0)
    plt.legend()
    plt.title("Average Cosine Similarity of a Keyword Across all Cluster Pairs vs Bert Representations across all layers")
    plt.grid(False, linestyle='--')
    plt.tight_layout()
    plt.show()
    
def plot_keyword_cluster_dist(keyword_cluster_dist_map):
    """
    """
    fig,ax = plt.subplots(len(keyword_cluster_dist_map.keys()),1,figsize=(20,20))
    axes = ax.ravel()
    
    for k_ind,keyword in enumerate(keyword_cluster_dist_map.keys()):
        dist = [keyword_cluster_dist_map[keyword][cluster] for cluster in sorted(keyword_cluster_dist_map[keyword].keys())]
        clusters = [cluster for cluster in sorted(keyword_cluster_dist_map[keyword].keys())]
        
        colors = np.where(np.array(dist)<10,'r','b')
        
        axes[k_ind].scatter(range(len(clusters)),dist,c=colors,s=100)
        axes[k_ind].set_xlabel("Clusters")
        axes[k_ind].set_ylabel("Number of Docs containing %s"%str(keyword))
        axes[k_ind].set_xticks([i for i in range(len(clusters))])
        axes[k_ind].set_xticklabels(clusters)
    
    plt.show()
        
        
        