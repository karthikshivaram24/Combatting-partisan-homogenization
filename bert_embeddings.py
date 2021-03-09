from general_utils import timer
from config import RANDOM_SEED
from preprocess_utils import preprocess_texts, tfidf_vectorization, dimensionality_reduction
from clustering_utils import run_clustering, get_cluster_sizes, score_cluster, get_cluster_pairs, get_pairwise_dist, cluster2doc, filter_clusters, get_top_100_clusterpairs
from data_utils import load_data, sample_data, balanced_sampling, create_train_test
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
import itertools
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
import pickle
import torch
import pickle
from transformers import BertModel, BertTokenizer
from general_utils import timer
import os
from bert_utils import load_model, load_tokenizer, batch_text_gen, tokenize_bert, load_bert_output
import torch
import pickle
from general_utils import timer




def load_pkl_file(file_path):
    """
    """
    article_dict = pickle.load(open(file_path,"rb"))
    articles = []
    for article in article_dict.keys():
        articles.append(article_dict[article])
    
    print("Number of Articles : %s"%str(len(articles)))
    articles_df = pd.DataFrame(articles)
    print("Shape Before Processing : %s" %str(articles_df.shape))
    #drop columns
    articles_df.drop(columns=["article_id",
                              "url",
                              "source",
                              "tweet_id",
                              "tweet_text",
                              "kws_label",
                              "cls_label",
                              "tweet_screen_name",
                              "tweet_created_at"],inplace=True)
    #reset index
    articles_df.reset_index(inplace=True,drop=True)
    #drop partisan of 0.0
    articles_df = articles_df.loc[articles_df["source_partisan_score"] != 0.0]
    articles_df["binary_ps"] = articles_df["source_partisan_score"].apply(lambda x: 1 if x>0 else 0)
    print("Shape after dropping Neutral Articles : %s" %str(articles_df.shape))
    print(articles_df.columns)
    return articles_df


@timer
def save_and_infer_bert_embeddings(text_list,batch_size=50,save_path="/media/karthikshivaram/SABER_4TB"):
    """
    doc_index = (batch_index) * batch_size + index_in_batch
    eg: 3rd doc in batch 1(i.e. batch 2) (where batch_size = 50) (should have index 52)
    doc_index = (1 * 50) + 2 = 52
    
    1st doc in batch (i.e batch 0) should have index 0
    doc_index = (0*50) + 0 = 0
    
    50th doc in batch 1 should have index 49
    doc_index = (0*50) + 49 = 49
    
    5th doc in batcch 3 with batch_index 2 should have index 104
    doc_index = (2*50) + 4 = 104
    """
    model = load_model()
    tokenizer = load_tokenizer()
    num_batches = int(len(text_list)/batch_size)
    print("Number of Batches : %s\n"%str(num_batches))
    
    with torch.no_grad():
        batch_no = 0
        for text_batch in batch_text_gen(text_list,batch_size):
            if batch_no > 0 and batch_no % 100 == 0:
                print("Running Batch : %s"%str(batch_no))
            batch_tensor = tokenize_bert(text_batch,tokenizer)
            batch_out = model(input_ids=batch_tensor)
            batch_hidden_states = batch_out[2]
            batch_all_layer_tensor = torch.cat(batch_hidden_states[1:],2)
            
            np.save("%s/%s.npy"%(save_path,str(batch_no)),batch_all_layer_tensor.cpu().numpy())
            
            batch_no +=1
    
    print("\nTotal Batches Saved : %s" %str(batch_no))
    print("Finished Inferring and saving Embeddings .....")

@timer
def load_embeddings_for_word_usecase(saved_path,batch_size,doc_indices,layer):
    """
    returns vector for article of size (500,768)
    """
    layers_start = [i* 768  for i in range(12)]
    layers_stop = [i+768 for i in layers_start]
    
    doc_outputs = []
    
    def get_batch_arr(file,layer):
        """
        """
        batch_arr = np.load(f)
        # 3d matrix [batch_size,max_length,all_12_layer_output (12*768)]
        layer_slice_start = layers_start[layer-1]
        layer_slice_stop = layers_stop[layer-1]
        # get [batch_size,max_length,layer_output(768)]
        batch_layer_slice = batch_arr[:,:,layer_slice_start:layer_slice_stop]
        return batch_layer_slice
    
    for doc_index in doc_indices:
        batch_num = int(doc_index/batch_size)
        doc_index_in_batch = doc_index - (batch_num*batch_size)
        file = save_path + os.path.sep + str(batch_num)+".npy"
        
        #(batch_size,max_length,768)
        batch_arr = get_batch_arr(file,layer=layer)
        doc_arr = batch_arr[doc_index_in_batch,:,:]
        
        doc_outputs.append(doc_arr)
    
    return doc_outputs

def get_tokens_from_bert_tokenizer(text,tokenizer):
    """
    """
    tokenized_tensor = tokenizer.encode(text,
                                        truncation=True,
                                        padding="max_length",
                                        max_length=500, 
                                        add_special_tokens=False)  # Add [CLS] and [SEP],) 
    
    
    actual_tokens = tokenizer.convert_ids_to_tokens(tokenized_tensor)
    actual_tokens = [ a for a in actual_tokens if a != "[PAD]"]
    
    return actual_tokens,[i for i in range(len(actual_tokens))]

def load_bert_with_cv(text_list,bert_reps,batch_size,layer,cv,aggregation):
    """
    steps:
    ------
    1) Tokenize all docs using bert tokenizer and get vocab
    2) for each token get a list of documents that contain it and a list that has corresponding index of that token
    3) Sample 100-cv % of the documents for each token and get it's corresponding representation from that document tensor
    4) Average over these to get a map of token to representation
    5) build the representation for each document using the above by aggregation method
    """
    token_doc_map = defaultdict(lambda : defaultdict(list))
    token_bert_reps_in_doc_map = defaultdict(list)
    token_reps = defaultdict()
    article_reps = []
    
    tokenizer = load_tokenizer()
    
    for article_id,article in enumerate(text_list):
        tokens,token_indices = get_tokens_from_bert_tokenizer(article,tokenizer)
        # ADD THE cv SAMPLING HERE
        for token_id , token in enumerate(tokens):
            start = token_indices[token_id]
            token_bert_reps_in_doc_map[token].append(bert_reps[article_id,start,:])
            
    print("\nFinished BERT Tokenization")
    
    for token in token_bert_reps_in_doc_map.keys():
        token_reps[token] = np.mean(token_bert_reps_in_doc_map[token],axis=0)
        
    print("\nFinished Taking Means of token reps")
    
    for article_id,article in enumerate(text_list):
        tokens,token_indices = get_tokens_from_bert_tokenizer(article)
        article_rep = []
        for token in tokens:
            article_rep.append(token_reps[token])
        
        article_rep = np.mean(article_rep)
        article_reps.append(article_rep)
    
    print("\nFinished Taking Means for article reps")
    
    return np.array(article_reps)

@timer
def load_bert_embeddings(df,saved_path,batch_size,layer,context_var,aggregation="mean+max"):
    """
    Use cases:
    ----------
    1) 0% - No contextuality :
    --------------------------
       * For this say an article has w words (w1,w2,w3,etcc .. til w500)
       * For each token w_i 
           * Find all documents that contain w_i with the corresponding token index of w_i in that document (might need to use bert tokenizer)
           * Average over all the representations of w_i from the list of documents that contain it
       * Now average over all of the w_i and max across all w_i's and concatenate them to get the document representation
    
    2) 100% - Complete contextuality :
    ---------------------------------
       * Normaly load and take the average across tokens to get the corresponding document representation
    
    3) 0% < k% < 100% : k% of contextuality
    -------------------
       * Same as use case one but sample (100-k)% of the documents that contain w_i and average over the different representations
       
    """
    
    if context_var != 100:
        bert_mat = load_bert_output(folder=saved_path,
                                    layer=layer,
                                    aggregation=None)
        
        return load_bert_with_cv(text_list=df["all_text"].tolist(),
                                 bert_reps=bert_mat,
                                 batch_size=batch_size,
                                 layer=layer,
                                 cv=context_var,
                                 aggregation=aggregation)
    
    else:
        return load_bert_output(folder=saved_path,
                                layer=layer,
                                aggregation=aggregation)



if __name__ == "__main__":
    
    # Set GPU ID
    torch.cuda.set_device(0)
    
    # get our sample dataset
    print("\nSampling Dataset and Saving ......")
    
    if os.path.isfile("../sampled_articles_from_relevant_data.csv"):
        sampled_df = pd.read_csv("../sampled_articles_from_relevant_data.csv")
    
    else:
        path =  "../labeled_political_articles.pkl"
        articles_df = load_pkl_file(path)
        sampled_df = sample_data(df=articles_df,sample_size=100000,seed=RANDOM_SEED)
        print("Sampled Size: %s" %str(sampled_df.shape[0]))
        sampled_df["processed_text"] = preprocess_texts(text_lists=sampled_df["text"])
        sampled_df["all_text"] = sampled_df["title"]+ " " + sampled_df["processed_text"]
        sampled_df.to_csv("../sampled_articles_from_relevant_data.csv",index=False)
    
    print("\nInferring Bert Embeddings for Sampled Dataset ......")
    save_and_infer_bert_embeddings(text_list=sampled_df["all_text"].tolist(),batch_size=50,save_path="/media/karthikshivaram/SABER_4TB")