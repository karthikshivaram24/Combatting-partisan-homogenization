from collections import Counter, defaultdict
from Scripts.utils.bert_utils import load_tokenizer
import Scripts.utils.config as CONFIG
from sklearn.model_selection import train_test_split
from Scripts.utils.clustering_utils import run_clustering, get_cluster_sizes, score_cluster, get_cluster_pairs, get_pairwise_dist, cluster2doc, filter_clusters, get_top_100_clusterpairs
from Scripts.utils.general_utils import timer
import numpy as np
import pandas as pd
import random
import itertools


def load_data(file_path):
    """
    """
    tokenizer = load_tokenizer()
    df = pd.read_csv(data_path)
    df.drop(columns=["all_text"],inplace=True)
    df["processed_title"] = preprocess_texts(df["title"].tolist())
    return df
    

def calc_idf(vocab_dict,num_docs = 100000):
    """
    idf_t = log(N/df_t)
    
    Large it is rarer the term, smaller it is more frequent the term
    """
    return {key: np.log2(num_docs/vocab_dict[key]) for key in vocab_dict.keys()}

def filter_vocab(vocab_idf_dict,thresh=10.0,filter_word_pieces=True):
    """
    """
    # remove words from title vocab that have very low idf values and are stopwords
    stop_words = set(stopwords.words("english"))
    vocab_idf = dict(filter(lambda x: x[1]>=thresh, vocab_idf_dict.items()))
    vocab_idf = dict(filter(lambda x: x[0] not in stop_words, vocab_idf.items()))
    vocab_idf = dict(filter(lambda x: "UNK" not in x[0], vocab_idf.items())) # Token absent from bert's vocab
    vocab_idf = dict(filter(lambda x: not x[0].isnumeric(), vocab_idf.items())) # Don't include numbers to represent context words
    vocab_idf = dict(filter(lambda x: len(x[0]) >= 3,vocab_idf.items())) # filter out small words
    
    if filter_word_pieces:
        vocab_idf = dict(filter(lambda x: "##" not in x[0], vocab_idf.items()))
    
    
    
    return vocab_idf

def get_vocab(text_list):
    """
    Here we get the vocab to sample from for obtaining our context word also has idf information
    
    Strategy 1: Using words from the title
    """
    tokens_list = []
    tokenizer = load_tokenizer()
    vocab_df = defaultdict(int)
    for text in text_list:
        token_ids = tokenizer.encode(text,add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        for token in tokens:
            vocab_df[token] +=1
        
    vocab_idf = calc_idf(vocab_df)
    
    print("Vocab_size : %s" %str(len(vocab_idf.keys())))
    
    return vocab_idf

def replace_context_word(token_text,context_word):
    """
    """
    token_text.remove(context_word)
    
    return token_text

def sample_context_words(text_list,idf_thresh=10.0,pos_sample_num=3,neg_sample_num=3,cword_type="pos",filter_word_pieces=True):
    """
    """
    vocab_idf = get_vocab(text_list)
    vocab_idf = filter_vocab(vocab_idf,
                             thresh=idf_thresh,
                             filter_word_pieces=filter_word_pieces)
    
    print("Vocab Size After Filtering : %s" %str(len(vocab_idf.keys())))
    tokenizer = load_tokenizer()
    
    if cword_type == "pos":
        
        context_words = []
        fails = 0
        for text in text_list :
            context_word = sample_context_words_pos(text,vocab_idf,tokenizer,filter_word_pieces=filter_word_pieces,pos_sample_num=pos_sample_num)
            if "DROP_THIS" in Counter(context_word) and Counter(context_word)["DROP_THIS"] == pos_sample_num:
                fails+=1
            context_words.append(context_word)
        
        print("Failed to find context words for : %s docs" %str(fails))
        return context_words
    
    else:
        
        context_words = []
        fails= 0
        for text in text_list :
            # result is list of lists where list[0] has length = neg_sample_num
            con_wl = sample_context_words_neg(text,vocab_idf,tokenizer,neg_sample_num=neg_sample_num)
            if con_wl == None:
                fails+=1
            context_words.append(con_wl)
            
        print("Failed to find context words for : %s docs" %str(fails))
        return context_words

def sample_context_words_pos(text,vocab_idf,tokenizer,pos_sample_num=3,filter_word_pieces=True):
    """
    Sample from titles vocab but don't include stopwords or low-idf terms (frequent terms)
    """
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text,add_special_tokens=False))
    
    if filter_word_pieces:
        tokens = [t for t in tokens if "#" not in t]
    
    sampled_indices = random.sample(range(len(tokens)),k=len(tokens))
    context_word = None
    loop_counter = 0
    temp_c = 0
    context_words = []
    for indice in sampled_indices:
        if temp_c == pos_sample_num:
            break
        context_word = tokens[sampled_indices[indice]]
        if context_word in vocab_idf:
            context_words.append(context_word)
            context_word = None
            temp_c +=1
            
    # Nones to add:
    nones_to_add = pos_sample_num - len(context_words)
    if nones_to_add > 0:
        context_words += ["DROP_THIS"] * nones_to_add
    
    return context_words


def sample_context_words_neg(text,vocab_idf,tokenizer,neg_sample_num=3):
    """
    Similar to Pos context word sampling 
    
    sample from titles vocab but don't include stopwords or low-idf terms (frequent terms) and not in text
    """
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text,add_special_tokens=False))
    
    context_words = None
    hard_stop = 0
    while True:
        context_words = random.sample(list(vocab_idf.keys()),k=neg_sample_num)
        hard_stop += 1
        not_in_vocab = 0
        for c_w in context_words:
            if c_w not in tokens:
                not_in_vocab +=1
        
        if not_in_vocab == neg_sample_num:
            break
        
        if hard_stop == 11:
            context_words = None
            break
    
    return context_words


def get_cluster_pairs_top(df,vectors,sample_size=3):
    """
    """
    # Now drop the rows with Nones for context_word_pos and similarly do the same to the embeddings
    
    df["drop_"] = df['context_word_pos'].apply(lambda x: x == ["DROP_THIS"]*sample_size)
    
    indices_to_drop = df.index[df['drop_'] == True].tolist()
    print("Indices to drop : %s" %str(len(indices_to_drop)))
    clean_df = df.drop(df.index[indices_to_drop])
    clean_df.reset_index(drop=True,inplace=True)
    print(clean_df.head(2))
    print("New Df Shape : %s" %str(clean_df.shape))
    
    vectors = np.delete(vectors,indices_to_drop,axis=0)
    
    clusters,cluster_clf = run_clustering(vectors=vectors,
                                              seed=CONFIG.RANDOM_SEED,
                                              num_clusters=CONFIG.num_clusters,
                                              clus_type="kmeans")
    
    doc_2_cluster_map = cluster2doc(num_texts= clean_df.shape[0],
                                    cluster_labels=cluster_clf.labels_)
    
    
    cluster_sizes = get_cluster_sizes(cluster_clf)
    
    cluster_pairs = get_cluster_pairs(num_clusters=CONFIG.num_clusters)
    
    filtered_cluster_pairs = filter_clusters(cluster_pairs=cluster_pairs,
                                            doc_2_cluster_map=doc_2_cluster_map,
                                            cluster_sizes=cluster_sizes,
                                            partisan_scores=clean_df["binary_ps"].tolist(),
                                            min_size=CONFIG.min_cluster_size,
                                            max_size=CONFIG.max_cluster_size,
                                            min_partisan_size=CONFIG.min_partisan_size)
    
    top100 = None
    
    if len(filtered_cluster_pairs) > 100:
        print("\nNumber of Filtered Cluster Pairs are greater 100, picking top 100 most similar cluster pairs")
        cluster_pair_dist_mat = get_pairwise_dist(cluster_clf,dist_type="cosine")
        top100 = get_top_100_clusterpairs(cluster_pairs=filtered_cluster_pairs,dist_matrix=cluster_pair_dist_mat,reverse=True)
        
    else:
        top100 = filtered_cluster_pairs
        print("\nNumber of Filtered Cluster Pairs is less than 100 so skipping top 100 selection")
    
    return clean_df , doc_2_cluster_map, top100 

def gen_samples(df,neg_sample_size=3):
    """
    columns = processed_text,processed_title, context_word_pos, context_word_neg
    """
    
    df["processed_all"] = df["processed_title"] + " " + df["processed_text"]
    
    text_list = df["processed_all"].tolist()
    ps_labels = df["binary_ps"].tolist()
    pos_con_word = df["context_word_pos"].tolist()
    neg_con_word = df["context_word_neg"].tolist()
    which_cluster = df["which_cluster"].tolist()
    
#     print("Which_cluster size : %s" %str(len(which_cluster)))
    
    text_list_neg = []
    ps_labels_neg = []
    text_list_pos = []
    ps_labels_pos = []
    which_cluster_pos = []
    which_cluster_neg = []
    
    for ind_t, text in enumerate(text_list):
        text_list_neg.append([text]*neg_sample_size)
        ps_labels_neg.append([ps_labels[ind_t]]*neg_sample_size)
        which_cluster_neg.append([which_cluster[ind_t]]*neg_sample_size)
        text_list_pos.append([text]*neg_sample_size)
        ps_labels_pos.append([ps_labels[ind_t]]*neg_sample_size)
        which_cluster_pos.append([which_cluster[ind_t]]*neg_sample_size)
        
    text_list_neg = list(itertools.chain(*text_list_neg))
    neg_con_word = list(itertools.chain(*neg_con_word))
    ps_labels_neg = list(itertools.chain(*ps_labels_neg))
    which_cluster_neg = list(itertools.chain(*which_cluster_neg))
    
    text_list_pos = list(itertools.chain(*text_list_pos))
    pos_con_word = list(itertools.chain(*pos_con_word))
    ps_labels_pos = list(itertools.chain(*ps_labels_pos))
    which_cluster_pos = list(itertools.chain(*which_cluster_pos))
    
    assert len(text_list_neg) == len(neg_con_word)
    assert len(text_list_neg) == len(text_list_pos)
    assert len(text_list_pos) == len(pos_con_word)
    
    all_text_list = text_list_pos + text_list_neg
    all_con_word = pos_con_word + neg_con_word
    all_word_labels = ([1]*len(pos_con_word)) + ([0] * len(neg_con_word))
    all_ps_labels = ps_labels_pos + ps_labels_neg
    all_which_clusters = which_cluster_pos + which_cluster_neg
    
    df_sample = pd.DataFrame()
    df_sample["text"] = all_text_list
    df_sample["context_word"] = all_con_word
    df_sample["word_label"] = all_word_labels
    df_sample["class_label"] = all_ps_labels
    df_sample["which_cluster"] = all_which_clusters
    
    df_sample = df_sample.loc[df_sample["context_word"] != "DROP_THIS"].reset_index(drop=True)
    
    # Shuffle twice
    df_sample = df_sample.sample(frac=1.0,random_state=CONFIG.RANDOM_SEED)
    df_sample = df_sample.sample(frac=1.0,random_state=CONFIG.RANDOM_SEED+1)
    
    return df_sample
    

def get_label_dist(labels):
    """
    """
    print("1 : %s" %str(sum(labels)))
    print("0 : %s" %str(len(labels) - sum(labels)))

def get_train_test_attm(df,cp,doc_2_cluster_map,neg_sample_size=3,single_task=True):
    """
    train and test we need to first subsample using doc_2_cluster_map for each cluster
    get train and test , then create negative samples and finally shuffle both train and test
    """
    cluster1_indices = doc_2_cluster_map[cp[0]]
    cluster2_indices = doc_2_cluster_map[cp[1]]
    
    train_df = df.iloc[cluster1_indices].reset_index(drop=True)
    test_df = df.iloc[cluster2_indices].reset_index(drop=True)
    # Label Flip
    test_df["binary_ps"] = test_df["binary_ps"].apply(lambda x: np.abs(x+(-1)))
                                                              
    print("Original Train Shape : %s" %str(train_df.shape))
    print("Original Test Shape : %s" %str(test_df.shape))
    
    if not single_task:
        train_df = gen_samples(train_df,neg_sample_size=neg_sample_size)
        test_df = gen_samples(test_df,neg_sample_size=neg_sample_size)

        train_df = train_df.sample(frac=1,random_state=CONFIG.RANDOM_SEED).reset_index(drop=True)
        test_df = test_df.sample(frac=1,random_state=CONFIG.RANDOM_SEED).reset_index(drop=True)

        print("Exploded Train Shape : %s" %str(train_df.shape))
        print("Exploded Test Shape : %s" %str(test_df.shape))
        
    return train_df,test_df

def get_train_test_ssda(df,cp,doc_2_cluster_map,neg_sample_size=3,single_task=True):
    """
    * train = 90% c1 data + 10% c2 data
    * test = 50% c1 data + 50% c2 data
    
    * From cluster 1 choose 70% of the data for train, 30% for test
    * From cluster 2 choose 10% of 70% from c1 for train, test of cluster 1 test size
    """
    strata_columns = []
    if single_task:
        strata_columns = ["binary_ps"]
    
    if not single_task:
        strata_columns = ["class_label","word_label"]
    c1_df, c2_df = get_train_test_attm(df,cp,doc_2_cluster_map,neg_sample_size=neg_sample_size,single_task=single_task)
    c1_df["which_cluster"] = [1]*c1_df.shape[0]
    c2_df["which_cluster"] = [2]*c2_df.shape[0]
    
    c1_train, c1_test = train_test_split(c1_df,test_size=0.20, stratify=c1_df[strata_columns], random_state=CONFIG.RANDOM_SEED)
    
    c1_train, c1_val = train_test_split(c1_train,test_size=0.10,stratify=c1_train[strata_columns], random_state=CONFIG.RANDOM_SEED)
    
    c1_train_num = c1_train.shape[0]
    
    c2_df_train,c2_df_test = train_test_split(c2_df,train_size=int(0.1*c1_train_num),stratify=c2_df[strata_columns], random_state=CONFIG.RANDOM_SEED)
    
    print("\nSample size from C1 in Train : %s" %str(c1_train.shape))
    if single_task:
        get_label_dist(c1_train["binary_ps"].tolist())
    else:
        get_label_dist(c1_train["class_label"].tolist())
    print("\nSample size from C2 in Train : %s" %str(c2_df_train.shape))
    if single_task:
        get_label_dist(c2_df_train["binary_ps"].tolist())
    else:
        get_label_dist(c2_df_train["class_label"].tolist())
    
    train = pd.concat([c1_train,c2_df_train],axis=0)
    print("\nTrain Size : %s"%str(train.shape))
    if single_task:
        get_label_dist(train["binary_ps"].tolist())
    else:
        get_label_dist(train["class_label"].tolist())
    
    c2_test_,c2_val_temp = train_test_split(c2_df_test,train_size=c1_test.shape[0],stratify=c2_df_test[strata_columns], random_state=CONFIG.RANDOM_SEED)
    
    print("\nSample Size from C1 in Test : %s" %str(c1_test.shape))
    if single_task:
        get_label_dist(c1_test["binary_ps"].tolist())
    else:
        get_label_dist(c1_test["class_label"].tolist())
    print("\nSample Size from C2 in Test : %s" %str(c2_test_.shape))
    if single_task:
        get_label_dist(c2_test_["binary_ps"].tolist())
    else:
        get_label_dist(c2_test_["class_label"].tolist())
    
    test = pd.concat([c1_test,c2_test_],axis=0)
    print("\nTest Size : %s" %str(test.shape))
    if single_task:
        get_label_dist(test["binary_ps"].tolist())
    else:
        get_label_dist(test["class_label"].tolist())
        
    c2_val , _ = train_test_split(c2_val_temp,train_size=c1_val.shape[0],stratify=c2_val_temp[strata_columns], random_state=CONFIG.RANDOM_SEED)
    
    train = train.sample(frac=1.0,random_state=int(CONFIG.RANDOM_SEED * 0.01))
    test = test.sample(frac=1.0,random_state=int(CONFIG.RANDOM_SEED * 0.1))
    
    val = pd.concat([c1_val,c2_val],axis=0)
    print("\nSample size from C1 in Val : %s" %str(c1_val.shape))
    if single_task:
        get_label_dist(c1_val["binary_ps"].tolist())
    else:
        get_label_dist(c1_val["class_label"].tolist())
    print("\nSample size from C2 in Val : %s" %str(c2_val.shape))
    if single_task:
        get_label_dist(c2_val["binary_ps"].tolist())
    else:
        get_label_dist(c2_val["class_label"].tolist())
    print("\nVal Size : %s" %str(val.shape))
    if single_task:
        get_label_dist(val["binary_ps"].tolist())
    else:
        get_label_dist(val["class_label"].tolist())
    val = val.sample(frac=1.0,random_state=int(CONFIG.RANDOM_SEED * 0.01))
    
    return train,test,val

def get_train_test_ssda_updated(df,cp,doc_2_cluster_map,neg_sample_size=3,single_task=True):
    """
    Use context word enlargement after the splitting has been done
    """
    strata_columns = ["binary_ps"]
    # get our cluster dfs
    c1_df, c2_df = get_train_test_attm(df,cp,doc_2_cluster_map,neg_sample_size=neg_sample_size,single_task=True)
    c1_df["which_cluster"] = [1]*c1_df.shape[0]
    c2_df["which_cluster"] = [2]*c2_df.shape[0]
    
    # Split into Train, Val, Test 
    
    # Train
    c1_train, c1_test = train_test_split(c1_df,test_size=0.20, stratify=c1_df[strata_columns], random_state=CONFIG.RANDOM_SEED)
    c1_train, c1_val = train_test_split(c1_train,test_size=0.10,stratify=c1_train[strata_columns], random_state=CONFIG.RANDOM_SEED)
    c2_df_train,c2_df_test = train_test_split(c2_df,train_size=int(0.1*c1_train.shape[0]),stratify=c2_df[strata_columns], random_state=CONFIG.RANDOM_SEED)
    train = pd.concat([c1_train,c2_df_train],axis=0)
    
    # Test
    c2_test_,c2_val_temp = train_test_split(c2_df_test,train_size=c1_test.shape[0],stratify=c2_df_test[strata_columns], random_state=CONFIG.RANDOM_SEED)
    test = pd.concat([c1_test,c2_test_],axis=0)
    
    # Val
    c2_val , _ = train_test_split(c2_val_temp,train_size=c1_val.shape[0],stratify=c2_val_temp[strata_columns], random_state=CONFIG.RANDOM_SEED)
    val = pd.concat([c1_val,c2_val],axis=0)
    
    print("Data Sizes before addition of context words and context labels :")
    print("Train : %s" %str(train.shape))
    print("Val : %s" %str(val.shape))
    print("Test : %s" %str(test.shape))
    # if multitask expand datasets to include context words and corresponding context labels
    
    if not single_task:
        # expand train and val only
        train = gen_samples(train,neg_sample_size=neg_sample_size)
        val = gen_samples(val,neg_sample_size=neg_sample_size)
        
        # Test set should reflect the same columns for the dataloader but these values should not be used during inference
        test["text"] = test["processed_title"] + " " + test["processed_text"]
        test["context_word"] = "PLACEHOLDER"
        test["word_label"] = 0 #Dummy Label
        test["class_label"] = test["binary_ps"]
    
    # Shuffle all three datasets
    train = train.sample(frac=1.0,random_state=int(CONFIG.RANDOM_SEED * 0.01))
    test = test.sample(frac=1.0,random_state=int(CONFIG.RANDOM_SEED * 0.1))
    val = val.sample(frac=1.0,random_state=int(CONFIG.RANDOM_SEED * 0.01))
    
    print("Data Sizes after addition of context words and context labels :")
    print("Train : %s" %str(train.shape))
    print("Val : %s" %str(val.shape))
    print("Test : %s" %str(test.shape))
    
    print("Train Columns :\n %s"%str(train.columns))
    print("Test Columns :\n %s"%str(test.columns))
    print("Val Columns :\n %s"%str(val.columns))
    
    return train,test,val