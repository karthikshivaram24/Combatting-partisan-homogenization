import torch
import numpy as np
from collections import Counter, defaultdict
from torch_datasets import CPDatasetST, CPDatasetMT
from torch.utils.data import DataLoader
from bert_utils import load_model, load_tokenizer, get_bert_embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


# def get_test_vocab(test):
#     """
#     """
#     tokenizer = load_tokenizer()
#     tokens_vocab = Counter()
#     for text in test:
#         encoded_batch = tokenizer.batch_encode_plus(batched_text, add_special_tokens=True, max_length = 350, padding='max_length', return_attention_mask = False,truncation=True, return_tensors = 'pt')
#         token_tensors = encoded_batch["input_ids"]
#         tokens = tokenizer.covert_ids_to_tokens(token_tensors)
#         tokens_vocab.update(tokens)
#     return tokens_vocab.keys()

def get_count_vec_vocab(test):
    """
    """
    tfidf_vectorizer = TfidfVectorizer(min_df=15, binary=True, max_df=0.90, stop_words='english',max_features=None)
    tfidf_vectorizer.fit(test)
    
    vocab = tfidf_vectorizer.get_feature_names()
    return vocab
    
        
def get_attention_weights(model,test,single=True):
    """
    """
    # RUn throught the test set and get corresponding attentions scores for words in our test vocabulary
    # aggregate and return
    tokenizer = load_tokenizer()
    
    bmodel = load_model()
    bmodel.to(torch.device('cuda:1'))
    bmodel.eval()
    
    cw_embed_val = get_bert_embeddings(test,bmodel,tokenizer)
    
    vocab = get_count_vec_vocab(test["all_text"])
    
    if single:
        test = CPDatasetST(test)
    
    if not single:
        test = CPDatasetMT(test,cw_embed_val)
    
    test_dataloader = DataLoader(test,batch_size=1,num_workers=1, shuffle=True)
    tokenizer = load_tokenizer()
    
    attention_weights = defaultdict(list)
    
    model.eval()
    
    with torch.no_grad():
        for batch_num, data in enumerate(test_dataloader):

            if single:
                x1 = data[0]
                y1 = data[1]
                t1 = data[2]
                wc = data[3]


            if not single:
                x1 = data[0]
                x2 = data[1]
                y1 = data[2]
                t1 = data[3]
                wc = data[4]
            
            tokens = tokenizer.encode(t1[0], add_special_tokens=True, max_length = 350, padding='max_length', return_attention_mask = False,truncation=True, return_tensors = 'pt')
            token_ids = tokenizer.convert_ids_to_tokens(tokens.flatten())

            x1 = x1.to(torch.device('cuda:1'))

            if single:
                y_pred = model(x1)

            if not single:
                y_pred, context_pred,_= model(x1,None)

            att_w = model.normalized_word_weights.cpu().numpy()

            for i in range(350):
                
                if token_ids[i] in vocab:

                    attention_weights[token_ids[i]].append(att_w[i])
    
    return attention_weights

def aggregate_attW(attw):
    """
    """
    agg_att_w = {}
    total_sum = []
    for token in attw.keys():
        agg_att_w[token] = np.sum(attw[token],axis=0)
    
    total_sum = sum(agg_att_w.values())
    for token in agg_att_w.keys():
        agg_att_w[token] /= total_sum
    
    return agg_att_w

def rank_weights(attw):
    """
    """
    sortedattw = sorted(attw.items(),key=lambda x:x[1],reverse=True)
    token_ranks = {}
    for rank, token_tuple in enumerate(sortedattw):
        token_ranks[token_tuple[0]] = rank
    
    return token_ranks
    

def calc_change_in_rank(testwa1,testwa2):
    """
    iterate through words in vocab and check difference in ranks
    """
    agg_aw1 = aggregate_attW(testwa1)
    agg_aw2 = aggregate_attW(testwa2)
    
    aw1_ranks = rank_weights(agg_aw1)
    aw2_ranks = rank_weights(agg_aw2)
    
    assert len(aw1_ranks.keys()) == len(aw2_ranks.keys())
    
    change_in_rank = {}
    for token in aw1_ranks.keys():
        change_in_rank[token] = aw1_ranks[token] - aw2_ranks[token]
    
    return change_in_rank
