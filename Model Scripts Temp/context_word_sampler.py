
import tqdm
import numpy as np
import random
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from bert_utils import load_tokenizer,batch_text_gen


def vectorize_text(text_list):
    """
    """
    tfidf_vectorizer = TfidfVectorizer(min_df=30, binary=False, max_df=0.90, stop_words='english',max_features=None)
    tf_idf_vectors = tfidf_vectorizer.fit_transform(text_list)
    print("Vocab Size : %s "%str(len(tfidf_vectorizer.get_feature_names())))
    return tf_idf_vectors, tfidf_vectorizer

def get_tokens(text_list,batch_size=100,max_length=15):
    tokenizer = load_tokenizer()
    all_toks = []
    for batch_num,batched_text in enumerate(tqdm(batch_text_gen(text_list,batch_size=batch_size),total=int(len(text_list)/batch_size)+1)):
        encoded_batch = tokenizer.batch_encode_plus(batched_text, add_special_tokens=True, max_length = max_length, padding='max_length', return_attention_mask = False,truncation=True, return_tensors = 'pt')
        token_tensors = encoded_batch["input_ids"] # 2d , (batch_size,max_length)
        batched_toks = []
        for sent in range(token_tensors.size()[0]):
            sent_toks = []
            tokens = tokenizer.convert_ids_to_tokens(token_tensors[sent])
            for t in tokens:
                if t not in ['[SEP]', '[PAD]']:
                    sent_toks.append(t)
            batched_toks.append(sent_toks)
        
        all_toks += batched_toks
    
    return all_toks

def sample_pos_context_word(tokens,tfidf_vec,tf_idf_vectorizer,bad_terms,search_range=50,sample_size=1):
    """
    """
    sampled_words = []
    tokenslc = [t.lower() for t in tokens]
    # high tfidf terms in text argsort descending
    sorted_indices = np.argsort(tfidf_vec.todense())[::-1].tolist()[0]
    sorted_tokens_tfidf = np.array(tf_idf_vectorizer.get_feature_names())[sorted_indices]
    bad_term_dict = defaultdict(int)
    
    for b in bad_terms:
        bad_term_dict[b] = 1
    # bert tokens of text
    for tftoken in sorted_tokens_tfidf.tolist():
        if tftoken in tokenslc and bad_term_dict[tftoken] != 1:
            sampled_words.append(tftoken)

        if len(sampled_words) == sample_size:
            break
    return sampled_words

def sample_neg_context_word(tokens,pos_words_set,sample_size=2):
    """
    """
    tokenslc = [t.lower() for t in tokens]
    sampled_words = []
    hard_stop = 0
    while True:
        sw = random.choice(pos_words_set)
        hard_stop +=1
        if sw not in tokenslc:
            sampled_words.append(sw)
        
        if len(sampled_words) == sample_size:
            break
        
        if hard_stop == 20:
            break
        
    return sampled_words
            

def sample_context_words(df,tfidf_vecs,all_tokens,tf_idf_vectorizer,bad_terms,sample_size=3,search_range=20):
    """
    """
    sampled_pos = []
    all_pos = []
    sampled_neg = []
    for i in tqdm(range(df.shape[0]),total=df.shape[0]):
        
        sample_pos = sample_pos_context_word(tokens=all_tokens[i],
                                             tfidf_vec=tfidf_vecs[i],
                                             tf_idf_vectorizer=tf_idf_vectorizer,
                                             bad_terms=bad_terms,
                                             search_range=search_range,
                                             sample_size=sample_size)
    
        sampled_pos.append(sample_pos)
        all_pos = all_pos + sample_pos
    
    sampled_pos_unique = list(set(all_pos))
    
    print("Pos tokens size : %s"%str(len(all_pos)))
    print("Pos vocab size : %s"%str(len(sampled_pos_unique)))
    
    for i in tqdm(range(df.shape[0]),total=df.shape[0]):
        
        sample_neg = sample_neg_context_word(tokens=all_tokens[i],
                                             pos_words_set=sampled_pos_unique,
                                             sample_size=sample_size)
        
        sampled_neg.append(sample_neg)
    
    unsampled_pos = 0
    unsampled_neg = 0
    
    for i in tqdm(range(df.shape[0]),total=df.shape[0]):
        
        if len(sampled_pos[i]) < sample_size:
            unsampled_pos +=1
        
        if len(sampled_neg[i]) < sample_size:
            unsampled_neg +=1
    
    print("Articles missing a pos context word : %s"%str(unsampled_pos))
    print("Articles missing a neg context word : %s"%str(unsampled_neg))
    
    return sampled_pos, sampled_neg
    
def check_bad_termsincontext(bad_terms,context_words):
    """
    """
    c_1_n = []
    
    for word in context_words:
        if len(word)>=1:
            c_1_n.append(word[0])
    
    
    context_words_as_bad_term = 0
    bad_terms_ = [b for b in bad_terms]
    for c in c_1_n:
        for b in bad_terms_:
            if c == b:
                context_words_as_bad_term += 1

    return context_words_as_bad_term/len(c_1_n)

def sample_context_words_fin(df,bad_terms):
    """
    """
    text_list = df["all_text"].tolist()
    tfidf_vecs,tf_idf_vectorizer =  vectorize_text(text_list)
    all_tokens = get_tokens(text_list,batch_size=100,max_length=15)
    return sample_context_words(df,tfidf_vecs,all_tokens,tf_idf_vectorizer,bad_terms,sample_size=1,search_range=50)
    