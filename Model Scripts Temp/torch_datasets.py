"""
Contains the torch dataset classes for our Single Task and Multitask Network Models
"""



import h5py
import torch
from attm_utils import load_pickle


class CPDatasetST(torch.utils.data.Dataset):
    """
    Torch dataset class that loads in a pre-calculated bert feature vectors for our train and test or glove feature vectors.
    It iterates over samples of the data that consists of the Feature vector, class label, actual text and the cluster the sample belongs to for the Single Task Models.
    """
    def __init__(self, df,hdf_file_path="/media/karthikshivaram/SABER_4TB/attm_bert_embeddings/token_bert_12_embeds_attm.hdf5",glove_vocab_path="",glove=False,max_length = 350):
        self.df = df
        self.glove = glove
        if not self.glove :
            self.hdf_path = hdf_file_path
            self.h5file = h5py.File(self.hdf_path, "r")
            self.d_set = self.h5file.get("bert_embeds")
        
        if self.glove:
            self.glove_path = glove_vocab_path
            self.glove_vocab = load_pickle(self.glove_path)
            self.glove_w2idx = {k: v for v, k in enumerate(self.glove_vocab)}
            self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        t1 = self.df["all_text"].iloc[index]
        y1 = self.df["binary_ps"].iloc[index]
        wc = self.df["which_cluster"].iloc[index]
        
        
        y1 = torch.Tensor([y1])
        wc = torch.Tensor([wc])
        
        x1 = None
        
        if not self.glove:
            x1 = self.d_set[self.df["doc_indices"].iloc[index],:,:]
            x1 = torch.Tensor(x1)
            
        if self.glove:
            x1_indices = [self.glove_w2idx[token] for token in word_tokenize(t1)]
            
            if len(x1_indices) > self.max_length:
                x1_indices = x1_indices[:self.max_length]
            
            if len(x1_indices) < self.max_length:
                to_pad = self.max_length - len(x1_indices)
                x1_indices = x1_indices + [self.glove_w2idx["PAD"]]* to_pad
                
            x1 = torch.LongTensor(x1_indices)
        

        return x1, y1, t1, wc


class CPDatasetMT(torch.utils.data.Dataset):
    """
    Torch dataset class that loads in a pre-calculated bert feature vectors for our train and test or glove feature vectors.
    It iterates over samples of the data that consists of the Feature vector, Context word Feature vector, class label, actual text and the cluster the sample belongs to for the Multi Task Models.
    """
    def __init__(self, df,embeds_list,hdf_file_path="/media/karthikshivaram/SABER_4TB/attm_bert_embeddings/token_bert_12_embeds_attm.hdf5",context_word_sample_size=2,glove_vocab_path="",glove=False,max_length = 350):
        self.df = df
        self.glove = glove
        
        if not self.glove :
            self.hdf_path = hdf_file_path
            self.h5file = h5py.File(self.hdf_path, "r")
            self.d_set = self.h5file.get("bert_embeds")
            self.embeds_list = embeds_list
        
        if self.glove:
            self.glove_path = glove_vocab_path
            self.glove_vocab = load_pickle(self.glove_path)
            self.glove_w2idx = {k: v for v, k in enumerate(self.glove_vocab)}
            self.max_length = max_length


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        t1 = self.df["all_text"].iloc[index]
        y1 = self.df["binary_ps"].iloc[index]
        wc = self.df["which_cluster"].iloc[index]
        
        
        y1 = torch.Tensor([y1])
        wc = torch.Tensor([wc])
        
        x1 = None
        x2 = None
        
        if not self.glove:
            x1 = self.d_set[self.df["doc_indices"].iloc[index],:,:]
            # create a dictionary of tensors here, with labels
            #{pos_embeds, pos_labels, neg_embeds, neg_labels}
            x2 = torch.cat([self.embeds_list[index]["pos"],self.embeds_list[index]["neg"]],dim=0)
            x1 = torch.Tensor(x1)
        
        if self.glove:
            context_pos_words = self.df["context_pos_words"].iloc[index]
            context_neg_words = self.df["context_neg_words"].iloc[index]
            
            pos_glove_idx = torch.Tensor([self.glove_w2idx[context_pos_words[0]]])
            neg_glove_idx = torch.Tensor([self.glove_w2idx[context_neg_words[0]]])
            
            x2 = torch.cat([pos_glove_idx,neg_glove_idx],dim=0)
            x2 = x2.type(torch.LongTensor)
            
            x1_indices = [self.glove_w2idx[token] for token in word_tokenize(t1)]
            
            if len(x1_indices) > self.max_length:
                x1_indices = x1_indices[:self.max_length]
            
            if len(x1_indices) < self.max_length:
                to_pad = self.max_length - len(x1_indices)
                x1_indices = x1_indices + [self.glove_w2idx["PAD"]]* to_pad
                
            x1 = torch.LongTensor(x1_indices)
            
        return x1, x2, y1, t1, wc