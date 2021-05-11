"""
"""

import torch
from torch import nn
from transformers import BertModel, BertTokenizer

"""

"""

class AttentionSTUpdated(nn.Module):
    
    def __init__(self,embedding_size=768,verbose=True,which_forward=2,with_attention=True,dropout=0.3):
        super(AttentionSTUpdated,self).__init__()
        
        self.verbose = verbose
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.with_attention = with_attention
        
        for p in self.bert.parameters():
            p.requires_grad = False
        
        
        self.attention = nn.Linear(in_features=embedding_size,
                                       out_features=1,
                                       bias=False)
        
        self.attention2 = nn.Linear(in_features=embedding_size,
                                       out_features=1,
                                       bias=False)
        
        self.attention_combine =  nn.Linear(in_features=embedding_size*2,out_features=embedding_size,bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.recom_pred = nn.Linear(in_features=embedding_size*2,
                                    out_features=1,
                                    bias=True)
        
        # Tanh Activation
        self.tanh = nn.Tanh()
        
        # Softmax Activation
        self.softmax = nn.Softmax(dim=-1)
        
        # Sigmoid Activation
        self.sigmoid = nn.Sigmoid()
        
        self.which_forward = which_forward
    
    def forward(self,bert_tokenized_words,attention_masks):
        """
        """
        
        bert_output = self.bert(input_ids=bert_tokenized_words,attention_mask=attention_masks)
        bert_hidden_states  = bert_output[2]
        bert_layer12_hidden_states = bert_hidden_states[-1]
        
        
        if self.with_attention:
        # iterate over articles in a given batch
            attention_vector_batch = []
            attention_vector_batch_2 = []

            for sent in range(bert_layer12_hidden_states.size(0)):

                sent_embed_matrix = bert_layer12_hidden_states[sent,:,:] # 500 x 768

                word_weights = self.tanh(self.dropout(self.attention(sent_embed_matrix)).squeeze())

                self.normalized_word_weights = self.softmax(word_weights) # input is 500 x 1, output is 500 x 1

                attention_vector = sent_embed_matrix.T .mul(self.normalized_word_weights).sum(dim=1)

                attention_vector_batch.append(attention_vector.unsqueeze(-1).T) 
                
                
                
                word_weights_2 = self.tanh(self.dropout(self.attention2(sent_embed_matrix)).squeeze())

                self.normalized_word_weights_2 = self.softmax(word_weights_2) # input is 500 x 1, output is 500 x 1

                attention_vector_2 = sent_embed_matrix.T .mul(self.normalized_word_weights_2).sum(dim=1)

                attention_vector_batch_2.append(attention_vector_2.unsqueeze(-1).T) 
                
                

            attention_vector_batch = torch.cat(attention_vector_batch,dim=0) # we get (batch_size,768)
            attention_vector_batch_2 = torch.cat(attention_vector_batch_2,dim=0)
            
            total_attention = torch.cat([attention_vector_batch,attention_vector_batch_2],dim=1)

            y_preds = self.sigmoid(self.recom_pred(total_attention)).squeeze()
        
            return y_preds, attention_vector_batch, None
        
        else:
            
#             bert_layer12_hidden_states = torch.cat([bert_layer12_hidden_states[article,0,:].unsqueeze(0) for article in range(bert_layer12_hidden_states.size(0))],dim=0)
            bert_layer12_hidden_states = bert_layer12_hidden_states[:,0,:]
            print(bert_layer12_hidden_states.size())
            y_pred = self.sigmoid(self.recom_pred(bert_layer12_hidden_states)).squeeze()
            
            return y_pred, None,None
        
