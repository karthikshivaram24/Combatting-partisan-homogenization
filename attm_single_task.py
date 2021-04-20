"""
"""

import torch
from torch import nn
from transformers import BertModel, BertTokenizer

"""

"""

class AttentionST(nn.Module):
    
    def __init__(self,embedding_size=768,verbose=True,which_forward=2,with_attention=True):
        super(AttentionST,self).__init__()
        
        self.verbose = verbose
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.with_attention = with_attention
        
        for p in self.bert.parameters():
            p.requires_grad = False
        
        if self.with_attention:
            self.attention = nn.Linear(in_features=embedding_size,
                                       out_features=1,
                                       bias=False)
        
        self.intermed = nn.Linear(in_features=embedding_size, 
                                  out_features=embedding_size,
                                  bias=True)
        
        self.recom_pred = nn.Linear(in_features=embedding_size,
                                    out_features=1,
                                    bias=True)
        
        # Softmax Activation
        self.softmax = nn.Softmax(dim=-1)
        
        # Sigmoid Activation
        self.sigmoid = nn.Sigmoid()
        
        self.which_forward = which_forward
    
    def forward(self,bert_tokenized_words):
        """
        """
        if self.which_forward == 1:
            return self.forward_single(bert_tokenized_words)
        if self.which_forward == 2:
            return self.forward_batch(bert_tokenized_words)
        
    def forward_single(self,
                bert_tokenized_words):
        """
        Inputs:
        ------
        
        Outputs:
        --------
        """
        
        # ************ Recommendation Network ********************
        
        # output shape should be (max_length,dim)
        bert_output = self.bert(input_ids=bert_tokenized_words)
        bert_hidden_states  = bert_output[2]
        bert_layer12_hidden_states = bert_hidden_states[-1][0,:,:]
        
        if self.with_attention:
            attention_un = torch.cat([self.attention(embedding) for embedding in bert_layer12_hidden_states])  # shape (N,max_length,1) *** u_it **
            attentions = self.softmax(attention_un) # shape (N,max_length,1) ***  a_it ***
            attention_cvector = bert_layer12_hidden_states.T.mul(attentions).sum(dim=1) # shape (N,768,1) *** s_i ***

            y_pred = self.sigmoid(self.recom_pred(attention_cvector.T))
            
            return y_pred, attention_cvector
        
        else:
            # Use cls token for prediction
            y_pred = self.sigmoid(self.recom_pred(bert_layer12_hidden_states[0,:]))
            
            return y_pred,None

        
#         if self.verbose:
#             print("\nShape Details :")
#             print("1. Bert Embeddings Shape : %s" %str(bert_layer12_hidden_states.size()))
#             print("2. attention_un Shape : %s" %str(attention_un.size()))
#             print("3. attention_norm Shape : %s" %str(attentions.size()))
#             print("4. attention_cvector shape : %s" %str(attention_cvector.size()))
#             print("5. y_pred shape : %s" %str(y_pred.size()))
#             print(str(y_pred.item()))
        
#         return y_pred, attention_cvector
    
    def forward_batch(self,
                      bert_tokenized_words_batch):
        """
        Shape Details :
        
        when considering single samples
        1. Bert Embeddings Shape : torch.Size([500, 768])
        2. attention_un Shape : torch.Size([500])
        3. attention_norm Shape : torch.Size([500])
        4. attention_cvector shape : torch.Size([768])
        5. y_pred shape : torch.Size([1])
        0.5326072573661804
        6. context_word_embed_bert shape : torch.Size([1, 768])
        7. context_pred shape : torch.Size([1])
        
        when considering batches
        1. Bert Embeddings Shape : (N,500,768)-> Done
        2. attention_un Shape : (N,500)
        3. attention_norm Shape : (N,500)
        4. attention_cvector Shape : (N,768)
        5. ypred Shape : (N,1)
        6. context_word_embed_bert shape : torch.Size([N, 768])
        7. context_pred shape : torch.Size([N,1])
        
        """
        
        # ************* REC NETWORK *********************
        
        bert_output = self.bert(input_ids=bert_tokenized_words_batch)
        bert_hidden_states  = bert_output[2]
        bert_layer12_hidden_states = bert_hidden_states[-1]
        
        # Unormalized Word Attentions
        if self.with_attention:
            atten_un_batch = []
            for item_in_batch in range(bert_layer12_hidden_states.size(0)):
                atten_un_sent = []
                for word_in_sent in range(bert_layer12_hidden_states.size(1)):
                    word_atten_un = self.attention(bert_layer12_hidden_states[item_in_batch,word_in_sent,:]).unsqueeze(0) # should be of size [1,1]
                    atten_un_sent.append(word_atten_un)  

                atten_un_sent = torch.cat(atten_un_sent,dim=0).T # should have shape (1,500)
                atten_un_batch.append(atten_un_sent)

            atten_un_batch = torch.cat(atten_un_batch,dim=0) # should have shape (N,500)


            # Normalized Word Attentions
            attentions_norm_batch = self.softmax(atten_un_batch).squeeze(-1) # should have shape (N,500) or (N,500,1)

            # Sentence Context vector
            attention_cvector_batch = []

            for item_in_batch in range(bert_layer12_hidden_states.size(0)):
                attention_cvector = bert_layer12_hidden_states[item_in_batch,:,:].T.mul(attentions_norm_batch[item_in_batch,:]).sum(dim=1) # shape (1,768)
                attention_cvector_batch.append(attention_cvector.unsqueeze(0)) 


            attention_cvector_batch = torch.cat(attention_cvector_batch,dim=0) # shape (N,768) *** s_i ***

            # Output Layer 1
            y_pred = self.sigmoid(self.recom_pred(attention_cvector_batch)).squeeze() # shape (N,1)
            return y_pred, attention_cvector_batch
        
        else:
            bert_layer12_hidden_states = torch.cat([bert_layer12_hidden_states[article,0,:].unsqueeze(0) for article in range(bert_layer12_hidden_states.size(0))],dim=0)
            y_pred = self.sigmoid(self.recom_pred(bert_layer12_hidden_states)).squeeze()
            return y_pred,None