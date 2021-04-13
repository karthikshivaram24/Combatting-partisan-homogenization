import torch
from torch import nn
from transformers import BertModel, BertTokenizer

class AttentionMT(nn.Module):
    
    def __init__(self,embedding_size=768,verbose=True,which_forward=2):
        super(AttentionMT,self).__init__()
        
        self.verbose = verbose
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        
        for p in self.bert.parameters():
            p.requires_grad = False
        
        self.attention = nn.Linear(in_features=embedding_size,
                                   out_features=1,
                                   bias=False)
        
        # Recommendation Network
        # ------------------------
        
        self.recom_pred = nn.Linear(in_features=embedding_size,
                                    out_features=1,
                                    bias=True)
        # Tanh Activation
        self.tanh = nn.Tanh()
        
        # Softmax Activation
        self.softmax = nn.Softmax(dim=-1)
        
        # Sigmoid Activation
        self.sigmoid = nn.Sigmoid()
        
        self.which_forward = which_forward
    
    def forward(self,bert_tokenized_words,bert_tokenized_word_to_predict):
        """
        """
        if self.which_forward == 1:
            return self.forward_single(bert_tokenized_words,bert_tokenized_word_to_predict)
        if self.which_forward == 2:
            return self.forward_batch(bert_tokenized_words,bert_tokenized_word_to_predict)
        if self.which_forward == 3:
            return self.forward_batch_optim(bert_tokenized_words,bert_tokenized_word_to_predict)
        
    def forward_single(self,
                bert_tokenized_words,
                bert_tokenized_word_to_predict):
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

        attention_un = torch.cat([self.attention(embedding) for embedding in bert_layer12_hidden_states])  # shape (N,max_length,1) *** u_it **
        attentions = self.softmax(attention_un) # shape (N,max_length,1) ***  a_it ***
        attention_cvector = bert_layer12_hidden_states.T.mul(attentions).sum(dim=1) # shape (N,768,1) *** s_i ***
        y_pred = self.sigmoid(self.recom_pred(attention_cvector.T))
        
        # ************** Word Prediction Network ********************
        context_word_embed_bert = self.bert(input_ids=bert_tokenized_word_to_predict)
        context_word_embed_bert_hs = context_word_embed_bert[2]
        context_word_embed_bert_layer_12 = context_word_embed_bert_hs[-1][0,:,:]
        
        context_pred = self.sigmoid(torch.mul(attention_cvector,context_word_embed_bert_layer_12).sum(dim=1))
        
        if self.verbose:
            print("\nShape Details :")
            print("1. Bert Embeddings Shape : %s" %str(bert_layer12_hidden_states.size()))
            print("2. attention_un Shape : %s" %str(attention_un.size()))
            print("3. attention_norm Shape : %s" %str(attentions.size()))
            print("4. attention_cvector shape : %s" %str(attention_cvector.size()))
            print("5. y_pred shape : %s" %str(y_pred.size()))
            print(str(y_pred.item()))
            print("6. context_word_embed_bert shape : %s" %str(context_word_embed_bert_layer_12.size()))
            print("7. context_pred shape : %s" %str(context_pred.size()))
            print(str(context_pred.item()))
        
        return y_pred, context_pred, attention_cvector
    

    def forward_batch(self,
                      bert_tokenized_words_batch,
                      bert_tokenized_word_to_predict_batch):
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
        y_pred = self.sigmoid(self.recom_pred(attention_cvector_batch)) # shape (N,1)
        # **************** WORD PRED NETWORK ****************   
        context_word_embed_bert = self.bert(input_ids=bert_tokenized_word_to_predict_batch)
        context_word_embed_bert_hs = context_word_embed_bert[2]
        context_word_embed_bert_layer_12 = context_word_embed_bert_hs[-1] # shape (N,1,768)
        
        attention_wcprod_batch = []
        for item_in_batch in range(context_word_embed_bert_layer_12.size(0)):
            
            attention_wc_prod = torch.mul(attention_cvector_batch[item_in_batch,:],
                                          context_word_embed_bert_layer_12[item_in_batch,:,:]).sum(dim=1) # shape (1,1)
            attention_wcprod_batch.append(attention_wc_prod)
        
        attention_wcprod_batch = torch.cat(attention_wcprod_batch,dim=0) # shape (N,1)
        context_pred = self.sigmoid(attention_wcprod_batch).unsqueeze(-1) # shape (N,1)
    
        return y_pred, context_pred, attention_cvector_batch

def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print("Reset trainable parameters of layer : %s"%str(layer))
            layer.reset_parameters()