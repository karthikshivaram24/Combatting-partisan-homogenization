import torch
from torch import nn
from transformers import BertModel, BertTokenizer


class AttentionMTUpdated(nn.Module):
    
    def __init__(self,embedding_size=768,verbose=True,which_forward=2,dropout=0.1):
        super(AttentionMTUpdated,self).__init__()
        
        self.verbose = verbose
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        
        for p in self.bert.parameters():
            p.requires_grad = False
        
        self.attention = nn.Linear(in_features=embedding_size,
                                   out_features=1,
                                   bias=False)
        
        # New stuff trial
        self.attention2 = nn.Linear(in_features=embedding_size, 
                                    out_features=1, bias=False)
        
        self.attention_combine = nn.Linear(in_features=2*embedding_size,out_features=embedding_size,bias=False)
        
        self.interm_word = nn.Linear(in_features=embedding_size,
                                    out_features=embedding_size,
                                    bias=True)
        
        self.dropout = nn.Dropout(dropout)
        
        self.word_pred = nn.Linear(in_features=1,
                                   out_features=1,
                                   bias=True)
        
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
    
    def forward(self,bert_tokenized_words,attention_masks,bert_tokenized_word_to_predict):
        """
        """
        
        # Recommendation Part
        
        bert_output = self.bert(input_ids=bert_tokenized_words,attention_mask=attention_masks)
        bert_hidden_states  = bert_output[2]
        bert_layer12_hidden_states = bert_hidden_states[-1]
        
        # iterate over articles in a given batch
        attention_vector_batch = []
#         attention_vector_batch_2 = []

        for sent in range(bert_layer12_hidden_states.size(0)):

            sent_embed_matrix = bert_layer12_hidden_states[sent,:,:] # 500 x 768

            word_weights = self.dropout(self.attention(sent_embed_matrix)).squeeze()

            self.normalized_word_weights = self.softmax(word_weights) # input is 500 x 1, output is 500 x 1

            attention_vector = sent_embed_matrix.T .mul(self.normalized_word_weights).sum(dim=1)

            attention_vector_batch.append(attention_vector.unsqueeze(-1).T) 
                
                
                
#             word_weights_2 = self.tanh(self.dropout(self.attention2(sent_embed_matrix))).squeeze()

#             self.normalized_word_weights_2 = self.softmax(word_weights_2) # input is 500 x 1, output is 500 x 1

#             attention_vector_2 = sent_embed_matrix.T .mul(self.normalized_word_weights_2).sum(dim=1)

#             attention_vector_batch_2.append(attention_vector_2.unsqueeze(-1).T) 
                
                

        attention_vector_batch = torch.cat(attention_vector_batch,dim=0) # we get (batch_size,768)
#         attention_vector_batch_2 = torch.cat(attention_vector_batch_2,dim=0)
            
#         total_attention = torch.cat([attention_vector_batch,attention_vector_batch_2],dim=1)

        y_preds = self.sigmoid(self.recom_pred(attention_vector_batch))
        
        
        # Word Prediction Part
        
        context_word_embed_bert = self.bert(input_ids=bert_tokenized_word_to_predict)
        context_word_embed_bert_hs = context_word_embed_bert[2]
        context_word_embed_bert_layer_12 = context_word_embed_bert_hs[-1] # shape (N,1,768)
        
        # convert 3d to 2d matrix
        original_size = context_word_embed_bert_layer_12.size()
        context_word_embed_bert_layer_12 = context_word_embed_bert_layer_12.view(original_size[0],original_size[-1])
        
        # Convert dual attention layer weights to single set of weights for dot product
#         total_attention = self.softmax(self.dropout(self.attention_combine(total_attention)))
        
        dot_prod_sim = torch.mul(attention_vector_batch, context_word_embed_bert_layer_12).sum(dim=1) # dot product between (N,768) and (N,768)
        
        word_preds = self.sigmoid(dot_prod_sim)
        
        word_preds = word_preds.view(word_preds.size()[0],1)
        
        return y_preds,word_preds, None, None