import torch
import torch.nn as nn




def create_emb_layer(weights_matrix, non_trainable=False):
    """
    Loads in pre-trained embeddings like glove into a torch embedding lookup
    """
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class AttentionST(nn.Module):
    """
    The single task network class
    """
    
    def __init__(self,embedding_size=768,verbose=True,which_forward=2,with_attention=True,dropout=0.3,glove=False,weights_matrix=None):
        super(AttentionST,self).__init__()
        
        self.with_attention = with_attention
        self.embedding_size = embedding_size
        
        self.glove = glove
        
        if self.glove:
        
            self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, False)
        
        if self.glove:
            self.embedding_size=300
        
        self.attention = nn.Linear(in_features=self.embedding_size,
                                       out_features=1,
                                       bias=False)
        
        
        self.dropout = nn.Dropout(dropout)
        
        self.recom_pred = nn.Linear(in_features=self.embedding_size,
                                    out_features=1,
                                    bias=True)

        
        # Softmax Activation
        self.softmax = nn.Softmax(dim=-1)
        
        # Sigmoid Activation
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self,embeddings):
        """
        """
        
        batch_size = embeddings.size(0)
        
        if self.with_attention:
        # iterate over articles in a given batch
            attention_vector_batch = []

            for sent in range(batch_size):
                
                sent_embed_matrix = None
                
                if not self.glove:

                    sent_embed_matrix = embeddings[sent,:,:] # 500 x 768
                
                if self.glove:
                    sent_embed_matrix = self.embedding(embeddings[sent,:])

                word_weights = self.dropout(self.attention(sent_embed_matrix)).squeeze()

                self.normalized_word_weights = self.softmax(word_weights) # input is 500 x 1, output is 500 x 1

                attention_vector = sent_embed_matrix.T .mul(self.normalized_word_weights).sum(dim=1)

                attention_vector_batch.append(attention_vector.unsqueeze(-1).T) 
                

            attention_vector_batch = torch.cat(attention_vector_batch,dim=0) # we get (batch_size,768)
            
            y_preds = self.sigmoid(self.recom_pred(attention_vector_batch)).squeeze()
        
            return y_preds
        
        else:
            
            y_pred = None
            
            if self.glove:
                glove_embeds = torch.stack([torch.mean(self.embedding(embedding),dim=0) for embedding in embeddings])
                y_pred = self.sigmoid(self.recom_pred(glove_embeds)).squeeze()
            
            if not self.glove:
                y_pred = self.sigmoid(self.recom_pred(embeddings[:,0,:])).squeeze()
            
            return y_pred


        

class AttentionMT(nn.Module):
    """
    The multitask network class
    """
    
    def __init__(self,embedding_size=768,dropout=0.1,glove=False,weights_matrix=None,bad_embeds=None,use_loss2=False):
        super(AttentionMT,self).__init__()
        
        self.glove = glove
        self.embedding_size=embedding_size
        
        if self.glove:
            self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, False)
            
            self.embedding_size = 300
        
        self.attention = nn.Linear(in_features=self.embedding_size,
                                   out_features=1,
                                   bias=False)
        
        self.interm_word = nn.Linear(in_features=self.embedding_size,
                                    out_features=self.embedding_size,
                                    bias=True)
        
        self.dropout = nn.Dropout(dropout)
        
        self.word_weights = nn.Linear(in_features=self.embedding_size,
                                   out_features=1,
                                   bias=True)
        
        # Recommendation Network
        # ------------------------
        
        self.recom_pred = nn.Linear(in_features=self.embedding_size,
                                    out_features=1,
                                    bias=True)
        
        # Softmax Activation
        self.softmax = nn.Softmax(dim=-1)
        
        # Sigmoid Activation
        self.sigmoid = nn.Sigmoid()
        
        self.att_con_vec = None
        
        self.bad_embeds = bad_embeds
        
        self.use_loss2 = use_loss2
    
    def forward(self,text_embeds,context_embed):
        """
        """        
        attention_vector_batch = []
        
        batch_size = text_embeds.size(0)
        
        for sent in range(batch_size):
            
            sent_embed_matrix = None

            if not self.glove:

                sent_embed_matrix = text_embeds[sent,:,:] # 500 x 768

            if self.glove:
                sent_embed_matrix = self.embedding(text_embeds[sent,:])

            word_weights = self.dropout(self.attention(sent_embed_matrix)).squeeze()

            self.normalized_word_weights = self.softmax(word_weights) # input is 500 x 1, output is 500 x 1

            attention_vector = sent_embed_matrix.T.mul(self.normalized_word_weights).sum(dim=1)

            attention_vector_batch.append(attention_vector.unsqueeze(-1).T) 
                
                
        self.att_con_vec  = torch.cat(attention_vector_batch,dim=0) # we get (batch_size,768)

        y_preds = self.sigmoid(self.recom_pred(self.att_con_vec))
        
        
        word_preds = None
        
        if context_embed is not None:
        # Word Prediction Part
        
            # convert 3d to 2d matrix
            if self.glove:
                context_embed = self.embedding(context_embed)
            
            if not self.glove:
                original_size = context_embed.size()
                context_embed = context_embed.view(original_size[0],original_size[-1])

            dot_prod_sim = torch.mul(self.att_con_vec, context_embed) # dot product between (N,768) and (N,768)

            word_preds = self.sigmoid(self.word_weights(dot_prod_sim))

            word_preds = word_preds.view(word_preds.size()[0],1)
        
        return y_preds,word_preds,self.att_con_vec