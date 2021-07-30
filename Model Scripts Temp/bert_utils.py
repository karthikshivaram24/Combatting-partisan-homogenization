
import torch
import tqdm
from transformers import BertModel, BertTokenizer


def batch_text_gen(text_list,batch_size=500):
    """
    """
    for ndx in range(0,len(text_list),batch_size):
        yield text_list[ndx:min(ndx+batch_size,len(text_list))]


def load_model(m_str='bert-base-uncased'):
    """
    """
    # has limit of 512 sequence length
    model = BertModel.from_pretrained(m_str, output_hidden_states=True)
    model.eval()
    model = model.to('cuda')
    return model


def load_tokenizer(t_str="bert-base-uncased"):
    """
    """
    tokenizer = BertTokenizer.from_pretrained(t_str)
    return tokenizer

def get_bert_embeddings(df,model,tokenizer):
    """
    should return a list of dictionaries where dictionary has context word embedding and label
    """
    
    texts_pos = df["context_pos_words"].tolist()
    texts_neg = df["context_neg_words"].tolist()
    # single_pos = [pos[0] for pos in texts_pos]
    # single_neg = [neg[0] for neg in texts_neg]
    
    embeds_list = []
    
    with torch.no_grad():
        for i in tqdm(range(len(texts_pos)),total=len(texts_pos)):
            
            embed_dict = {}

            encoded_pos = tokenizer.encode_plus(texts_pos[i], add_special_tokens=False, max_length = 1, padding='max_length', return_attention_mask = False,truncation=True, return_tensors = 'pt')
            encoded_neg = tokenizer.encode_plus(texts_neg[i], add_special_tokens=False, max_length = 1, padding='max_length', return_attention_mask = False,truncation=True, return_tensors = 'pt')

            token_pos = encoded_pos["input_ids"]
            token_neg = encoded_neg["input_ids"]

            token_tensors = torch.cat([token_pos,token_neg],dim=0)
            token_tensors = token_tensors.to(torch.device('cuda:1'))
            batch_out = model(token_tensors)
            batch_hidden_states = batch_out[2]
            batch_12_layer_tensor = batch_hidden_states[-1]

            pos_embed = batch_12_layer_tensor[0,:,:].cpu()
            neg_embed = batch_12_layer_tensor[1,:,:].cpu()
            
            embed_dict["pos"] = pos_embed
            embed_dict["neg"] = neg_embed
            
            embeds_list.append(embed_dict)
    
    return embeds_list