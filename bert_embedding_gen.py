import torch
from transformers import BertModel, BertTokenizer

def get_emebddings(text_batch,model):
    """
    """
    pass

def load_model(m_str='bert-base-uncased'):
    """
    """
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    

def create_batches(text_list,batch_size):
    """
    """
    pass
