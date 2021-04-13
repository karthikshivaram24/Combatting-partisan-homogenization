import torch
import pickle


def load_pickle(file):
    """
    """
    obj = None
    with open(file,'rb') as handle:
        obj = pickle.load(handle)
    return obj


def batch_text_gen(input_1_list,
                   input_2_list,
                   output_1_list,
                   output_2_list,
                   which_cluster,batch_size=500):
    """
    """
    for ndx in range(0,len(input_1_list),batch_size):
        yield input_1_list[ndx:min(ndx+batch_size,len(input_1_list))], input_2_list[ndx:min(ndx+batch_size,len(input_2_list))], output_1_list[ndx:min(ndx+batch_size,len(output_1_list))], output_2_list[ndx:min(ndx+batch_size,len(output_2_list))], which_cluster[ndx:min(ndx+batch_size,len(which_cluster))]

def batch_gen_pred(x1_tensor,x2_tensor,y1_tensor,y2_tensor,batch_size):
    """
    """

    for ndx in range(0,list(x1_tensor.size())[0],batch_size):
        yield x1_tensor[ndx:min(ndx+batch_size,list(x1_tensor.size())[0]),:],  x2_tensor[ndx:min(ndx+batch_size,list(x2_tensor.size())[0]),:], y1_tensor[ndx:min(ndx+batch_size,list(y1_tensor.size())[0])], y2_tensor[ndx:min(ndx+batch_size,list(y2_tensor.size())[0])]

def tokenize_4bert_batch(input1,input2,output1,output2,tokenizer,cuda_device=torch.device('cuda:1')):
    """
    """
    
    input_1_cor = []
    
    for a,b in zip(input1,input2):
        input_1_cor.append(a.replace(b,""))
    
    
    tokenized_tensor = torch.LongTensor([tokenizer.encode(text,
                                                          truncation=True,
                                                          padding="max_length",
                                                          max_length=500, 
                                                          add_special_tokens=True)
                                                          for text in input_1_cor])
    
    tokenized_context_word = torch.LongTensor([tokenizer.encode(word,truncation=True,add_special_tokens=False,max_length=1) for word in input2])
    
    class_labels = torch.FloatTensor(output1)
    word_labels = torch.FloatTensor(output2)
    
    tokenized_tensor = tokenized_tensor.to(cuda_device)
    tokenized_context_word = tokenized_context_word.to(cuda_device)
    class_labels = class_labels.to(cuda_device)
    word_labels = word_labels.to(cuda_device)
    
    return tokenized_tensor, tokenized_context_word, class_labels, word_labels