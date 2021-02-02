import torch
import pickle
from transformers import BertModel, BertTokenizer
from general_utils import timer
import os
import numpy as np

def load_model(m_str='bert-base-uncased'):
    """
    """
    # has limit of 512 sequence length
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()
    model = model.to('cuda')
    return model

def load_tokenizer(t_str="bert-base-uncased"):
    """
    """
    tokenizer = BertTokenizer.from_pretrained(t_str)
    return tokenizer

def tokenize_bert(text_batch,tokenizer):
    """
    """
    tokenized_tensor = torch.LongTensor([tokenizer.encode(text,
                                                          truncation=True,
                                                          padding="max_length",
                                                          max_length=500, 
                                                          add_special_tokens=True)  # Add [CLS] and [SEP],) 
                                                          for text in text_batch])
    tokenized_tensor = tokenized_tensor.to('cuda')
    return tokenized_tensor
        
def batch_text_gen(text_list,batch_size=500):
    """
    """
    for ndx in range(0,len(text_list),batch_size):
        yield text_list[ndx:min(ndx+batch_size,len(text_list))]

@timer
def infer_embeddings(text_list,
                     batch_size=50,
                     save_folder_1="/media/karthikshivaram/Extra_disk_1/Bert_model_outputs",
                     save_folder_2="/media/karthikshivaram/Extra_Disk_2/Bert_model_outputs"):
    """
    """
    # 512 token limit
    
    model = load_model()
    tokenizer = load_tokenizer()
    num_batches = int(len(text_list)/batch_size)
    print("Number of Batches : %s\n"%str(num_batches))
    batches_sf1 = 0
    batches_sf2 = 0
    with torch.no_grad():
        batch_no = 0
        for text_batch in batch_text_gen(text_list,batch_size):
            if batch_no > 0 and batch_no % 100 == 0:
                print("Running Batch : %s"%str(batch_no))
            batch_tensor = tokenize_bert(text_batch,tokenizer)
            batch_out = model(input_ids=batch_tensor)
            batch_hidden_states = batch_out[2]
            batch_all_layer_tensor = torch.cat(batch_hidden_states[1:],2)
            
            if batch_no == 0:
                print(batch_all_layer_tensor.size())
            
            if batch_no < int(num_batches/2):
                np.save("%s/%s.npy"%(save_folder_1,str(batch_no)),batch_all_layer_tensor.cpu().numpy())
                batches_sf1 +=1
                    
            else:
                np.save("%s/%s.npy"%(save_folder_2,str(batch_no)),batch_all_layer_tensor.cpu().numpy())
                batches_sf2 +=1
                
            batch_no +=1
    
    print("\nTotal Batches Saved : %s" %str(batch_no))
    print("Batches saved to : %s    \n%s"%(save_folder_1,str(batches_sf1)))
    print("Batches saved to : %s    \n%s"%(save_folder_2,str(batches_sf2)))

def load_bert_embeddings(folder="Bert_embed_files"):
    """
    """
    files = os.listdir(folder)
    files = sorted(files,key=lambda x: int(x.split(".")[0]),reverse=False)
    print(files[:10])
    batch_list_arrs = []
    for f in files:
        loaded_tensor = torch.load(folder+os.path.sep+f).numpy()
        batch_list_arrs.append(loaded_tensor)
    
    return np.concatenate(batch_list_arrs,axis=0)
        

def infer_embed_test():
    """
    Notes:
    * Number of hidden states is 13 why ?
    * output of the embeddings + one for the output of each layer (12)
    * output of the embeddings is the vector that's fed into the first layer of bert
    * output of the embedding = sum of the token embeddings + the segment embeddings + the position embeddings.
    
    Reference : https://github.com/huggingface/transformers/issues/2332
    """
    model = load_model()
    tokenizer = load_tokenizer()
    with torch.no_grad():
        test_sent = "this is a test"
        test_sent2 = "this is another test"
        tokenized_sent = torch.LongTensor([tokenizer.encode(test,
                                                           add_special_tokens=True,  # Add [CLS] and [SEP],
                                                           max_length = 10,  # maximum length of a sentence
                                                           padding="max_length",  # Add [PAD]s
                                                           truncation=True) for test in [test_sent,test_sent2]])
        
        print("tokenized output size : %s" %str(tokenized_sent.size()))
        print("tokenized output ids : \n%s" %str(tokenized_sent))
        print("tokenized output tokens : \n%s"%str(tokenizer.convert_ids_to_tokens(tokenized_sent[0])))
        tokenized_tensor = tokenized_sent.to("cuda")
        out = model(input_ids=tokenized_tensor)
        print("\nOutput Type : \n%s"%str(type(out)) )
        print("\nOutput Attr : \n%s"%str(out.__dict__.keys()))
        print("\nHidden States Type : \n%s"%str(type(out[2])))
        print("\nHidden States Length : \n%s"%str(len(out[2])))
        for i,n in enumerate(out[2]):
            print("Layer %s Hidden State size : %s" %(str(i),str(n.size())))
        
        print("\nPickle Test :")
        print("\nSaving Hidden State tuple as Pickle")
        with open("bert_pickle_hs_test.pkl",'wb') as bw:
            pickle.dump(out[2][1:],bw)
        print("Finished Saving")
        print("\nLoading Hidden State tuple from Pickle File")
        loaded_hs = None
        with open("bert_pickle_hs_test.pkl","rb") as br:
            loaded_hs = pickle.load(br)
        print("Finished Loading")
        print("\nLoaded Type : %s" %str(type(loaded_hs)))
        print("Loaded Size : %s" %str(len(loaded_hs)))
        print("Layer 1 Embedding Size : %s"%str(loaded_hs[0].size()))
        print("All Layer concatenation size : %s" %str(torch.cat(loaded_hs,2).size()))
        print("Numpy Version :\n%s" %str(loaded_hs[0].cpu().numpy()))

@timer
def load_bert_output(folder1="/media/karthikshivaram/Extra_disk_1/Bert_model_outputs",
                     folder2="/media/karthikshivaram/Extra_Disk_2/Bert_model_outputs",
                     layer=12,
                     aggregation="mean"):
    """
    Loads the bert hidden state matrix from the numpy file and performs aggregation
    to get the sentence vector and then combines the batch output to generate one overall 
    matrix of vectorized sentences.
    
    aggregation types:
    * max
    * mean
    * mean + max
    * cls (pick first out of all)
    * last 4 concat (concatenate last 4 layers) (This ignores layers argument)
    
    Parameters:
    * folder -> path of saved pickle files of bert output
    * layer -> int , the layer to extract representation from
    * aggregation -> str, the aggregation method to perform to convert token embeddings to sentence embeddings
    """
    batch_outputs = []
    
    layers_start = [i* 768  for i in range(12)]
    layers_stop = [i+768 for i in layers_start]
    
    def get_batch_arr(file,layer):
        """
        """
        batch_arr = np.load(f)
        # 3d matrix [batch_size,max_length,all_12_layer_output (12*768)]
        layer_slice_start = layers_start[layer-1]
        layer_slice_stop = layers_stop[layer-1]
        # get [batch_size,max_length,layer_output(768)]
        batch_layer_slice = batch_arr[:,:,layer_slice_start:layer_slice_stop]
        return batch_layer_slice
    
    def get_cls_rep(file,layer,token_index=0):
        """
        """
        batch_arr = np.load(f)
        layer_slice_start = layers_start[layer-1]
        layer_slice_stop = layers_stop[layer-1]
        # get [batch_size,max_length,layer_output(768)]
        batch_layer_slice = batch_arr[:,0,layer_slice_start:layer_slice_stop]
        return batch_layer_slice
    
    def get_max(batch_layer_slice):
        """
        """
        batch_agg_arr = np.max(batch_layer_slice,axis=1)
        return batch_agg_arr
    
    def get_mean(batch_layer_slice):
        """
        """
        batch_agg_arr = np.mean(batch_layer_slice,axis=1)
        return batch_agg_arr
    
    files1 = os.listdir(folder1)
    files1 = [folder1 + os.path.sep+f for f in files1]
    files2 = os.listdir(folder2)
    files2 = [folder2 + os.path.sep+f for f in files2]
    files = files1 + files2
    files = sorted(files,key=lambda x: int(x.split(os.path.sep)[-1].split(".")[0]),reverse=False)
    print("First Ten Files : %s" %str(files[:10]))
    
    for ind_f, f in enumerate(files):
        
        if ind_f > 100 and ind_f%100 == 0:
            print("Processing Batch No : %s" %str(ind_f))
        
        if aggregation == "max":
            batch_layer_slice = get_batch_arr(file=f,layer=layer)
            # mean over max_length axis
            batch_agg_arr = get_max(batch_layer_slice)
            batch_outputs.append(batch_agg_arr)
        
        if aggregation == "mean":
            batch_layer_slice = get_batch_arr(file=f,layer=layer)
            # mean over max_length axis
            batch_agg_arr = get_mean(batch_layer_slice)
            batch_outputs.append(batch_agg_arr)
        
        if aggregation == "mean + max":
            batch_layer_slice = get_batch_arr(file=f,layer=layer)
            batch_agg_arr_max = get_max(batch_layer_slice)
            batch_agg_arr_mean = get_mean(batch_layer_slice)
            batch_outputs.append(np.concatenate([batch_agg_arr_mean,batch_agg_arr_max],axis=1))
        
        if aggregation == "cls":
            batch_layer_slice = get_cls_rep(file=f,layer=layer,token_index=0)
            batch_outputs.append(batch_layer_slice)
            
        if aggregation == "last 4 concat":
            pass
    
    return np.concatenate(batch_outputs,axis=0)