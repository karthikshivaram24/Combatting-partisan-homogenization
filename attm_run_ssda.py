import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
from Scripts.utils.bert_utils import load_tokenizer
from nltk.corpus import stopwords 
from Scripts.utils.preprocess_utils import preprocess_texts
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random
import itertools
from Scripts.utils.bert_embeddings import load_bert_embeddings
from Scripts.utils.clustering_utils import run_clustering, get_cluster_sizes, score_cluster, get_cluster_pairs, get_pairwise_dist, cluster2doc, filter_clusters, get_top_100_clusterpairs
from Scripts.utils.general_utils import timer
import Scripts.utils.config as CONFIG
from attm_utils import load_pickle
from attm_dataloaders import CPDatasetMT, CPDatasetST
from attm_data_utils import get_train_test_ssda
from attm_metrics import calculate_scores, calculate_scores_single, get_accuracy_from_logits
from attm_model_utils import evaluate_mt, evaluate_st
from attm_single_task import AttentionST
from attm_multi_task import AttentionMT
import  gc
import time
import pickle
import argparse
import attm_config
import os
import dill
import ast

torch.multiprocessing.set_sharing_strategy('file_system')

def run_ssda_cp_mt(df,cp,doc_2_cluster_map,
                   learning_rates=[0.0001,0.001,0.01,0.1],
                   epochs=3,
                   word_pred_loss_weights=[0.3,0.8],
                   batch_size=8,
                   neg_sample_size=3,
                   single_task=False,
                   cuda_device=torch.device('cuda:1')):
    """
    Uses a self supervised domain adaptation setting to train the model
    
    * train = 90% c1 data + 10% c2 data
    * test = 50% c1 data + 50% c2 data
    
    * From cluster 1 choose 70% of the data for train, 30% for test
    * From cluster 2 choose 10% of 70% from c1 for train, 
    
    * Loss check
    * Metrics = F1, recall, precision, accuracy, roc
    """
    train, test = get_train_test_ssda(df,cp,doc_2_cluster_map,neg_sample_size=neg_sample_size,single_task=single_task)
    metrics_train = {}
    metrics_test = {}
    losses_train = {}
    # train ssda func
    for lr in learning_rates:
        for word_loss_w in word_pred_loss_weights:
            
            # FREE MEMORY
            model = None
            gc.collect()
            torch.cuda.empty_cache()

            model, epoch_losses, scores_train, scores_test = run_ssda_MT(train,test,lr,word_loss_w,epochs=epochs,batch_size=batch_size,cuda_device=cuda_device,num_workers=4)
            metrics_train[(lr,word_loss_w)] = scores_train
            metrics_test[(lr,word_loss_w)] = scores_test
            losses_train[(lr,word_loss_w)] = epoch_losses
            
    return metrics_train,metrics_test, losses_train

def run_ssda_cp_st(df,cp,doc_2_cluster_map,
                   learning_rates=[0.0001,0.001,0.01,0.1],
                   epochs=3,
                   word_pred_loss_weights=[0.3,0.8],
                   batch_size=8,
                   neg_sample_size=3,
                   single_task=True,
                   with_attention=True,
                   cuda_device=torch.device('cuda:1')):
    """
    Uses a self supervised domain adaptation setting to train the model
    
    * train = 90% c1 data + 10% c2 data
    * test = 50% c1 data + 50% c2 data
    
    * From cluster 1 choose 70% of the data for train, 30% for test
    * From cluster 2 choose 10% of 70% from c1 for train, 
    
    * Loss check
    * Metrics = F1, recall, precision, accuracy, roc
    """
    train, test = get_train_test_ssda(df,cp,doc_2_cluster_map,neg_sample_size=neg_sample_size,single_task=single_task)
    metrics_train = {}
    metrics_test = {}
    losses_train = {}
    # train ssda func
    for lr in learning_rates:
            
        # FREE MEMORY
        model = None
        gc.collect()
        torch.cuda.empty_cache()

        model, epoch_losses, scores_train, scores_test = run_ssda_ST(train,test,lr,epochs=epochs,batch_size=batch_size,cuda_device=cuda_device,num_workers=4,with_attention=with_attention)

        metrics_train[(lr)] = scores_train
        metrics_test[(lr)] = scores_test
        losses_train[(lr)] = epoch_losses
            
    return metrics_train,metrics_test, losses_train

@timer
def run_ssda_MT(train,test,lr,word_loss_w,epochs=2,batch_size=8,cuda_device=torch.device('cuda:1'),num_workers=1):
    """
    """
    model = AttentionMT(embedding_size=768,verbose=False,which_forward=2)
    model.to(cuda_device)
    loss_func = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    
    epoch_losses = {}
    total_losses = []
    word_losses = []
    rs_losses = []
    
    train_dataset = CPDatasetMT(train)
    test_dataset = CPDatasetMT(test)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    
    for epoch in range(epochs):
        
        for batch_num, (x1,x2,y1,y2,wc) in enumerate(train_dataloader):

            x1,x2,y1,y2 = x1.to(cuda_device),x2.to(cuda_device),y1.to(cuda_device),y2.to(cuda_device)
            
            opt.zero_grad() # reset all the gradient information
    
            y_pred, context_pred, attention_vector = model(x1, x2)
            
            rec_loss = loss_func(y_pred,y1)
            word_loss = loss_func(context_pred,y2)
            
            total_loss = rec_loss + (word_loss_w * word_loss)
            
            total_loss.backward()
            
            opt.step()
            
            rs_losses.append(rec_loss.item())
            word_losses.append(word_loss.item())
            total_losses.append(total_loss.item())
            
            if batch_num % 100 == 0 and batch_num >=100:
                print("Epoch : %s | Batch : %s | Total Loss : %s | Rec Loss : %s | Word Loss : %s" % (str(epoch),str(batch_num),str(total_loss.item()),str(rec_loss.item()),str(word_loss.item())))
                print("True Rec Labels : %s" %str(y1))
                print("True Word Labels : %s" %str(y2))
                print("Batch Class Predictions : %s"%str(y_pred))
                print("Batch Word Label Predictions : %s"%str(context_pred))
                print("Batch Accuracy class : %s"%str(get_accuracy_from_logits(y_pred,y1)))
                print("Batch Accuracy word : %s"%str(get_accuracy_from_logits(context_pred,y2)))
            
            
    epoch_losses["rs_loss"] = rs_losses
    epoch_losses["word_loss"] = word_losses
    epoch_losses["total_loss"] = total_losses

    scores_train = evaluate_mt(model,train_dataloader,device=cuda_device)
    scores_test = evaluate_mt(model,test_dataloader,device=cuda_device)
    
    x1 = None
    x2 = None
    y1 = None
    y2 = None
    rec_loss = None
    total_loss = None
    word_loss = None
    opt = None
    

    del x1
    del x2
    del y1
    del y2

    gc.collect()
    torch.cuda.empty_cache()
    
    return model, epoch_losses, scores_train, scores_test

@timer
def run_ssda_ST(train,test,lr,epochs=2,batch_size=8,cuda_device=torch.device('cuda:1'),num_workers=1,with_attention=True):
    """
    """
    model = AttentionST(embedding_size=768,verbose=False,which_forward=2,with_attention=with_attention)
    model.to(cuda_device)
    loss_func = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    
    epoch_losses = {}
    total_losses = []
    word_losses = []
    rs_losses = []
    
    train_dataset = CPDatasetST(train)
    test_dataset = CPDatasetST(test)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=20,num_workers=num_workers,shuffle=True)
    
    for epoch in range(epochs):
        
        for batch_num, (x1,y1,wc) in enumerate(train_dataloader):

            x1,y1 = x1.to(cuda_device),y1.to(cuda_device)
            
            opt.zero_grad() # reset all the gradient information
    
            y_pred, attention_vector = model(x1)
            
            total_loss = loss_func(y_pred,y1.squeeze()) 
            
            total_loss.backward()
            
            opt.step()
            
            total_losses.append(total_loss.item())
            
            if batch_num % 100 == 0 and batch_num >=100:
                print("Epoch : %s | Batch : %s | Total Loss : %s " % (str(epoch),str(batch_num),str(total_loss.item())))
                print("True Rec Labels : %s" %str(y1))
                print("Batch Class Predictions : %s"%str(y_pred))
                print("Batch Accuracy class : %s"%str(get_accuracy_from_logits(y_pred,y1)))
            
    epoch_losses["total_loss"] = total_losses
        
    scores_train = evaluate_st(model,train_dataloader,device=cuda_device)
    scores_test = evaluate_st(model,test_dataloader,device=cuda_device)
    
    x1 = None
    y1 = None
    total_loss = None
    opt = None
    

    del x1
    del y1

    gc.collect()
    torch.cuda.empty_cache()
    
    return model, epoch_losses, scores_train, scores_test
    
if __name__ == "__main__":
    
    # take args for cluster pairs and cuda devices
    # single or multitask
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-cp",
                        "--cluster_pair", 
                        type=str,
                        default="(46,56)",
                        help= "The cluster pair to use")
    
    parser.add_argument("-d",
                        "--cuda_device", 
                        type=str,
                        default="1",
                        help= "The cuda device to run the model on")
    
    parser.add_argument("-t",
                        "--task",
                        type=str,
                        default="single",
                        help= "The type of model to run single task or multitask")
    
    parser.add_argument("-a",
                        "--at", 
                        type=str, 
                        default="True",
                        help="Whether to use attention in the single task model")
    
    args = parser.parse_args()
    print(args)
    
    df = load_pickle(attm_config.pickle_file_path+os.path.sep+"clean_df.pickle")
    print("Dataset Shape : %s" %str(df.shape))
    doc_2_cluster_map = load_pickle(attm_config.pickle_file_path+os.path.sep+"d2c_map.pickle")
    
    if args.task == "single" and args.at == "True":
        
        print("Setting : Single Task with Attention")
        metrics_train,metrics_test, losses_train = run_ssda_cp_st(df=df,
                                                                  cp = ast.literal_eval(args.cluster_pair),
                                                                  doc_2_cluster_map=doc_2_cluster_map,
                                                                  learning_rates=attm_config.lr,
                                                                   epochs=attm_config.epochs,
                                                                   word_pred_loss_weights=attm_config.word_weights,
                                                                   batch_size=attm_config.batch_size,
                                                                   neg_sample_size=3,
                                                                   single_task=True,
                                                                   with_attention=True,
                                                                   cuda_device=torch.device('cuda:'+args.cuda_device))
        
        save_path = attm_config.save_pickle_results_path
        
        with open(save_path+os.path.sep+"%s_metrics_train_single_att.pickle"%args.cluster_pair,"wb") as mtw, open(save_path+os.path.sep+"%s_metrics_test_single_att.pickle"%args.cluster_pair,"wb") as mtew, open(save_path+os.path.sep+"%s_loss_single_att.pickle"%args.cluster_pair,"wb") as lp:
            dill.dump(metrics_train,mtw,protocol=pickle.HIGHEST_PROTOCOL)
            dill.dump(metrics_test,mtew,protocol=pickle.HIGHEST_PROTOCOL)
            dill.dump(losses_train,lp,protocol=pickle.HIGHEST_PROTOCOL)
    
    if args.task == "single" and args.at == "False":
        
        print("Setting : Single Task without Attention")
        
        metrics_train,metrics_test, losses_train = run_ssda_cp_st(df=df,
                                                                  cp = ast.literal_eval(args.cluster_pair),
                                                                  doc_2_cluster_map=doc_2_cluster_map,
                                                                  learning_rates=attm_config.lr,
                                                                   epochs=attm_config.epochs,
                                                                   word_pred_loss_weights=attm_config.word_weights,
                                                                   batch_size=attm_config.batch_size,
                                                                   neg_sample_size=3,
                                                                   single_task=True,
                                                                   with_attention=False,
                                                                   cuda_device=torch.device('cuda:'+args.cuda_device))
        
        save_path = attm_config.save_pickle_results_path
        
        with open(save_path+os.path.sep+"%s_metrics_train_single.pickle"%args.cluster_pair,"wb") as mtw, open(save_path+os.path.sep+"%s_metrics_test_single.pickle"%args.cluster_pair,"wb") as mtew, open(save_path+os.path.sep+"%s_loss_single.pickle"%args.cluster_pair,"wb") as lp:
            dill.dump(metrics_train,mtw,protocol=pickle.HIGHEST_PROTOCOL)
            dill.dump(metrics_test,mtew,protocol=pickle.HIGHEST_PROTOCOL)
            dill.dump(losses_train,lp,protocol=pickle.HIGHEST_PROTOCOL)
        
    if args.task == "multi":
        
        print("Setting : Multi Task with Attention")
        
        metrics_train,metrics_test, losses_train = run_ssda_cp_mt(df=df,
                                                                  cp = ast.literal_eval(args.cluster_pair),
                                                                  doc_2_cluster_map=doc_2_cluster_map,
                                                                  learning_rates=attm_config.lr,
                                                                  epochs=attm_config.epochs,
                                                                   word_pred_loss_weights=attm_config.word_weights,
                                                                   batch_size=attm_config.batch_size,
                                                                   neg_sample_size=3,
                                                                   single_task=False,
                                                                   cuda_device=torch.device('cuda:'+args.cuda_device))
        save_path = attm_config.save_pickle_results_path
        
        with open(save_path+os.path.sep+"%s_metrics_train_multi_att.pickle"%args.cluster_pair,"wb") as mtw, open(save_path+os.path.sep+"%s_metrics_test_multi_att.pickle"%args.cluster_pair,"wb") as mtew, open(save_path+os.path.sep+"%s_loss_multi_att.pickle"%args.cluster_pair,"wb") as lp:
            dill.dump(metrics_train,mtw,protocol=pickle.HIGHEST_PROTOCOL)
            dill.dump(metrics_test,mtew,protocol=pickle.HIGHEST_PROTOCOL)
            dill.dump(losses_train,lp,protocol=pickle.HIGHEST_PROTOCOL)