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
from attm_data_utils import get_train_test_ssda_updated
from attm_metrics import calculate_scores, calculate_scores_single, get_accuracy_from_logits
from attm_model_utils import evaluate_mt, evaluate_st, EarlyStopping
from attm_single_task import AttentionST, AttentionSTUpdated
from attm_multi_task import AttentionMT
import  gc
import time
import pickle
import argparse
import attm_config
import os
import dill
import ast
import itertools

np.random.seed(CONFIG.RANDOM_SEED)
torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(24)

def run_ssda_cp_mt(df,cp,doc_2_cluster_map,
                   learning_rates=[0.0001,0.001,0.01,0.1],
                   epochs=[3],
                   word_pred_loss_weights=[0.3,0.8],
                   batch_sizes=[8],
                   neg_sample_size=3,
                   dropouts= [0.1,0.3,0.5],
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
    train, test,val = get_train_test_ssda_updated(df,cp,doc_2_cluster_map,neg_sample_size=neg_sample_size,single_task=single_task)
    metrics_train = {}
    metrics_test = {}
    losses_train = {}
    model_dict = {}
    params = [learning_rates,epochs,batch_sizes,dropouts,word_pred_loss_weights]
    
    params_all_combo = list(itertools.product(*params))
    
    print("Number of param combinations : %s" %str(len(params_all_combo)))
    
    for param_list in params_all_combo:
        
        lr = param_list[0]
        epoch = param_list[1]
        batch_size = param_list[2]
        dropout=param_list[3]
        wlw = param_list[4]
        
        print("Running model for ----\n lr : %s\n epoch : %s\n batch_size : %s\n dropout : %s\n"%(str(lr),str(epoch),str(batch_size),str(dropout)))
        
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        

        model, epoch_losses, scores_train, scores_test, end_epoch = run_ssda_MT(train,test,val,lr,wlw,epochs=epoch,batch_size=batch_size,dropout=dropout,cuda_device=cuda_device,num_workers=3)
        metrics_train[(lr,end_epoch+1,batch_size,dropout,wlw)] = scores_train
        metrics_test[(lr,end_epoch+1,batch_size,dropout,wlw)] = scores_test
        losses_train[(lr,end_epoch+1,batch_size,dropout,wlw)] = epoch_losses
#         model_dict[(lr,end_epoch+1,batch_size,dropout,wlw)] = model
            
    return metrics_train,metrics_test, losses_train, model_dict

def run_ssda_cp_st(df,cp,doc_2_cluster_map,
                   learning_rates=[0.0001,0.001,0.01,0.1],
                   epochs=[3],
                   word_pred_loss_weights=[0.3,0.8],
                   batch_sizes=[8],
                   dropouts=[0.1,0.3],
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
    train, test, val = get_train_test_ssda_updated(df,cp,doc_2_cluster_map,neg_sample_size=neg_sample_size,single_task=single_task)
    metrics_train = {}
    metrics_test = {}
    losses_train = {}
    model_dict = None
    # train ssda func
    params = [learning_rates,epochs,batch_sizes,dropouts]
    
    params_all_combo = list(itertools.product(*params))
    
    print("Number of param combinations : %s" %str(len(params_all_combo)))
    
    for param_list in params_all_combo:
        
        lr = param_list[0]
        epoch = param_list[1]
        batch_size = param_list[2]
        dropout=param_list[3]
        
        print("Running model for ----\n lr : %s\n epoch : %s\n batch_size : %s\n dropout : %s\n"%(str(lr),str(epoch),str(batch_size),str(dropout)))
        
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        
        model, epoch_losses, scores_train, scores_test,end_epoch= run_ssda_ST(train,
                                                                     test,
                                                                     val,
                                                                     lr,
                                                                     epochs=epoch,
                                                                     batch_size=batch_size,
                                                                     dropout = dropout,
                                                                     cuda_device=cuda_device,
                                                                     num_workers=4,
                                                                     with_attention=with_attention)

        metrics_train[(lr,end_epoch+1,batch_size,dropout)] = scores_train
        metrics_test[(lr,end_epoch+1,batch_size,dropout)] = scores_test
        losses_train[(lr,end_epoch+1,batch_size,dropout)] = epoch_losses
#         model_dict[(lr,end_poch+1,batch_size,dropout)] = model

        
    return metrics_train,metrics_test, losses_train, model_dict

@timer
def run_ssda_MT(train,test,val,lr,word_loss_w,epochs=2,batch_size=8,dropout=0.1,cuda_device=torch.device('cuda:1'),num_workers=1,patience=2):
    """
    """
    model = AttentionMT(embedding_size=768,verbose=False,which_forward=2,dropout=dropout)
    model.to(cuda_device)
    loss_func = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    
    epoch_losses = {}
    total_losses = []
    word_losses = []
    rs_losses = []
    total_losses_val = []
    word_losses_val = []
    rs_losses_val = []
    
    train_dataset = CPDatasetMT(train)
    test_dataset = CPDatasetMT(test)
    val_dataset = CPDatasetMT(val)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    
    early_stopping = EarlyStopping(patience=patience)
    end_epoch = None
    
    for epoch in range(epochs):
        
        batch_losses_total = []
        batch_losses_rec = []
        batch_losses_word = []
        
        for batch_num, (x1,x2,y1,y2,wc) in enumerate(train_dataloader):
            
            model.train()

            x1,x2,y1,y2 = x1.to(cuda_device),x2.to(cuda_device),y1.to(cuda_device),y2.to(cuda_device)
            
            opt.zero_grad() # reset all the gradient information
    
            y_pred, context_pred, weighted_avg_context_vector,attention_combined_weights = model(x1, x2)
        
            if batch_size == 1:
                y_pred = y_pred.view(1,1)
                context_pred = context_pred.view(1,1)
            
            rec_loss = loss_func(y_pred,y1)
            word_loss = loss_func(context_pred,y2)
            
            total_loss = rec_loss + (word_loss_w * word_loss)
            
            total_loss.backward()
            
            opt.step()
            
            batch_losses_rec.append(rec_loss.item())
            batch_losses_word.append(word_loss.item())
            batch_losses_total.append(total_loss.item())
            
            if batch_num % 100 == 0 and batch_num >=100:
                print("Epoch : %s | Batch : %s | Total Loss : %s | Rec Loss : %s | Word Loss : %s" % (str(epoch),str(batch_num),str(total_loss.item()),str(rec_loss.item()),str(word_loss.item())))
                print("True Rec Labels : %s" %str(y1))
                print("True Word Labels : %s" %str(y2))
                print("Batch Class Predictions : %s"%str(y_pred))
                print("Batch Word Label Predictions : %s"%str(context_pred))
                print("Batch Accuracy class : %s"%str(get_accuracy_from_logits(y_pred,y1)))
                print("Batch Accuracy word : %s"%str(get_accuracy_from_logits(context_pred,y2)))
            
        batch_losses_total_val = []
        batch_losses_rec_val = []
        batch_losses_word_val = []
        
        for bn_v, (xv1,xv2,yv1,yv2,wcv) in enumerate(val_dataloader):
            
            model.eval()
            with torch.no_grad():
                xv1,xv2,yv1,yv2 = xv1.to(cuda_device),xv2.to(cuda_device),yv1.to(cuda_device),yv2.to(cuda_device)

                y_pred, context_pred, weighted_avg_context_vector, attention_combined_weights = model(xv1, xv2)
                
                if batch_size == 1:
                    y_pred = y_pred.view(1,1)
                    context_pred = context_pred.view(1,1)
            
                rec_loss = loss_func(y_pred,yv1)
                word_loss = loss_func(context_pred,yv2)
            
                total_loss = rec_loss + (word_loss_w * word_loss)
                
                batch_losses_rec_val.append(rec_loss.item())
                batch_losses_word_val.append(word_loss.item())
                batch_losses_total_val.append(total_loss.item())
        
        avg_batch_losses_total = np.mean(batch_losses_total)
        avg_batch_losses_rec = np.mean(batch_losses_rec)
        avg_batch_losses_word = np.mean(batch_losses_word)
        
        avg_batch_losses_total_val =np.mean(batch_losses_total_val)
        avg_batch_losses_rec_val = np.mean(batch_losses_rec_val)
        avg_batch_losses_word_val =np.mean(batch_losses_word_val)
        
        print("Avg Batch Total Loss Train : %s" %str(avg_batch_losses_total))
        print("Avg Batch Toal Loss Val : %s" %str(avg_batch_losses_total_val))
        
        total_losses.append(avg_batch_losses_total)
        word_losses.append(avg_batch_losses_word)
        rs_losses.append(avg_batch_losses_rec)
        total_losses_val.append(avg_batch_losses_total_val)
        word_losses_val.append(avg_batch_losses_word_val)
        rs_losses_val.append(avg_batch_losses_rec_val)
        
        early_stopping(avg_batch_losses_total_val)
        
        if early_stopping.early_stop:
            end_epoch = epoch
            break
            
    epoch_losses["rs_loss"] = rs_losses
    epoch_losses["word_loss"] = word_losses
    epoch_losses["total_loss"] = total_losses
    epoch_losses["rs_loss_val"] = rs_losses_val
    epoch_losses["word_loss_val"] = word_losses_val
    epoch_losses["total_loss_val"] = total_losses_val

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
    
    if end_epoch == None:
        end_epoch = epochs
    
    return model, epoch_losses, scores_train, scores_test, end_epoch

@timer
def run_ssda_ST(train,test,val,lr,epochs=2,batch_size=8,dropout=0.1,cuda_device=torch.device('cuda:1'),num_workers=1,with_attention=True,patience=2):
    """
    """
    model = AttentionSTUpdated(embedding_size=768,verbose=False,which_forward=2,with_attention=with_attention,dropout=dropout)
    model.to(cuda_device)
    loss_func = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    
    epoch_losses = {}
    total_losses = []
    total_losses_val = []
    word_losses = []
    rs_losses = []
    
    train_dataset = CPDatasetST(train)
    test_dataset = CPDatasetST(test)
    val_dataset = CPDatasetST(val)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    
    early_stopping_val_loss = 10000.0 # arbitarily high loss value
    early_stopping = EarlyStopping(patience=patience)
    end_epoch = None
    for epoch in range(epochs):
        batch_losses =  []
        # Training
        for batch_num, (x1,y1,wc) in enumerate(train_dataloader):
            
            model.train()

            x1,y1 = x1.to(cuda_device),y1.to(cuda_device)
            
            opt.zero_grad() # reset all the gradient information
    
            y_pred, attention_vector, attention_combined_weights = model(x1)
            
            if batch_size == 1:
                y1 = y1.view(1)
            
            if batch_size > 1:
                y1 = y1.squeeze()
            
            
            total_loss = loss_func(y_pred,y1) 
            
            total_loss.backward()
            
            opt.step()
            
            batch_losses.append(total_loss.item())
            
            if batch_num % 100 == 0 and batch_num >=100:
                print("Epoch : %s | Batch : %s | Total Loss : %s " % (str(epoch),str(batch_num),str(total_loss.item())))
                print("True Rec Labels : %s" %str(y1))
                print("Batch Class Predictions : %s"%str(y_pred))
                print("Batch Accuracy class : %s"%str(get_accuracy_from_logits(y_pred,y1)))
        
        # Validation
        batch_losses_val =  []
        for bn_v, (xv1,yv1,wcv) in enumerate(val_dataloader):
            
            model.eval()
            with torch.no_grad():
                xv1,yv1 = xv1.to(cuda_device),yv1.to(cuda_device)

                y_pred, attention_vector, attention_combined_weights = model(xv1)
                
                if batch_size == 1:
                    yv1 = yv1.view(1)
            
                if batch_size > 1:
                    yv1 = yv1.squeeze()

                total_loss_val = loss_func(y_pred,yv1) 
                batch_losses_val.append(total_loss_val.item())
        
        avg_batch_loss = np.mean(batch_losses)
        avg_batch_loss_val = np.mean(batch_losses_val)
        print("Avg Batch Loss Train : %s" %str(avg_batch_loss))
        total_losses.append(avg_batch_loss)
        print("Avg Batch Loss Val : %s" %str(avg_batch_loss_val))
        total_losses_val.append(avg_batch_loss_val)
        
        early_stopping(avg_batch_loss_val)
        
        if early_stopping.early_stop:
            end_epoch = epoch
            break
        
#         if early_stopping_val_loss > avg_batch_loss_val:
#             early_stopping_val_loss = avg_batch_loss_val
        
#         if early_stopping_val_loss <= avg_batch_loss_val:
#             print("Validation Loss not decreasing - performing early stopping ...........")
#             break
        
    epoch_losses["total_loss"] = total_losses
    epoch_losses["total_loss_val"] = total_losses_val
        
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
    
    if end_epoch == None:
        end_epoch = epochs
    
    return model, epoch_losses, scores_train, scores_test, end_epoch
    
    
    
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
        metrics_train,metrics_test, losses_train,model_dict = run_ssda_cp_st(df=df,
                                                                  cp = ast.literal_eval(args.cluster_pair),
                                                                  doc_2_cluster_map=doc_2_cluster_map,
                                                                  learning_rates=attm_config.lr,
                                                                   epochs=attm_config.epochs,
                                                                   word_pred_loss_weights=attm_config.word_weights,
                                                                   batch_sizes=attm_config.batch_size,
                                                                   dropouts = attm_config.dropouts,
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
        
        metrics_train,metrics_test, losses_train, model_dict = run_ssda_cp_st(df=df,
                                                                  cp = ast.literal_eval(args.cluster_pair),
                                                                  doc_2_cluster_map=doc_2_cluster_map,
                                                                  learning_rates=attm_config.lr,
                                                                   epochs=attm_config.epochs,
                                                                   word_pred_loss_weights=attm_config.word_weights,
                                                                   batch_sizes=attm_config.batch_size,
                                                                   dropouts = attm_config.dropouts,
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
        
        metrics_train,metrics_test, losses_train, model_dict = run_ssda_cp_mt(df=df,
                                                                  cp = ast.literal_eval(args.cluster_pair),
                                                                  doc_2_cluster_map=doc_2_cluster_map,
                                                                  learning_rates=attm_config.lr,
                                                                   epochs=attm_config.epochs,
                                                                   word_pred_loss_weights=attm_config.word_weights,
                                                                   batch_sizes=attm_config.batch_size,
                                                                   dropouts = attm_config.dropouts,
                                                                   neg_sample_size=3,
                                                                   single_task=False,
                                                                   cuda_device=torch.device('cuda:'+args.cuda_device))
        save_path = attm_config.save_pickle_results_path
        
        with open(save_path+os.path.sep+"%s_metrics_train_multi_att.pickle"%args.cluster_pair,"wb") as mtw, open(save_path+os.path.sep+"%s_metrics_test_multi_att.pickle"%args.cluster_pair,"wb") as mtew, open(save_path+os.path.sep+"%s_loss_multi_att.pickle"%args.cluster_pair,"wb") as lp:
            dill.dump(metrics_train,mtw,protocol=pickle.HIGHEST_PROTOCOL)
            dill.dump(metrics_test,mtew,protocol=pickle.HIGHEST_PROTOCOL)
            dill.dump(losses_train,lp,protocol=pickle.HIGHEST_PROTOCOL)