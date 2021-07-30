import torch
import itertools
import gc
from model_data_utils import split_data_mixed, filter_4_pure, load_loss3_weights
from torch_model_runners import run_singleTask_exp, run_multiTask_exp
from loss_plotters import plot_epoch_loss, plot_epoch_loss_multi
from bert_utils import load_tokenizer, load_model, get_bert_embeddings
from metric_aggregators import print_settings

def gr_search_singleTask(df,c1s,c2,doc_2_cluster_map,
                   c2_train_perc=0.1,
                   learning_rates=[0.0001,0.001,0.01,0.1],
                   epochs=[3],
                   batch_sizes=[8],
                   dropouts=[0.0,0.1,0.3,0.5],
                   with_attention=True,
                   cuda_device=torch.device('cuda:1'), 
                   return_model=False,avg_type="binary",glove=False,extreme=False,weight_decays=[0.1,0.01,0.001,0.0001],pure=False):
    """
    Uses a self supervised domain adaptation setting to train the model
    
    * train = 90% c1 data + 10% c2 data
    * test = 50% c1 data + 50% c2 data
    
    * From cluster 1 choose 70% of the data for train, 30% for test
    * From cluster 2 choose 10% of 70% from c1 for train, 
    
    * Loss check
    * Metrics = F1, recall, precision, accuracy, roc
    """
#     train, test, val = split_data(cluster_pair=cp,cluster_2_doc_map=doc_2_cluster_map,df=df)
    
    train, test, val = split_data_mixed(c1s,c2,doc_2_cluster_map,df,c2_train_perc=c2_train_perc)
    if pure:
        train = filter_4_pure(train,cluster=2)
        test = filter_4_pure(test,cluster=2)
        val = filter_4_pure(val,cluster=2)
    
#     train = train[train.doc_indices <= 99402 ]
#     test = test[test.doc_indices <= 99402 ]
#     val = val[val.doc_indices <= 99402 ]
    
#     train = train.loc[train.context_pos_words_size> 0 ]
#     test = test.loc[test.context_pos_words_size> 0 ]
#     val = val.loc[val.context_pos_words_size > 0 ]
    
    metrics_train = {}
    metrics_test = {}
    metrics_val = {}
    losses_train = {}
    model_dict = {}
    # train ssda func
    params = [learning_rates,epochs,batch_sizes,dropouts,weight_decays]
    
    params_all_combo = list(itertools.product(*params))
    
    print("Number of param combinations : %s" %str(len(params_all_combo)))
    
    for p_id,param_list in enumerate(params_all_combo):
        
        print("Param_setting : %s" %str(p_id))
        
        lr = param_list[0]
        epoch = param_list[1]
        batch_size = param_list[2]
        dropout=param_list[3]
        weight_decay = param_list[4]
        
        print("Running model for ----\n lr : %s\n epoch : %s\n batch_size : %s\n dropout : %s\n"%(str(lr),str(epoch),str(batch_size),str(dropout)))
        
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        
        model, epoch_losses, scores_train, scores_test,scores_val  = run_singleTask_exp(train,
                                                                     test,
                                                                     val,
                                                                     lr,
                                                                     epochs=epoch,
                                                                     batch_size=batch_size,
                                                                     dropout = dropout,
                                                                     cuda_device=cuda_device,
                                                                     num_workers=1,
                                                                     with_attention=with_attention,avg_type=avg_type,glove=glove,extreme=extreme,weight_decay=weight_decay)
        
        print_settings((lr,epoch,batch_size,dropout,weight_decay),single=True)
        plot_epoch_loss(epoch_losses)

        metrics_train[(lr,epoch,batch_size,dropout,weight_decay)] = scores_train
        metrics_val[(lr,epoch,batch_size,dropout,weight_decay)] = scores_val
        metrics_test[(lr,epoch,batch_size,dropout,weight_decay)] = scores_test
        losses_train[(lr,epoch,batch_size,dropout,weight_decay)] = epoch_losses
        
        if return_model:
            model_dict[(lr,epoch,batch_size,dropout,weight_decay)] = model
            
    return metrics_train,metrics_test, metrics_val,losses_train,model_dict



def gr_search_multiTask(df,c1s,c2,doc_2_cluster_map,
                    c2_train_perc=0.1,
                   learning_rates=[0.0001,0.001,0.01,0.1],
                   epochs=[3],
                   word_pred_loss_weights=[0.3,0.8],
                   batch_sizes=[8],
                   dropouts=[0.1],
                   l2s=[0.05],
                   cuda_device=torch.device('cuda:1'), 
                   return_model=False,avg_type="binary",glove=False,extreme=False,loss2=False,loss3=False,bad_term_embeds=None):
    """
    Uses a self supervised domain adaptation setting to train the model
    
    * train = 90% c1 data + 10% c2 data
    * test = 50% c1 data + 50% c2 data
    
    * From cluster 1 choose 70% of the data for train, 30% for test
    * From cluster 2 choose 10% of 70% from c1 for train, 
    
    * Loss check
    * Metrics = F1, recall, precision, accuracy, roc
    """
    train, test, val = split_data_mixed(c1s,c2,doc_2_cluster_map,df,c2_train_perc=c2_train_perc)
    
#     train, test, val = split_data(cluster_pair=cp,cluster_2_doc_map=doc_2_cluster_map,df=df)
    
#     train = train[train.doc_indices <= 99402 ]
#     test = test[test.doc_indices <= 99402 ]
#     val = val[val.doc_indices <= 99402 ]
    
#     train = train.loc[train.context_pos_words_size> 0 ]
#     test = test.loc[test.context_pos_words_size> 0 ]
#     val = val.loc[val.context_pos_words_size > 0 ]
    
    bmodel = load_model()
    bmodel.to(torch.device('cuda:1'))
    bmodel.eval()
    tokenizer = load_tokenizer()

    cw_embed_train = get_bert_embeddings(train,bmodel,tokenizer)
    cw_embed_test = get_bert_embeddings(test,bmodel,tokenizer)
    cw_embed_val = get_bert_embeddings(val,bmodel,tokenizer)
    
    
    metrics_train = {}
    metrics_test = {}
    metrics_val = {}
    losses_train = {}
    model_dict = {}
    
    
    
    
    params = [learning_rates,epochs,batch_sizes,dropouts,word_pred_loss_weights,l2s]
    
    if loss3:
        w1s ,w2s = load_loss3_weights()
        params = [learning_rates,epochs,batch_sizes,dropouts,w1s,w2s,l2s]
    
    params_all_combo = list(itertools.product(*params))
    
    if loss3:
        # filter out all w1 and w2 combos that sum up to 1 or more than 1
        filtered_params = []
        for c in params_all_combo:
            if c[4] + c[5] < 1:
                filtered_params.append(c)
        params_all_combo = filtered_params
    
    print("Number of param combinations : %s" %str(len(params_all_combo)))
    
    for p_id,param_list in enumerate(params_all_combo):
        
        w1 = None
        w2= None
        
        print("Param_setting : %s" %str(p_id))
        
        lr = param_list[0]
        epoch = param_list[1]
        batch_size = param_list[2]
        dropout=param_list[3]
        wlw = param_list[4]
        l2 = param_list[5]
        
        if loss3:
            w1 = param_list[4]
            w2 = param_list[5]
        
        print("Running model for ----\n lr : %s\n epoch : %s\n batch_size : %s\n dropout : %s\n"%(str(lr),str(epoch),str(batch_size),str(dropout)))
        
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        

        model, epoch_losses, scores_train, scores_test,scores_val = run_multiTask_exp(train,test,val,cw_embed_train,cw_embed_test,cw_embed_val,
                                                                     lr,wlw,epochs=epoch,batch_size=batch_size,dropout=dropout,cuda_device=cuda_device,num_workers=1,
                                                                                avg_type=avg_type,l2=l2,glove=glove,extreme=extreme,
                                                                                loss2=loss2,loss3=loss3,w1lw=w1,w2lw=w2,bad_term_embeds=bad_term_embeds)
        
        if not loss3:
            print_settings((lr,epoch,batch_size,dropout,wlw,l2),single=False)
            plot_epoch_loss_multi(epoch_losses)

            metrics_train[(lr,epoch,batch_size,dropout,wlw,l2)] = scores_train
            metrics_test[(lr,epoch,batch_size,dropout,wlw,l2)] = scores_test
            metrics_val[(lr,epoch,batch_size,dropout,wlw,l2)] = scores_val
            losses_train[(lr,epoch,batch_size,dropout,wlw,l2)] = epoch_losses
            if return_model:
                model_dict[(lr,epoch,batch_size,dropout,wlw,l2)] = model
        
        if loss3:
            print_settings((lr,epoch,batch_size,dropout,w1,w2,l2),single=False,loss3=True)
            plot_epoch_loss_multi(epoch_losses)

            metrics_train[(lr,epoch,batch_size,dropout,w1,w2,l2)] = scores_train
            metrics_test[(lr,epoch,batch_size,dropout,w1,w2,l2)] = scores_test
            metrics_val[(lr,epoch,batch_size,dropout,w1,w2,l2)] = scores_val
            losses_train[(lr,epoch,batch_size,dropout,w1,w2,l2)] = epoch_losses
            if return_model:
                model_dict[(lr,epoch,batch_size,dropout,w1,w2,l2)] = model
            
            
    return metrics_train,metrics_test,metrics_val, losses_train,model_dict