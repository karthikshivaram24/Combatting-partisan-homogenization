
import torch
from torch_grid_search_runners import gr_search_singleTask, gr_search_multiTask
from metric_aggregators import select_top_settings
from model_metrics import print_res
from metric_aggregators import agg_multiscores, agg_multiscores_dual_loss, avg_scores

def run_4_cps_diagnostics(cps,
              df,
              doc_2_cluster_map,
              c2_train_perc=0.1,
              learning_rates=[0.001],
              epochs=[20,30,50],
              word_pred_loss_weights=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
              batch_sizes=[16],
              dropouts=[0.3],
              l2s=[0.05],
              weight_decays=[0.1,0.01,0.001,0.0001],
              cuda_device=torch.device('cuda:1'), 
              return_model=False,
              avg_type="binary",
              glove=False,
              extreme=False,
              loss2=False,
              loss3=False,
              bad_term_embeds=None,mode=5,pure=False):
    """
    """
    
    best_scores_single_wo_att = []
    best_scores_single_w_att = []
    best_scores_multi_loss1 = []
    best_scores_multi_loss2 = []
    best_scores_multi_loss3 = []

    run_1,run_2, run_3, run_4, run_5, run_6 = False,False,False,False,False, False
    
    if mode == 1:
        run_1 = True
    elif mode == 2:
        run_2 = True
    elif mode ==3:
        run_3 = True
    elif mode == 4:
        run_4 = True
    elif mode == 5:
        run_5 = True
    elif mode == 6:
        run_6 = True
    
    print("Mode Chosen : %s" %str(mode))
    
    for cp_id, cp in enumerate(cps):
        
        print("\n")
        print("\n################################### %s.CP -- %s ################################################\n" %(str(cp_id),str(cp)))
        
        if run_1 or run_5:

            print("\n ****** 1. Single Task without Attention ******** \n")

            metrics_train,metrics_test, metrics_val,losses_train,model_dict = gr_search_singleTask(df=df,
                                                                                             c1s=cp[0],
                                                                                             c2=cp[1],
                                                                                             doc_2_cluster_map=doc_2_cluster_map,
                                                                                             c2_train_perc=c2_train_perc,
                                                                                             learning_rates=learning_rates,
                                                                                             epochs=epochs,
                                                                                             batch_sizes=batch_sizes,
                                                                                             dropouts=dropouts,
                                                                                             with_attention=False,
                                                                                             cuda_device=cuda_device, 
                                                                                             return_model=False,avg_type="binary",glove=glove,extreme=extreme,weight_decays=weight_decays,pure=pure)

            best_val_setting = select_top_settings(metrics_val,metric_type="F1")[1]

            test_res_df = print_res(metrics_test)
            best_test_scores = test_res_df.loc[test_res_df.Settings == best_val_setting].reset_index(drop=True)
            best_scores_single_wo_att.append((cp,best_test_scores))

            print("\n----------------------------------- Best Setting : %s ---------------------------------------"%str(best_val_setting))
        
        if run_2 or run_5:
        
            print("\n ****** 2. Single Task with Attention ******** \n")

            metrics_train,metrics_test, metrics_val,losses_train,model_dict = gr_search_singleTask(df=df,
                                                                                     c1s=cp[0],
                                                                                     c2=cp[1],
                                                                                     doc_2_cluster_map=doc_2_cluster_map,
                                                                                     c2_train_perc=c2_train_perc,
                                                                                     learning_rates=learning_rates,
                                                                                     epochs=epochs,# This is done from observing loss curves
                                                                                     batch_sizes=batch_sizes,
                                                                                     dropouts=dropouts,
                                                                                     with_attention=True,
                                                                                     cuda_device=cuda_device, 
                                                                                     return_model=False,avg_type="binary",glove=glove,extreme=extreme,weight_decays=weight_decays,pure=pure)

            best_val_setting = select_top_settings(metrics_val,metric_type="F1")[1]
            test_res_df = print_res(metrics_test)
            best_test_scores = test_res_df.loc[test_res_df.Settings == best_val_setting].reset_index(drop=True)
            best_scores_single_w_att.append((cp,best_test_scores))

            print("\n----------------------------------- Best Setting : %s ---------------------------------------"%str(best_val_setting))
        
        if run_3 or run_5:
        
            print("\n ****** 3. Multitask Network - Loss 1 ******** \n")

            metrics_train,metrics_test, metrics_val,losses_train,model_dict  = gr_search_multiTask(df=df,
                                                                                              c1s=cp[0],
                                                                                              c2=cp[1],
                                                                                              doc_2_cluster_map=doc_2_cluster_map,
                                                                                                c2_train_perc=c2_train_perc,
                                                                                               learning_rates=learning_rates,
                                                                                               epochs=epochs,
                                                                                               word_pred_loss_weights=word_pred_loss_weights,
                                                                                               batch_sizes=batch_sizes,
                                                                                               dropouts=dropouts,
                                                                                               l2s=l2s,
                                                                                               cuda_device=torch.device('cuda:1'), 
                                                                                               return_model=False,
                                                                                              avg_type="binary",
                                                                                              glove=glove,
                                                                                              extreme=extreme,
                                                                                              loss2=False,
                                                                                              bad_term_embeds=bad_term_embeds)

            best_val_setting = select_top_settings(metrics_val,metric_type="F1")[1]

            best_setting_withlossweights = []

            for loss_weight in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:

                temp = list(best_val_setting)
                temp[4] = loss_weight
                best_setting_withlossweights.append(tuple(temp))

            test_res_df = print_res(metrics_test)
            best_test_scores = test_res_df.loc[test_res_df.Settings.isin(best_setting_withlossweights)].reset_index(drop=True)
            best_scores_multi_loss1.append((cp,best_test_scores))
            
            print("----------------------------------- Best Setting : %s ---------------------------------------"%str(best_val_setting))
        
        if run_4 or run_5:
        

            print("\n ****** 4. Multitask Network - Loss 2******** \n")

            metrics_train,metrics_test, metrics_val,losses_train,model_dict  = gr_search_multiTask(df=df,
                                                                                              c1s=cp[0],
                                                                                              c2=cp[1],
                                                                                              doc_2_cluster_map=doc_2_cluster_map,
                                                                                                c2_train_perc=c2_train_perc,
                                                                                               learning_rates=learning_rates,
                                                                                               epochs=epochs,
                                                                                               word_pred_loss_weights=word_pred_loss_weights,
                                                                                               batch_sizes=batch_sizes,
                                                                                               dropouts=dropouts,
                                                                                               l2s=l2s,
                                                                                               cuda_device=torch.device('cuda:1'), 
                                                                                               return_model=False,
                                                                                              avg_type="binary",
                                                                                              glove=glove,
                                                                                              extreme=extreme,
                                                                                              loss2=True,
                                                                                              bad_term_embeds=bad_term_embeds)

            best_val_setting = select_top_settings(metrics_val,metric_type="F1")[1]

            best_setting_withlossweights = []

            for loss_weight in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:

                temp = list(best_val_setting)
                temp[4] = loss_weight
                best_setting_withlossweights.append(tuple(temp))

            test_res_df = print_res(metrics_test)
            best_test_scores = test_res_df.loc[test_res_df.Settings.isin(best_setting_withlossweights)].reset_index(drop=True)
            best_scores_multi_loss2.append((cp,best_test_scores))
            
            print("----------------------------------- Best Setting : %s ---------------------------------------"%str(best_val_setting))
        
        if run_6 or run_5:

            print("\n ****** 4. Multitask Network - Loss 3******** \n")

            metrics_train,metrics_test, metrics_val,losses_train,model_dict  = gr_search_multiTask(df=df,
                                                                                              c1s=cp[0],
                                                                                              c2=cp[1],
                                                                                              doc_2_cluster_map=doc_2_cluster_map,
                                                                                                c2_train_perc=c2_train_perc,
                                                                                               learning_rates=learning_rates,
                                                                                               epochs=epochs,
                                                                                               word_pred_loss_weights=word_pred_loss_weights,
                                                                                               batch_sizes=batch_sizes,
                                                                                               dropouts=dropouts,
                                                                                               l2s=l2s,
                                                                                               cuda_device=torch.device('cuda:1'), 
                                                                                               return_model=False,
                                                                                              avg_type="binary",
                                                                                              glove=glove,
                                                                                              extreme=extreme,
                                                                                              loss2=False,loss3=True,
                                                                                              bad_term_embeds=bad_term_embeds)

            best_val_setting = select_top_settings(metrics_val,metric_type="F1")[1]

            test_res_df = print_res(metrics_test)
            
            test_res_df["selected"] = test_res_df["Settings"].apply(lambda x: x[:4] == best_val_setting[:4])
            
            best_test_scores = test_res_df.loc[test_res_df.selected == True].reset_index(drop=True)
            best_scores_multi_loss3.append((cp,best_test_scores))
            
            print("----------------------------------- Best Setting : %s ---------------------------------------"%str(best_val_setting))
        
        
        print("\n")
        print("\n")
    
    avg_single_wo_att = None
    avg_single_w_att = None
    avg_multi_loss1 = None
    avg_multi_loss2 = None
    avg_multi_loss3 = None
    
    if run_1 or run_5:
        avg_single_wo_att = avg_scores(best_scores_single_wo_att)
    if run_2 or run_5:
        avg_single_w_att = avg_scores(best_scores_single_w_att)
    if run_3 or run_5:
        avg_multi_loss1 = agg_multiscores(best_scores_multi_loss1)
    if run_4 or run_5:
        avg_multi_loss2 = agg_multiscores(best_scores_multi_loss2)
    
    if run_6 or run_5:
        avg_multi_loss3 = agg_multiscores_dual_loss(best_scores_multi_loss3)

    return avg_single_wo_att, avg_single_w_att,avg_multi_loss1, avg_multi_loss2,avg_multi_loss3


if __name__ == "__main__":
    pass