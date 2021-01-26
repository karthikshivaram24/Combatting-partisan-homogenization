import os
from collections import Counter,defaultdict
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pickle
import re
import seaborn as sns
from functools import wraps
from time import time
import itertools
from functools import partial
import functools
import time
import warnings
import copy

from .general_utils import timer
from .settings import CLUSTER_DIST_PATH, COSINE_SIM_METRIC_PATH, REGULARIZATION_PATH, LEARNING_RATE_PATH, SINGLE_CLUSTER_PATH, ALL_CP_PATH, MIXED_DATA_PATH

def plot_size_dist(cluster_sizes):
    """
    """
    plt.figure(figsize=(20,10))
    plt.bar(cluster_sizes.keys(), cluster_sizes.values(),width=2)
    plt.xlabel("Cluster-Number")
    plt.ylabel("Documents in Cluster")
    plt.title("Cluster Size Distribution")
    plt.savefig(CLUSTER_DIST_PATH+"/cluster_dist.pdf")
    plt.show()


def plot_helper(x1,x2,y1,y2,ax,ax_index,marker="*",s=50):
    """
    """
    ax[ax_index].scatter(x1,y1,marker=marker,color="tab:blue",s=s,label="Heterogeneous User")
    ax[ax_index].scatter(x2,y2,marker=marker,color="tab:red",s=s,label="Homogeneous User")
    
    sns.regplot(x=np.array(x1), y=np.array(y1),ax=ax[ax_index],color="cornflowerblue")
    sns.regplot(x=np.array(x2), y=np.array(y2),ax=ax[ax_index],color="lightcoral")
    
    ax[ax_index].legend(loc="upper right")
    
    
@timer
def plot_sim_vs_score_all_users(df_results_hetero,df_results_homo,threshold):
    """
    """
    fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(15,15))
    axes = ax.ravel()
    
    sim_score_hetero = df_results_hetero["Cosine Distance"].tolist()
    scores_hetero = df_results_hetero["threshold - %s"%str(threshold)].tolist()
    f1_hetero,precision_hetero,recall_hetero,accuracy_hetero = zip(*scores_hetero)
    
    sim_score_homo = df_results_homo["Cosine Distance"].tolist()
    scores_homo = df_results_homo["threshold - %s"%str(threshold)].tolist()
    f1_homo,precision_homo,recall_homo,accuracy_homo = zip(*scores_homo)
    
    s = 50
    
    plot_helper(x1=sim_score_hetero,
                x2=sim_score_homo,
                y1=f1_hetero,
                y2=f1_homo,
                ax=axes,
                ax_index=0,
                marker="*",
                s=50)
    axes[0].set_xlabel("Cosine Similarity between cluster pairs")
    axes[0].set_ylabel("F1-Score")
    axes[0].set_ylim(0.0,1.0)
    
    plot_helper(x1=sim_score_hetero,
            x2=sim_score_homo,
            y1=precision_hetero,
            y2=precision_homo,
            ax=axes,
            ax_index=1,
            marker="o",
            s=50)
    axes[1].set_xlabel("Cosine Similarity between cluster pairs")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0.0,1.0)

    plot_helper(x1=sim_score_hetero,
                x2=sim_score_homo,
                y1=recall_hetero,
                y2=recall_homo,
                ax=axes,
                ax_index=2,
                marker="^",
                s=50)
    axes[2].set_xlabel("Cosine Similarity between cluster pairs")
    axes[2].set_ylabel("Recall")
    axes[2].set_ylim(0.0,1.0)

    plot_helper(x1=sim_score_hetero,
                x2=sim_score_homo,
                y1=accuracy_hetero,
                y2=accuracy_homo,
                ax=axes,
                ax_index=3,
                marker="+",
                s=50)
    axes[3].set_xlabel("Cosine Similarity between cluster pairs")
    axes[3].set_ylabel("Accuracy")
    axes[3].set_ylim(0.0,1.0)
    
    
    fig.suptitle("Cluster Similarity vs Classifier Performance | Threshold : %s" %str(threshold))
    fig.tight_layout()
    fig.savefig(COSINE_SIM_METRIC_PATH+"/cluster_sim_vs_model_perf_%s.pdf"%str(int(threshold*10)))
    plt.show()



def regularization_plot_helper(reg_scores,clr,marker,axes,s=50,reg_score=0.0001):
    """
    """
    x = [i for i in range(len(reg_scores["f1"]))]
  
    axes[0].scatter(x,reg_scores["f1"],marker=marker,color=clr,s=s,label=str(reg_score))
    sns.regplot(x=np.array(x),y=reg_scores["f1"],ax=axes[0],color=clr)
    
    axes[1].scatter(x,reg_scores["precision"],marker=marker,color=clr,s=s,label=str(reg_score))
    sns.regplot(x=np.array(x),y=reg_scores["precision"],ax=axes[1],color=clr)
    
    axes[2].scatter(x,reg_scores["recall"],marker=marker,color=clr,s=s,label=str(reg_score))
    sns.regplot(x=np.array(x),y=reg_scores["recall"],ax=axes[2],color=clr)
    
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")
    

@timer
def plot_regularization_vs_metrics_at_k(scores_,regularization_constants=[0.0001,0.001,0.01,0.1,1.0],user_type="Heterogeneous",single=True):
    """
    3 sub plots for each metric:
    In each subplot, k lines (one for every regularization setting used)
    So multiple colors needed for each regularization setting
    """
    fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(15,15))
    axes = ax.ravel()
    
    clrs = sns.color_palette("tab10", n_colors=len(regularization_constants))
    markers = [i for i in Line2D.markers][:len(regularization_constants)]
    
    for index,rc in enumerate(scores_.keys()):
        regularization_plot_helper(reg_scores=scores_[str(rc)],
                                   clr=clrs[index],
                                   marker=markers[index],
                                   axes=axes,
                                   s=25,
                                   reg_score=rc)
        
    axes[0].set_xlabel("N Articles Recommended")
    axes[0].set_ylabel("F1-Score")
    axes[0].set_ylim(0.0,1.0)
    
    axes[1].set_xlabel("N Articles Recommended")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0.0,1.0)
    
    axes[2].set_xlabel("N Articles Recommended")
    axes[2].set_ylabel("Recall")
    axes[2].set_ylim(0.0,1.0)
    
    axes[3].axis("off")
    
    if single:
        fig.suptitle("Regularization Constant vs Model Performance (Metrics @K) | Random Single Cluster Pair --> %s" %str(user_type))
        fig.tight_layout()
        fig.savefig(REGULARIZATION_PATH+"/regularization_vs_model_performance_single_%s.pdf" %str(user_type))
    else:
        fig.suptitle("Regularization Constant vs Model Performance (Metrics @K) | Cumulative Performance across all Cluster Pairs --> %s" %str(user_type))
        fig.tight_layout()
        fig.savefig(REGULARIZATION_PATH+"/regularization_vs_model_performance_cumu_%s.pdf"%str(user_type))
    
    plt.show()


@timer
def combine_scores_reg_vs_metrics_at_k(scores_):
    """
    """
    combined_scores_map = defaultdict(lambda : defaultdict(list))
    f1_cumu = defaultdict(list)
    recall_cumu = defaultdict(list)
    precision_cumu = defaultdict(list)
    for cp in scores_:
        for rc in scores_[cp]:
            f1_cumu[rc].append(scores_[cp][rc]["f1"])
            recall_cumu[rc].append(scores_[cp][rc]["recall"])
            precision_cumu[rc].append(scores_[cp][rc]["precision"])
         
    
    # Average over N (articles recommended/columns)
    for rc in f1_cumu:
        combined_scores_map[rc]["f1"] = np.mean(np.array(f1_cumu[rc]),axis=0)
    
    for rc in recall_cumu:
        combined_scores_map[rc]["recall"] = np.mean(np.array(recall_cumu[rc]),axis=0)

    for rc in precision_cumu:
        combined_scores_map[rc]["precision"] = np.mean(np.array(precision_cumu[rc]),axis=0)
    
    return combined_scores_map

@timer
def plot_regularization_vs_metrics_at_k_cumu(scores_,user_type="Heterogeneous",regularization_constants=[0.0001,0.001,0.01,0.1,1.0]):
    """
    """
    cumu_scores = combine_scores_reg_vs_metrics_at_k(scores_)
    plot_regularization_vs_metrics_at_k(cumu_scores,
                                        regularization_constants=regularization_constants,
                                        user_type=user_type,
                                        single=False)
    
@timer
def plot_lr_vs_metrics_at_k(scores_,lr=[0.0001,0.001,0.01,0.1,1.0],user_type="Heterogeneous",single=True):
    """
    3 sub plots for each metric:
    In each subplot, k lines (one for every regularization setting used)
    So multiple colors needed for each regularization setting
    """
    fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(15,15))
    axes = ax.ravel()
    
    clrs = sns.color_palette("tab10", n_colors=len(lr))
    markers = [i for i in Line2D.markers][:len(lr)]
    
    for index,rc in enumerate(scores_.keys()):
        regularization_plot_helper(reg_scores=scores_[str(rc)],
                                   clr=clrs[index],
                                   marker=markers[index],
                                   axes=axes,
                                   s=25,
                                   reg_score=rc)
        
    axes[0].set_xlabel("N Articles Recommended")
    axes[0].set_ylabel("F1-Score")
    axes[0].set_ylim(0.0,1.0)
    
    axes[1].set_xlabel("N Articles Recommended")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0.0,1.0)
    
    axes[2].set_xlabel("N Articles Recommended")
    axes[2].set_ylabel("Recall")
    axes[2].set_ylim(0.0,1.0)
    
    axes[3].axis("off")
    
    if single:
        fig.suptitle("Learning Rate vs Model Performance (Metrics @K) | Random Single Cluster Pair --> %s" %str(user_type))
        fig.tight_layout()
        fig.savefig(LEARNING_RATE_PATH+"/lr_vs_model_performance_single_%s.pdf" %str(user_type))
    else:
        fig.suptitle("Learning Rate vs Model Performance (Metrics @K) | Cumulative Performance across all Cluster Pairs --> %s" %str(user_type))
        fig.tight_layout()
        fig.savefig(LEARNING_RATE_PATH+"/lr_vs_model_performance_cumu_%s.pdf"%str(user_type))
    
    plt.show()
    
@timer
def plot_lr_vs_metrics_at_k_cumu(scores_,user_type="Heterogeneous",lr=[0.0001,0.001,0.01,0.1,1.0]):
    """
    """
    cumu_scores = combine_scores_reg_vs_metrics_at_k(scores_)
    plot_lr_vs_metrics_at_k(cumu_scores,
                            lr=lr,
                            user_type=user_type,
                            single=False)

@timer
def plot_online_setting_per_clusterpair(scores_homo,scores_hetero,active=True,single=True):
    """
    We have to plot:
    1) N vs score for both users
    2) Cluster similarity vs score for both users
    """
    figsize = None
    if active:
        figsize = (15,15)
    
    if not active:
        figsize = (15,15)
        
    fig,ax = plt.subplots(nrows=2,ncols=2,figsize=figsize)
    axes = ax.ravel()
    
    plot_helper(x1=[i for i in range(len(scores_hetero["logistic_regression"]["f1"]))],
                x2=[i for i in range(len(scores_hetero["logistic_regression"]["f1"]))],
                y1=scores_hetero["logistic_regression"]["f1"],
                y2=scores_homo["logistic_regression"]["f1"],
                ax=axes,
                ax_index=0,
                marker="*",
                s=50)
    axes[0].set_xlabel("N Articles Recommended")
    axes[0].set_ylabel("F1-Score")
    axes[0].set_ylim(0.0,1.0)
    
    plot_helper(x1=[i for i in range(len(scores_hetero["logistic_regression"]["precision"]))],
                x2=[i for i in range(len(scores_hetero["logistic_regression"]["precision"]))],
                y1=scores_hetero["logistic_regression"]["precision"],
                y2=scores_homo["logistic_regression"]["precision"],
                ax=axes,
                ax_index=1,
                marker="o",
                s=50)
    axes[1].set_xlabel("N Articles Recommended")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0.0,1.0)
    
    plot_helper(x1=[i for i in range(len(scores_hetero["logistic_regression"]["recall"]))],
                x2=[i for i in range(len(scores_hetero["logistic_regression"]["recall"]))],
                y1=scores_hetero["logistic_regression"]["recall"],
                y2=scores_homo["logistic_regression"]["recall"],
                ax=axes,
                ax_index=2,
                marker="^",
                s=50)
    axes[2].set_xlabel("N Articles Recommended")
    axes[2].set_ylabel("Recall")
    axes[2].set_ylim(0.0,1.0)
    
    if active:
        plot_helper(x1=[i for i in range(len(scores_hetero["logistic_regression"]["accuracy"]))],
                    x2=[i for i in range(len(scores_hetero["logistic_regression"]["accuracy"]))],
                    y1=scores_hetero["logistic_regression"]["accuracy"],
                    y2=scores_homo["logistic_regression"]["accuracy"],
                    ax=axes,
                    ax_index=3,
                    marker="+",
                    s=50)
        axes[3].set_xlabel("N Articles Recommended")
        axes[3].set_ylabel("Accuracy")
        axes[3].set_ylim(0.0,1.0)
    
    if not active:
        axes[3].axis("off")
    
    if active:
        if single:
            fig.suptitle("Online Learning Setting :(Using Validation Set) | Random Single Cluster Pair")
            fig.tight_layout()
            fig.savefig(SINGLE_CLUSTER_PATH+"/user_interaction_vs_model_performance_using_val.pdf")
        else:
            fig.suptitle("Online Learning Setting :(Using Validation Set) | Cumulative Performance across all Cluster Pairs")
            fig.tight_layout()
            fig.savefig(SINGLE_CLUSTER_PATH+"/user_interaction_vs_model_performance_using_val_cumu.pdf")
    
    if not active:
        if single:
            fig.suptitle("Online Learning Setting: (Metrics @K) | Random Single Cluster Pair")
            fig.tight_layout()
            fig.savefig(SINGLE_CLUSTER_PATH+"/user_interaction_vs_model_performance.pdf")
        else:
            fig.suptitle("Online Learning Setting: (Metrics @K) | Cumulative Performance across all Cluster Pairs")
            fig.tight_layout()
            fig.savefig(SINGLE_CLUSTER_PATH+"/user_interaction_vs_model_performance_cumu.pdf")
            
    plt.show()

def plot_helper_all_cp(x1,y1,ax,color,cp,marker="*",s=50):
    """
    """
    ax.scatter(x1,y1,marker=marker,color=color,s=s,label=cp)
#     sns.regplot(x=np.array(x1), y=np.array(y1),ax=ax,color=color)

@timer
def plot_all_cp_online(scores_cp,user_type,metric="precision"):
    """
    """
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(20,10))
    axes = ax
    
    norm = matplotlib.colors.Normalize(vmin=1.0, vmax=len(scores_cp.keys()))
    
    cmap = matplotlib.cm.get_cmap('Spectral')
    
    colors = [cmap(norm(index+1)) for index,_ in enumerate(scores_cp.keys())]
    
    for i,cp in enumerate(scores_cp.keys()):
        plot_helper_all_cp(x1 = [k for k in range(len(scores_cp[cp]['logistic_regression'][metric]))],
                           y1 = scores_cp[cp]['logistic_regression'][metric],
                           ax = ax,
                           color = colors[i],
                           cp = cp,
                           marker="o",
                           s=20)
    
    axes.legend(bbox_to_anchor=(1.1, 1.05),ncol=4,handleheight=2.4, labelspacing=0.05,title="Cluster Pairs")
    axes.set_xlabel("N Articles Recommended")
    axes.set_ylabel(metric.upper()+" @K")
    axes.set_ylim(0.0,1.0)
    fig.suptitle("%s | %s" %(metric.upper(),user_type))
    fig.tight_layout()
    fig.savefig(ALL_CP_PATH+"/user_interaction_vs_model_performance_precision_all_cps_%s.pdf" %str(user_type))
    plt.show()
    
    pass

@timer
def plot_all_cp_singlecluster(scores_cp,metric="precision"):
    """
    """
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(20,10))
    axes = ax
    
    norm = matplotlib.colors.Normalize(vmin=1.0, vmax=len(scores_cp.keys()))
    
    cmap = matplotlib.cm.get_cmap('Spectral')
    
    colors = [cmap(norm(index+1)) for index,_ in enumerate(scores_cp.keys())]
    
    for i,cp in enumerate(scores_cp.keys()):
        plot_helper_all_cp(x1 = [k for k in range(len(scores_cp[cp]['logistic_regression'][metric]))],
                           y1 = scores_cp[cp]['logistic_regression'][metric],
                           ax = ax,
                           color = colors[i],
                           cp = cp,
                           marker="o",
                           s=20)
    
    axes.legend(bbox_to_anchor=(1.1, 1.05),ncol=4,handleheight=2.4, labelspacing=0.05,title="Cluster Pairs")
    axes.set_xlabel("N Articles Recommended")
    axes.set_ylabel(metric.upper()+" @K")
    axes.set_ylim(0.0,1.0)
    fig.suptitle("%s | %s" %(metric.upper(),"Single Cluster Performance"))
    fig.tight_layout()
    fig.savefig(ALL_CP_PATH+"/user_interaction_vs_model_performance_precision_all_cps_single_cluster.pdf")
    plt.show()
    

def plot_helper_scp(x1,x2,y1,y2,ax,ax_index,marker="*",s=50):
    """
    """
    ax[ax_index].scatter(x1,y1,marker=marker,color="tab:blue",s=s,label="Liberal")
    ax[ax_index].scatter(x2,y2,marker=marker,color="tab:red",s=s,label="Conservative")
    
    sns.regplot(x=np.array(x1), y=np.array(y1),ax=ax[ax_index],color="cornflowerblue")
    sns.regplot(x=np.array(x2), y=np.array(y2),ax=ax[ax_index],color="lightcoral")
    
    ax[ax_index].legend(loc="upper right")
    
@timer
def plot_single_cluster_performance(scores_homo,scores_hetero,single=True):
    """
    """
    fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(15,15))
    axes = ax.ravel()
    
    plot_helper_scp(x1=[i for i in range(len(scores_hetero["logistic_regression"]["f1"]))],
                x2=[i for i in range(len(scores_homo["logistic_regression"]["f1"]))],
                y1=scores_hetero["logistic_regression"]["f1"],
                y2=scores_homo["logistic_regression"]["f1"],
                ax=axes,
                ax_index=0,
                marker="*",
                s=50)
    axes[0].set_xlabel("N Articles Recommended")
    axes[0].set_ylabel("F1-Score")
    axes[0].set_ylim(0.0,1.0)
    
    plot_helper_scp(x1=[i for i in range(len(scores_hetero["logistic_regression"]["precision"]))],
                x2=[i for i in range(len(scores_homo["logistic_regression"]["precision"]))],
                y1=scores_hetero["logistic_regression"]["precision"],
                y2=scores_homo["logistic_regression"]["precision"],
                ax=axes,
                ax_index=1,
                marker="o",
                s=50)
    axes[1].set_xlabel("N Articles Recommended")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0.0,1.0)
    
    plot_helper_scp(x1=[i for i in range(len(scores_hetero["logistic_regression"]["recall"]))],
                x2=[i for i in range(len(scores_homo["logistic_regression"]["recall"]))],
                y1=scores_hetero["logistic_regression"]["recall"],
                y2=scores_homo["logistic_regression"]["recall"],
                ax=axes,
                ax_index=2,
                marker="^",
                s=50)
    axes[2].set_xlabel("N Articles Recommended")
    axes[2].set_ylabel("Recall")
    axes[2].set_ylim(0.0,1.0)
    
    axes[3].axis("off")
    
    if single:
        fig.suptitle("Single Cluster Performance: (Metrics @K) | Random Single Cluster Pair")
        fig.tight_layout()
        fig.savefig(SINGLE_CLUSTER_PATH+"/user_interaction_vs_model_performance_single_cluster.pdf")
    else:
        fig.suptitle("Single Cluster Performance: (Metrics @K) | Cumulative Performance across all Cluster Pairs")
        fig.tight_layout()
        fig.savefig(SINGLE_CLUSTER_PATH+"/user_interaction_vs_model_performance_cumu_single_cluster.pdf")
    
    plt.show()

@timer
def plot_mixed_data_all_cp_perf(scores_cp,user_type,metric="precision"):
    """
    """
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(20,10))
    axes = ax
    
    norm = matplotlib.colors.Normalize(vmin=1.0, vmax=len(scores_cp.keys()))
    
    cmap = matplotlib.cm.get_cmap('Spectral')
#     cmap = matplotlib.cm.get_cmap('viridis')
    
    colors = [cmap(norm(index+1)) for index,_ in enumerate(scores_cp.keys())]
    
    for i,cp in enumerate(scores_cp.keys()):
        plot_helper_all_cp(x1 = [k for k in range(len(scores_cp[cp]['logistic_regression'][metric]))],
                           y1 = scores_cp[cp]['logistic_regression'][metric],
                           ax = ax,
                           color = colors[i],
                           cp = cp,
                           marker="o",
                           s=20)
    
    axes.legend(bbox_to_anchor=(1.1, 1.05),ncol=4,handleheight=2.4, labelspacing=0.05,title="Cluster Pairs")
    axes.set_xlabel("N Articles Recommended")
    axes.set_ylabel(metric.upper()+" @K")
    axes.set_ylim(0.0,1.0)
    fig.suptitle("%s | %s -> %s" %(metric.upper(),"Mixed Data Performance",user_type))
    fig.tight_layout()
    fig.savefig(MIXED_DATA_PATH+"/user_interaction_vs_model_performance_precision_all_cps_mixed_data_%s.pdf"%str(user_type))
    plt.show()
    
@timer
def plot_mixed_data_performance(scores_homo,scores_hetero,single=True):
    """
    """
    fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(15,15))
    axes = ax.ravel()
    
    plot_helper(x1=[i for i in range(len(scores_hetero["logistic_regression"]["f1"]))],
                x2=[i for i in range(len(scores_hetero["logistic_regression"]["f1"]))],
                y1=scores_hetero["logistic_regression"]["f1"],
                y2=scores_homo["logistic_regression"]["f1"],
                ax=axes,
                ax_index=0,
                marker="*",
                s=50)
    axes[0].set_xlabel("N Articles Recommended")
    axes[0].set_ylabel("F1-Score")
    axes[0].set_ylim(0.0,1.0)
    
    plot_helper(x1=[i for i in range(len(scores_hetero["logistic_regression"]["precision"]))],
                x2=[i for i in range(len(scores_hetero["logistic_regression"]["precision"]))],
                y1=scores_hetero["logistic_regression"]["precision"],
                y2=scores_homo["logistic_regression"]["precision"],
                ax=axes,
                ax_index=1,
                marker="o",
                s=50)
    axes[1].set_xlabel("N Articles Recommended")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0.0,1.0)
    
    plot_helper(x1=[i for i in range(len(scores_hetero["logistic_regression"]["recall"]))],
                x2=[i for i in range(len(scores_hetero["logistic_regression"]["recall"]))],
                y1=scores_hetero["logistic_regression"]["recall"],
                y2=scores_homo["logistic_regression"]["recall"],
                ax=axes,
                ax_index=2,
                marker="^",
                s=50)
    axes[2].set_xlabel("N Articles Recommended")
    axes[2].set_ylabel("Recall")
    axes[2].set_ylim(0.0,1.0)
    
    axes[3].axis("off")
    
    if single:
        fig.suptitle("Mixed Cluster Performance: (Metrics @K) | Random Single Cluster Pair")
        fig.tight_layout()
        fig.savefig(MIXED_DATA_PATH+"/user_interaction_vs_model_performance_mixed_cluster.pdf")
    else:
        fig.suptitle("Mixed Cluster Performance: (Metrics @K) | Cumulative Performance across all Cluster Pairs")
        fig.tight_layout()
        fig.savefig(MIXED_DATA_PATH+"/user_interaction_vs_model_performance_cumu_mixed_cluster.pdf")
    
    plt.show()
    
@timer
def combine_scores_cumu_online_setting(scores_,active=True):
    """
    """
    combined_scores_map = defaultdict(lambda : defaultdict(list))
    f1_cumu = []
    recall_cumu = []
    precision_cumu = []
    accuracy_cumu = []
    for cp in scores_:
        f1_cumu.append(scores_[cp]["logistic_regression"]["f1"])
        recall_cumu.append(scores_[cp]["logistic_regression"]["recall"])
        precision_cumu.append(scores_[cp]["logistic_regression"]["precision"])
        if active:
            accuracy_cumu.append(scores_[cp]["logistic_regression"]["accuracy"])
    
    # Average over N (articles recommended/columns)
    combined_scores_map["logistic_regression"]["f1"] = np.mean(np.array(f1_cumu),axis=0)
    combined_scores_map["logistic_regression"]["recall"] = np.mean(np.array(recall_cumu),axis=0)
    combined_scores_map["logistic_regression"]["precision"] = np.mean(np.array(precision_cumu),axis=0)
    if active:
        combined_scores_map["logistic_regression"]["accuracy"] = np.mean(np.array(accuracy_cumu),axis=0)
    
    return combined_scores_map
        
@timer
def plot_mixed_cluster_cumulative(scores_homo_cumulative,scores_hetero_cumulative):
    """
    """
    scores_homo = combine_scores_cumu_online_setting(scores_homo_cumulative,active=False)
    scores_hetero = combine_scores_cumu_online_setting(scores_hetero_cumulative,active=False)
    plot_mixed_data_performance(scores_homo,scores_hetero,single=False)
    
    
@timer
def plot_online_learning_cumulative(scores_homo_cumulative,scores_hetero_cumulative,active=True):
    """
    """
    scores_homo = combine_scores_cumu_online_setting(scores_homo_cumulative,active=active)
    scores_hetero = combine_scores_cumu_online_setting(scores_hetero_cumulative,active=active)
    plot_online_setting_per_clusterpair(scores_homo,scores_hetero,active=active,single=False)

@timer
def plot_single_cluster_cumulative(scores_homo_cumulative,scores_hetero_cumulative):
    """
    """
    scores_homo = combine_scores_cumu_online_setting(scores_homo_cumulative,active=False)
    scores_hetero = combine_scores_cumu_online_setting(scores_hetero_cumulative,active=False)
    plot_single_cluster_performance(scores_homo,scores_hetero,single=False)
    
    