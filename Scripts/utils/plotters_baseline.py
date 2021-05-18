import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Scripts.utils.config import RANDOM_SEED
from Scripts.utils.general_utils import timer
import numpy as np
import pandas as pd
import pickle
import re
import seaborn as sns
from collections import defaultdict, Counter



SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['axes.grid'] = True




def plot_helper(x1,x2,y1,y2,ax,ax_index,marker="*",s=50):
    """
    Utility function that plots trend lines for each user type
    """
    ax[ax_index].scatter(x1,y1,marker=marker,color="tab:blue",s=s,label="Heterogeneous User")
    ax[ax_index].scatter(x2,y2,marker=marker,color="tab:red",s=s,label="Homogeneous User")
    
    sns.regplot(x=np.array(x1), y=np.array(y1),ax=ax[ax_index],color="cornflowerblue")
    sns.regplot(x=np.array(x2), y=np.array(y2),ax=ax[ax_index],color="lightcoral")
    
    ax[ax_index].legend(loc="upper right")



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

def plot_size_dist(cluster_sizes):
    """
    Plots the size distribution of all the clusters
    """
    plt.figure(figsize=(20,10))
    plt.bar(cluster_sizes.keys(), cluster_sizes.values(),width=2)
    plt.xlabel("Cluster-Number")
    plt.ylabel("Documents in Cluster")
    
    plt.title("Cluster Size Distribution")
    plt.savefig("Graphs/cluster_size_dist.pdf")
    plt.show()

@timer
def bsl1_sim_vs_score(df_results,threshold):
    """
    Plots cosine similarity vs metric (non-online setting) for Baseline 1
    """
    fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(15,15))
    axes = ax.ravel()
    sim_score = df_results["Cosine Distance"].tolist()
    scores = df_results["threshold - %s"%str(threshold)].tolist()
    f1,precision,recall,accuracy = zip(*scores)
    
    s = 50
    
    axes[0].scatter(sim_score,f1,marker="*",s=s)
    z = np.polyfit(sim_score,f1, 1)
    p = np.poly1d(z)
    axes[0].plot(sim_score,p(sim_score),"r--")
    axes[0].set_xlabel("Cosine Similarity between cluster pairs")
    axes[0].set_ylabel("F1-Score")
    axes[0].set_ylim(0.0,1.0)
    
    axes[1].scatter(sim_score,precision,marker="o",s=s)
    z = np.polyfit(sim_score,precision, 1)
    p = np.poly1d(z)
    axes[1].plot(sim_score,p(sim_score),"r--")
    axes[1].set_xlabel("Cosine Similarity between cluster pairs")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0.0,1.0)
    
    axes[2].scatter(sim_score,recall,marker="^",s=s)
    z = np.polyfit(sim_score,recall, 1)
    p = np.poly1d(z)
    axes[2].plot(sim_score,p(sim_score),"r--")
    axes[2].set_xlabel("Cosine Similarity between cluster pairs")
    axes[2].set_ylabel("Recall")
    axes[2].set_ylim(0.0,1.0)
    
    axes[3].scatter(sim_score,accuracy,marker="+",s=s)
    z = np.polyfit(sim_score,accuracy, 1)
    p = np.poly1d(z)
    axes[3].plot(sim_score,p(sim_score),"r--")
    axes[3].set_xlabel("Cosine Similarity between cluster pairs")
    axes[3].set_ylabel("Accuracy")
    axes[3].set_ylim(0.0,1.0)
    
    fig.suptitle("Cluster Similarity vs Classifier Performance | Threshold : %s" %str(threshold))
    fig.tight_layout()
    fig.savefig("Graphs/baseline1_sim_vs_score.pdf")
    plt.show()

@timer
def bsl1_sim_vs_score_all_users(df_results_hetero,df_results_homo,threshold):
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
    fig.savefig("Graphs/baseline1_sim_vs_model_perf_%s.pdf"%str(int(threshold*10)))
    plt.show()
    
@timer
def plot_regularization_vs_metrics_at_k(scores_,regularization_constants=[0.0001,0.001,0.01,0.1,1.0],user_type="Heterogeneous",single=True,savefolder="Graphs/TFIDF/"):
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
        fig.savefig(savefolder+"regularization_vs_model_performance_single_%s.pdf" %str(user_type))
    else:
        fig.suptitle("Regularization Constant vs Model Performance (Metrics @K) | Cumulative Performance across all Cluster Pairs --> %s" %str(user_type))
        fig.tight_layout()
        fig.savefig(savefolder+"regularization_vs_model_performance_cumu_%s.pdf"%str(user_type))
    
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
def plot_lr_vs_metrics_at_k(scores_,lr=[0.0001,0.001,0.01,0.1,1.0],user_type="Heterogeneous",single=True,mixed_data=False):
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
        if mixed_data:
            fig.savefig("Graphs/lr_vs_model_performance_single_mixed_%s.pdf"%str(user_type))
        else:
            fig.savefig("Graphs/lr_vs_model_performance_single_%s.pdf" %str(user_type))
    else:
        fig.suptitle("Learning Rate vs Model Performance (Metrics @K) | Cumulative Performance across all Cluster Pairs --> %s" %str(user_type))
        fig.tight_layout()
        if mixed_data:
            fig.savefig("Graphs/lr_vs_model_performance_cumu_mixed_%s.pdf"%str(user_type))
        else:
            fig.savefig("Graphs/lr_vs_model_performance_cumu_%s.pdf"%str(user_type))
    
    plt.show()
    
@timer
def plot_lr_vs_metrics_at_k_cumu(scores_,user_type="Heterogeneous",lr=[0.0001,0.001,0.01,0.1,1.0],mixed_data=False):
    """
    """
    cumu_scores = combine_scores_reg_vs_metrics_at_k(scores_)
    plot_lr_vs_metrics_at_k(cumu_scores,
                            lr=lr,
                            user_type=user_type,
                            mixed_data=mixed_data,
                            single=False)

    
# --------
    
@timer
def plot_online_setting_per_clusterpair(scores_homo,scores_hetero,active=True,single=True,savefolder="Graphs/TFIDF/"):
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
            fig.savefig(savefolder+"user_interaction_vs_model_performance_using_val.pdf")
        else:
            fig.suptitle("Online Learning Setting :(Using Validation Set) | Cumulative Performance across all Cluster Pairs")
            fig.tight_layout()
            fig.savefig(savefolder+"user_interaction_vs_model_performance_using_val_cumu.pdf")
    
    if not active:
        if single:
            fig.suptitle("Online Learning Setting: (Metrics @K) | Random Single Cluster Pair")
            fig.tight_layout()
            fig.savefig(savefolder+"user_interaction_vs_model_performance.pdf")
        else:
            fig.suptitle("Online Learning Setting: (Metrics @K) | Cumulative Performance across all Cluster Pairs")
            fig.tight_layout()
            fig.savefig(savefolder+"user_interaction_vs_model_performance_cumu.pdf")
            
    plt.show()

def plot_helper_all_cp(x1,y1,ax,color,cp,marker="*",s=50):
    """
    """
    ax.scatter(x1,y1,marker=marker,color=color,s=s,label=cp)
#     sns.regplot(x=np.array(x1), y=np.array(y1),ax=ax,color=color)

@timer
def plot_all_cp_online(scores_cp,user_type,metric="precision",savefolder="Graphs/TFIDF/"):
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
    fig.savefig(savefolder+"user_interaction_vs_model_performance_precision_all_cps_%s.pdf" %str(user_type))
    plt.show()
    
    pass

@timer
def plot_all_cp_singlecluster(scores_cp,metric="precision",savefolder="Graphs/TFIDF/"):
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
    fig.savefig(savefolder+"user_interaction_vs_model_performance_precision_all_cps_single_cluster.pdf")
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
def plot_single_cluster_performance(scores_homo,scores_hetero,single=True,savefolder="Graphs/TFIDF/"):
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
        fig.savefig(savefolder+"user_interaction_vs_model_performance_single_cluster.pdf")
    else:
        fig.suptitle("Single Cluster Performance: (Metrics @K) | Cumulative Performance across all Cluster Pairs")
        fig.tight_layout()
        fig.savefig(savefolder+"user_interaction_vs_model_performance_cumu_single_cluster.pdf")
    
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
    fig.savefig("Graphs/user_interaction_vs_model_performance_precision_all_cps_mixed_data_%s.pdf"%str(user_type))
    plt.show()

@timer
def plot_helper_md(x1,x2,y1,y2,ax,ax_index,which_cluster_1,which_cluster_2,marker="*",s=50):
    """
    """
    which_cluster_1 = np.array(which_cluster_1)
    which_cluster_2 = np.array(which_cluster_2)
    y1_masked_c1 = np.ma.masked_where(which_cluster_1==2,y1)
    y1_masked_c2 = np.ma.masked_where(which_cluster_1==1,y1)
    y2_masked_c1 = np.ma.masked_where(which_cluster_2==2,y2)
    y2_masked_c2 = np.ma.masked_where(which_cluster_2==1,y2)
    
    # Heterogeneous User
    ax[ax_index].scatter(x1,y1_masked_c1,marker=marker,alpha=0.5,color="orange",s=s,label="Cluster 1 - Heterogeneous User")
    sns.regplot(x=x1, y=y1_masked_c1,ax=ax[ax_index],color="orange")
    ax[ax_index].scatter(x1,y1_masked_c2,marker=marker,alpha=0.5,color="tab:blue",s=s,label="Cluster 2 - Heterogeneous User")
    sns.regplot(x=x1, y=y1_masked_c2,ax=ax[ax_index],color="tab:blue")
    # Heterogeneous User
    ax[ax_index].scatter(x2,y2_masked_c1,marker=marker,alpha=0.5,color="green",s=s,label="Cluster 1 - Homogeneous User")
    ax[ax_index].scatter(x2,y2_masked_c2,marker=marker,alpha=0.5,color="tab:red",s=s,label="Cluster 2 - Homogeneous User")
    
    ax[ax_index].legend(loc="upper right")

def plot_prec_sep(x,y_c1,y_c2,which_cluster,ax,col1="orange",col2="tab:blue",user_type="Heterogenous",marker="*",s=50,mask=True):
    """
    per metric per user 
    plots a metric for a given user type (2 lines, one for cluster 1 and one for cluster 2)
    
    x -> K interactions
    y_c1 -> c1 metric
    y_c2 -> c2 metric
    which_cluster -> cluster the recommended item at kth interaction belongs to
    """
    y_c1_masked = y_c1
    y_c2_masked = y_c2
    
    if mask == True:
        y_c1_masked = np.ma.masked_where(which_cluster==2,y_c1)
        y_c2_masked = np.ma.masked_where(which_cluster==1,y_c2)
    
    ax.scatter(x,y_c1_masked,marker=marker,alpha=0.5,color=col1,s=s,label="%s User - Cluster 1"%str(user_type))
    ax.scatter(x,y_c2_masked,marker=marker,alpha=0.5,color=col2,s=s,label="%s User - Cluster 2"%str(user_type))
    

@timer
def combine_scores_mixed_data(scores_):
    """
    """
    combined_scores_map = defaultdict(lambda : defaultdict(list))
    f1_cumu = []
    f1_c1_cumu = []
    f1_c2_cumu = []
    recall_cumu = []
    recall_c1_cumu = []
    recall_c2_cumu = []
    precision_cumu = []
    precision_c1_cumu = []
    precision_c2_cumu = []
    for cp in scores_:
        f1_cumu.append(scores_[cp]["logistic_regression"]["f1"])
        f1_c1_cumu.append(scores_[cp]["logistic_regression"]["f1_c1"])
        f1_c2_cumu.append(scores_[cp]["logistic_regression"]["f1_c2"])
        
        recall_cumu.append(scores_[cp]["logistic_regression"]["recall"])
        recall_c1_cumu.append(scores_[cp]["logistic_regression"]["recall_c1"])
        recall_c2_cumu.append(scores_[cp]["logistic_regression"]["recall_c2"])
        
        precision_cumu.append(scores_[cp]["logistic_regression"]["precision"])
        precision_c1_cumu.append(scores_[cp]["logistic_regression"]["precision_c1"])
        precision_c2_cumu.append(scores_[cp]["logistic_regression"]["precision_c2"])

    
    # Average over N (articles recommended/columns)
    combined_scores_map["logistic_regression"]["f1"] = np.mean(np.array(f1_cumu),axis=0)
    combined_scores_map["logistic_regression"]["f1_c1"] = np.mean(np.array(f1_c1_cumu),axis=0)
    combined_scores_map["logistic_regression"]["f1_c2"] = np.mean(np.array(f1_c2_cumu),axis=0)
    
    combined_scores_map["logistic_regression"]["recall"] = np.mean(np.array(recall_cumu),axis=0)
    combined_scores_map["logistic_regression"]["recall_c1"] = np.mean(np.array(recall_c1_cumu),axis=0)
    combined_scores_map["logistic_regression"]["recall_c2"] = np.mean(np.array(recall_c2_cumu),axis=0)
    
    combined_scores_map["logistic_regression"]["precision"] = np.mean(np.array(precision_cumu),axis=0)
    combined_scores_map["logistic_regression"]["precision_c1"] = np.mean(np.array(precision_c1_cumu),axis=0)
    combined_scores_map["logistic_regression"]["precision_c2"] = np.mean(np.array(precision_c2_cumu),axis=0)
    
    return combined_scores_map    
    
@timer
def plot_mixed_cluster_cumulative_per_cluster(scores_homo_cumulative,scores_hetero_cumulative):
    """
    """
    scores_homo = combine_scores_mixed_data(scores_homo_cumulative)
    scores_hetero = combine_scores_mixed_data(scores_hetero_cumulative)
    plot_mixed_data_performance_per_cluster(scores_homo,scores_hetero,mask=False)
    
    
@timer
def plot_mixed_data_performance_per_cluster(scores_homo,scores_hetero,mask=True,savefolder="Graphs/TFIDF/"):
    """
    """
    fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(26,15))
    axes = ax.ravel()
    
    # f1
    
    #hetero
    plot_prec_sep(x=[i for i in range(len(scores_hetero["logistic_regression"]["f1"]))],
                  y_c1 = scores_hetero["logistic_regression"]["f1_c1"],
                  y_c2=scores_hetero["logistic_regression"]["f1_c2"],
                  which_cluster=scores_hetero["logistic_regression"]["which_cluster"],
                  ax=axes[0],
                  col1="orange",
                  col2="tab:blue",
                  user_type="Heterogeneous",
                  marker="*",
                  s=50,mask=mask)
    
    plot_prec_sep(x=[i for i in range(len(scores_homo["logistic_regression"]["f1"]))],
              y_c1 = scores_homo["logistic_regression"]["f1_c1"],
              y_c2=scores_homo["logistic_regression"]["f1_c2"],
              which_cluster=scores_homo["logistic_regression"]["which_cluster"],
              ax=axes[0],
              col1="green",
              col2="tab:red",
              user_type="Homogeneous",
              marker="*",
              s=50,mask=mask)
    
    axes[0].legend(loc="upper right",bbox_to_anchor=(1.55, 1))
    axes[0].set_xlabel("N Articles Recommended")
    axes[0].set_ylabel("F1-Score")
    axes[0].set_ylim(0.0,1.0)
    
    #precision
    plot_prec_sep(x=[i for i in range(len(scores_hetero["logistic_regression"]["precision"]))],
                  y_c1 = scores_hetero["logistic_regression"]["precision_c1"],
                  y_c2=scores_hetero["logistic_regression"]["precision_c2"],
                  which_cluster=scores_hetero["logistic_regression"]["which_cluster"],
                  ax=axes[1],
                  col1="orange",
                  col2="tab:blue",
                  user_type="Heterogeneous",
                  marker="o",
                  s=50,mask=mask)
    
    plot_prec_sep(x=[i for i in range(len(scores_homo["logistic_regression"]["precision"]))],
              y_c1 = scores_homo["logistic_regression"]["precision_c1"],
              y_c2=scores_homo["logistic_regression"]["precision_c2"],
              which_cluster=scores_homo["logistic_regression"]["which_cluster"],
              ax=axes[1],
              col1="green",
              col2="tab:red",
              user_type="Homogeneous",
              marker="o",
              s=50,mask=mask)
    
    axes[1].legend(loc="upper right",bbox_to_anchor=(1.55, 1))
    axes[1].set_xlabel("N Articles Recommended")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0.0,1.0)
    
    #recall
    plot_prec_sep(x=[i for i in range(len(scores_hetero["logistic_regression"]["recall"]))],
                  y_c1 = scores_hetero["logistic_regression"]["recall_c1"],
                  y_c2=scores_hetero["logistic_regression"]["recall_c2"],
                  which_cluster=scores_hetero["logistic_regression"]["which_cluster"],
                  ax=axes[2],
                  col1="orange",
                  col2="tab:blue",
                  user_type="Heterogeneous",
                  marker="^",
                  s=50,mask=mask)
    
    plot_prec_sep(x=[i for i in range(len(scores_homo["logistic_regression"]["recall"]))],
              y_c1 = scores_homo["logistic_regression"]["recall_c1"],
              y_c2=scores_homo["logistic_regression"]["recall_c2"],
              which_cluster=scores_homo["logistic_regression"]["which_cluster"],
              ax=axes[2],
              col1="green",
              col2="tab:red",
              user_type="Homogeneous",
              marker="^",
              s=50,mask=mask)
    
    axes[2].legend(loc="upper right",bbox_to_anchor=(1.55, 1))
    axes[2].set_xlabel("N Articles Recommended")
    axes[2].set_ylabel("Recall")
    axes[2].set_ylim(0.0,1.0)
    
    axes[3].axis("off")
    
    
    if mask == True:
        fig.suptitle("Mixed Cluster Performance: (Metrics @K) | Random Single Cluster Pair")
        fig.tight_layout()
        fig.savefig(savefolder+"user_interaction_vs_model_performance_mixed_cluster_single.pdf")
    else:
        fig.suptitle("Mixed Cluster Performance: (Metrics @K) | All Cluster Pairs")
        fig.tight_layout()
        fig.savefig(savefolder+"user_interaction_vs_model_performance_mixed_cluster_all.pdf")
    
    plt.show()
    
    
@timer
def plot_mixed_data_performance(scores_homo,scores_hetero,single=True):
    """
    """
    fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(15,15))
    axes = ax.ravel()
    
    plot_helper_func = None
    if not single:
        plot_helper_func = plot_helper
    else:
        plot_helper_func = partial(plot_helper_md,
                                   which_cluster_1=scores_hetero["logistic_regression"]["which_cluster"],
                                   which_cluster_2=scores_homo["logistic_regression"]["which_cluster"])
    
    plot_helper_func(x1=[i for i in range(len(scores_hetero["logistic_regression"]["f1"]))],
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
    
    plot_helper_func(x1=[i for i in range(len(scores_hetero["logistic_regression"]["precision"]))],
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
    
    plot_helper_func(x1=[i for i in range(len(scores_hetero["logistic_regression"]["recall"]))],
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
        fig.savefig("Graphs/user_interaction_vs_model_performance_mixed_cluster.pdf")
    else:
        fig.suptitle("Mixed Cluster Performance: (Metrics @K) | Cumulative Performance across all Cluster Pairs")
        fig.tight_layout()
        fig.savefig("Graphs/user_interaction_vs_model_performance_cumu_mixed_cluster.pdf")
    
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
    

@timer
def plot_mixed_data_all_cp_perf(scores_cp,user_type,metric="precision",savefolder="Graphs/TFIDF/"):
    """
    """
    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(20,10*2))
    axes = ax.ravel()
    
    norm = matplotlib.colors.Normalize(vmin=1.0, vmax=len(scores_cp.keys()))
    
    cmap_c1 = matplotlib.cm.get_cmap('Spectral')
    cmap_c2 = matplotlib.cm.get_cmap('viridis')
    
    colors_c1 = [cmap_c1(norm(index+1)) for index,_ in enumerate(scores_cp.keys())]
    colors_c2 = [cmap_c2(norm(index+1)) for index,_ in enumerate(scores_cp.keys())]
    
    for i,cp in enumerate(scores_cp.keys()):
        y1 = np.array(scores_cp[cp]['logistic_regression']["%s_c1"%metric])
        y2 = np.array(scores_cp[cp]['logistic_regression']["%s_c2"%metric])
        which_cluster = np.array(scores_cp[cp]['logistic_regression']["which_cluster"])
        y1_c1_masked = np.ma.masked_where(which_cluster==2,y1)
        y1_c2_masked = np.ma.masked_where(which_cluster==1,y2)
        
        plot_helper_all_cp(x1 = [k for k in range(len(scores_cp[cp]['logistic_regression'][metric]))],
                           y1 = y1_c1_masked,
                           ax = axes[0],
                           color = colors_c1[i],
                           cp = cp,
                           marker="o",
                           s=20)
        axes[0].title.set_text('Cluster 1')
        plot_helper_all_cp(x1 = [k for k in range(len(scores_cp[cp]['logistic_regression'][metric]))],
                           y1 = y1_c2_masked,
                           ax = axes[1],
                           color = colors_c2[i],
                           cp = cp,
                           marker="o",
                           s=20)
        axes[1].title.set_text('Cluster 2')
    axes[0].legend(bbox_to_anchor=(1.1, 1.05),ncol=4,handleheight=2.4, labelspacing=0.05,title="Cluster Pairs")
    axes[1].legend(bbox_to_anchor=(1.1, 1.05),ncol=4,handleheight=2.4, labelspacing=0.05,title="Cluster Pairs")
    axes[0].set_xlabel("N Articles Recommended")
    axes[1].set_xlabel("N Articles Recommended")
    axes[0].set_ylabel(metric.upper()+" @K")
    axes[1].set_ylabel(metric.upper()+" @K")
    axes[0].set_ylim(0.0,1.0)
    axes[1].set_ylim(0.0,1.0)
    fig.suptitle("%s | %s -> %s" %(metric.upper(),"Mixed Data Performance",user_type))
    fig.tight_layout()
    fig.savefig(savefolder+"user_interaction_vs_model_performance_precision_all_cps_mixed_data_sep_%s.pdf"%str(user_type))
    plt.show()

def lr_mixed_plot_helper(x,y_c1,y_c2,which_cluster,clr,ax,marker,s=50,lr_score=0.001):
    """
    """
#     x = [i for i in range(len(reg_scores["f1"]))]
    y1_c1_masked = np.ma.masked_where(which_cluster==2,y_c1)
    y1_c2_masked = np.ma.masked_where(which_cluster==1,y_c2)
    
    #c1
    ax[0].scatter(x,y1_c1_masked,marker=marker,color=clr,s=s,label=str(lr_score))
    sns.regplot(x=np.array(x),y=y1_c1_masked,ax=ax[0],color=clr)
    #c2
    ax[1].scatter(x,y1_c2_masked,marker=marker,color=clr,s=s,label=str(lr_score))
    sns.regplot(x=np.array(x),y=y1_c2_masked,ax=ax[1],color=clr)
    
    
@timer
def plot_lr_vs_metrics_at_k_mixed(scores_,lr,user_type="Heterogeneous",savefolder="Graphs/TFIDF/"):
    """
    for precision:
    2 plots side by side , one for cluster1, one for cluster2
    with precision on y
    """
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(17,8))
    axes = ax.ravel()
    
    clrs = sns.color_palette("tab10", n_colors=len(lr))
    markers = [i for i in Line2D.markers][:len(lr)]
    
    for index,l_r in enumerate(scores_.keys()):
        # for each lr setting we plot 2 subplots for each user type
        lr_mixed_plot_helper(x=[i for i in range(len(scores_[l_r]["f1"]))],
                             y_c1=scores_[l_r]["precision_c1"],
                             y_c2=scores_[l_r]["precision_c2"],
                             which_cluster=scores_[l_r]["which_cluster"],
                             clr=clrs[index],
                             ax=axes,
                             marker="o",
                             s=25,
                             lr_score=l_r)
    
    axes[0].set_xlabel("N Articles Recommended")
    axes[0].set_ylabel("Precision")
    axes[0].set_ylim(0.0,1.0)
    axes[0].set_title("Cluster 1 Performance")
    
    axes[1].set_xlabel("N Articles Recommended")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0.0,1.0)
    axes[1].set_title("Cluster 2 Performance")
    
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    
    fig.suptitle("Learning Rate vs Model Performance (Metrics @K) | Performance for single cluster pair --> %s" %str(user_type))
    fig.tight_layout()
    fig.savefig(savefolder+"lr_vs_model_performance_single_mixed_%s.pdf"%str(user_type))
    
    plt.show()

def plot_mixed_data_all_cp_precis_lr(scores_,user_type,lr,metric="precision"):
    """
    """
    lr_settings = len(lr)
    fig,ax = plt.subplots(nrows=lr_settings,ncols=2,figsize=(20,60))
    axes=ax.ravel()
    cmap_c1 = matplotlib.cm.get_cmap('Spectral')
    cmap_c2 = matplotlib.cm.get_cmap('viridis')
    
    norm = matplotlib.colors.Normalize(vmin=1.0, vmax=len(scores_.keys()))
    
    colors_c1 = [cmap_c1(norm(index+1)) for index,_ in enumerate(scores_.keys())]
    colors_c2 = [cmap_c2(norm(index+1)) for index,_ in enumerate(scores_.keys())]
    
    for j,cp in enumerate(scores_.keys()):
        for i,l_r in enumerate(scores_[cp].keys()):
            ax1 = ax[i,0]
            ax2 = ax[i,1]
            y1 = np.array(scores_[cp][l_r]["%s_c1"%metric])
            y2 = np.array(scores_[cp][l_r]["%s_c2"%metric])
            which_cluster = np.array(scores_[cp][l_r]["which_cluster"])
            y1_c1_masked = np.ma.masked_where(which_cluster==2,y1)
            y1_c2_masked = np.ma.masked_where(which_cluster==1,y2)

            plot_helper_all_cp(x1 = [k for k in range(len(scores_[cp][l_r][metric]))],
                               y1 = y1_c1_masked,
                               ax = ax1,
                               color = colors_c1[j],
                               cp = cp,
                               marker="o",
                               s=20)
            ax1.title.set_text('Cluster 1 - %s' %str(l_r))
            plot_helper_all_cp(x1 = [k for k in range(len(scores_[cp][l_r][metric]))],
                               y1 = y1_c2_masked,
                               ax = ax2,
                               color = colors_c2[j],
                               cp = cp,
                               marker="o",
                               s=20)
            ax2.title.set_text('Cluster 2 - %s' %str(l_r))
        
    for ax_ in axes:        
#         ax_.legend(bbox_to_anchor=(1.1, 1.05),ncol=4,handleheight=2.4, labelspacing=0.05,title="Cluster Pairs")
        ax_.set_xlabel("N Articles Recommended")
        ax_.set_ylabel(metric.upper()+" @K")
        ax_.set_ylim(0.0,1.0)
        
    fig.suptitle("%s | %s -> %s" %(metric.upper(),"Mixed Data Performance",user_type))
    fig.tight_layout()
    fig.savefig("Graphs/lr_vs_model_performance_precision_all_cps_mixed_data_sep_%s.pdf"%str(user_type))
    plt.show()