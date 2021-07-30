import matplotlib.pyplot as plt
from model_metrics import print_res

def init_plt_settings():
    """
    """
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams['axes.grid'] = True


def plot_epoch_loss(epoch_losses):
    """
    """
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(5,5))
    ax.plot(range(len(epoch_losses["total_loss"])),epoch_losses["total_loss"],c="green",label="Training Loss")
    ax.plot(range(len(epoch_losses["total_loss_val"])),epoch_losses["total_loss_val"],c="red",label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    plt.legend()
    plt.show()

def plot_epoch_loss_multi(epoch_losses):
    """
    """
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
    
    axes = ax.ravel()
    
    ax[0].plot(range(len(epoch_losses["total_loss"])),epoch_losses["total_loss"],c="green",alpha=0.4,label="Total Loss")
    ax[0].plot(range(len(epoch_losses["word_loss"])),epoch_losses["word_loss"],c="blue",alpha=0.3,label="Word Loss")
    ax[0].plot(range(len(epoch_losses["rs_loss"])),epoch_losses["rs_loss"],c="red",label="RS Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Training Loss")
    ax[0].set_title("Training Loss for ATMT")
    
    ax[1].plot(range(len(epoch_losses["total_loss_val"])),epoch_losses["total_loss_val"],c="green",alpha=0.4,label="Total Loss")
    ax[1].plot(range(len(epoch_losses["word_loss_val"])),epoch_losses["word_loss_val"],c="blue",alpha=0.3,label="Word Loss")
    ax[1].plot(range(len(epoch_losses["rs_loss_val"])),epoch_losses["rs_loss_val"],c="red",label="RS Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Validation Loss")
    ax[1].set_title("Validation Loss for ATMT")
    plt.legend()
    plt.show()

def plot_clus1_vs_clus2_acc(epoch_scores,ss=100):
    """
    """
    c1_f1s = []
    c2_f1s =[]
    c1_p =[]
    c2_p =[]
    c1_r =[]
    c2_r =[]
    c1_acc =[]
    c2_acc = []
    for param,hyper_setting_df in print_res(epoch_scores).groupby("Settings"):
        c1_score = hyper_setting_df.loc[hyper_setting_df.Score_type == "cluster1"]
        c2_score = hyper_setting_df.loc[hyper_setting_df.Score_type == "cluster2"]
        
        assert c1_score.shape[0] == 1
        assert c2_score.shape[0] == 1
        
        c1_f1s.append(c1_score["F1"].tolist()[0])
        c1_p.append(c1_score["Precision"].tolist()[0])
        c1_r.append(c1_score["Recall"].tolist()[0])
        c1_acc.append(c1_score["Accuracy"].tolist()[0])
        
        c2_f1s.append(c2_score["F1"].tolist()[0])
        c2_p.append(c2_score["Precision"].tolist()[0])
        c2_r.append(c2_score["Recall"].tolist()[0])
        c2_acc.append(c2_score["Accuracy"].tolist()[0])
    
    
    fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(20,15))
    axes = ax.ravel()
    
    axes[0].scatter(c1_f1s,c2_f1s,c="blue",alpha=0.8,s=ss,label="F1 - Scores")
    axes[0].plot([0, 1], [0, 1],c="red",alpha=0.8,linestyle='dashed',linewidth=3, transform=axes[0].transAxes)
    axes[0].set_xlim([0,1])
    axes[0].set_ylim([0,1])
    axes[0].set_xlabel("C1-Score")
    axes[0].set_ylabel("C2-Score")
    axes[0].set_title("F1 Comparison")
    
    axes[1].scatter(c1_p,c2_p,c="blue",alpha=0.8,s=ss,label="Precision")
    axes[1].plot([0, 1], [0, 1],c="red",alpha=0.8,linestyle='dashed',linewidth=3, transform=axes[1].transAxes)
    axes[1].set_xlim([0,1])
    axes[1].set_ylim([0,1])
    axes[1].set_xlabel("C1-Score")
    axes[1].set_ylabel("C2-Score")
    axes[1].set_title("Precision Comparison")
    
    axes[2].scatter(c1_r,c2_r,c="blue",alpha=0.8,s=ss,label="Recall")
    axes[2].plot([0, 1], [0, 1],c="red",alpha=0.8,linestyle='dashed',linewidth=3, transform=axes[2].transAxes)
    axes[2].set_xlim([0,1])
    axes[2].set_ylim([0,1])
    axes[2].set_xlabel("C1-Score")
    axes[2].set_ylabel("C2-Score")
    axes[2].set_title("Recall Comparison")
    
    axes[3].scatter(c1_acc,c2_acc,c="blue",alpha=0.8,s=ss,label="Accuracy")
    axes[3].plot([0, 1], [0, 1],c="red",alpha=0.8, linestyle='dashed',linewidth=3,transform=axes[3].transAxes)
    axes[3].set_xlim([0,1])
    axes[3].set_ylim([0,1])
    axes[3].set_xlabel("C1-Score")
    axes[3].set_ylabel("C2-Score")
    axes[3].set_title("Accuracy Comparison")
    
    fig.suptitle("Pairwise Score Comparison")
    plt.tight_layout()
    plt.show()

init_plt_settings()