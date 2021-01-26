
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
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn import metrics
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV,SGDClassifier,PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import joblib
from joblib import Parallel, delayed
from functools import wraps
from time import time
import itertools
from functools import partial
import functools
import time
import warnings
import copy

from model_utils import create_train_test, get_scores, get_scores_wot
from settings import RANDOM_SEED




@timer
def run_model(x_train,x_test,y_train,y_test,seed=RANDOM_SEED):
    """
    """
    clf = LogisticRegressionCV(cv=5,random_state=seed,max_iter=1000,n_jobs=-1,class_weight="balanced").fit(x_train, y_train)
    predicted_probas = clf.predict_proba(x_test)
    print(predicted_probas.shape)
    return clf,predicted_probas

@timer
def run_train_all(X,cluster_2_doc_map,df,cluster_pairs,cosine_mat,thresholds = [0.5,0.7,0.9],user_type="Heterogeneous"):
    """
    """
    results = defaultdict(list)
    for index,cp in enumerate(cluster_pairs):
            print("Training model for cluster pair : %s" %str(index))
            x_train,x_test,y_train,y_test = create_train_test(cluster_pair=cp,
                                                              cluster2doc=cluster_2_doc_map,
                                                              X_feats=X,
                                                              df=df,
                                                              user_type=user_type)

            clf,predicted_probas = run_model(x_train,x_test,y_train,y_test)

            for t in thresholds:
                f1,precision,recall,accuracy = get_scores(y_test,
                                                          predictions=predicted_probas,
                                                          threshold=t)
                results[t].append((f1,precision,recall,accuracy))
    
    
    
    df_results = pd.DataFrame(cluster_pairs,columns=["Cluster1","Cluster2"])
    sim_score = [cosine_mat[cp[0],cp[1]] for cp in cluster_pairs]
    df_results["Cosine Distance"] = sim_score
    for k in results:
        df_results["threshold - %s"%str(k)] = results[k]
    
    return df_results

@timer
def run_online_setting_active(X,cluster_2_doc_map,df,cluster_pairs,cosine_mat,user_type="Heterogeneous"):
    """
    """
    cp_scores_map = {}
    
    for index,cp in enumerate(cluster_pairs):
        print("Training model for cluster pair : %s" %str(index))
        x_train,x_test,y_train,y_test = create_train_test(cluster_pair=cp,
                                                          cluster2doc=cluster_2_doc_map,
                                                          X_feats=X,
                                                          df=df,
                                                          user_type=user_type)
        
        
        # Initial Training on Cluster 1
        estimators = [SGDClassifier()]
        all_param_grids = {0:{"loss":["log"],
                              "penalty":["l1","l2"],
                              "alpha":[0.0001,0.001,0.01],
                              "random_state":[RANDOM_SEED],
                              "n_jobs":[-1]}}
        best_models = []
        for index_est,estimator in enumerate(estimators):
            gcv = GridSearchCV(estimator,all_param_grids[index_est],scoring="f1_macro",n_jobs=-1)
            gcv.fit(x_train,y_train)
            best_model = gcv.best_estimator_
            best_models.append(best_model)
        
        # Online Training on Cluster 2 with updates(partial_fit)
        # for each estimator run 
        scores_map = defaultdict(lambda : defaultdict(list))
        models=["logistic_regression"]
        for index_bm,clf in enumerate(best_models):
            model = models[index_bm]
            N = 100
            
            # Lets try stratified split
            x_test_sub, x_test_val, y_test_sub, y_test_val = train_test_split(x_test,
                                                                              y_test,
                                                                              train_size=N,
                                                                              random_state=RANDOM_SEED,
                                                                              shuffle=True,
                                                                              stratify=y_test)
            
            print("Cluster 2: Data Distribution (Candidate Pool and Validation Set) ")
            u_train,c_train = np.unique(y_test_sub, return_counts=True)
            u_test,c_test = np.unique(y_test_val, return_counts=True)
            print("Candidate Pool Label Dist :\n %s : %s\n %s:%s" %(str(u_train[0]),str(c_train[0]),str(u_train[1]),str(c_train[1])))
            print("Validation Label Dist :\n %s : %s\n %s:%s" %(str(u_test[0]),str(c_test[0]),str(u_test[1]),str(c_test[1])))
            
            y_true_sub = []
            y_preds = []
            
            candidate_pool_x = copy.deepcopy(x_test_sub)
            candidate_pool_y = copy.deepcopy(y_test_sub)
            
            for i in range(N):
                # recommend article to user
                probas = clf.predict_proba(candidate_pool_x)[:,1]
                rank_indices = np.argsort(probas)
                top_index = rank_indices[-1]
                
                # user likes/dislikes
                # Heterogeneous user's will like liberal articles, while Homogeneous users will like conservative articles
                # So given a recommended article he will like or unlike it depending on the partisan of the article(labels are already created)
                # To measure precision and recall , is it after N updates that we measure model performance on the validation set of the articles
                pred_verdict = candidate_pool_y[top_index]
                y_preds.append(pred_verdict)
                    
                
                # update the model only if recommended item was liked by the user
              
                clf.partial_fit(candidate_pool_x[top_index].reshape(1, -1),np.array([candidate_pool_y[top_index]]))
                
                # update candidate pool
                candidate_pool_x = np.delete(candidate_pool_x, (top_index), axis=0)
                candidate_pool_y = np.delete(candidate_pool_y,(top_index),axis=0)
                
      
                # Active Learning Based Metric Calculations
                f1,precision,recall,accuracy = get_scores_wot(y_test_val,
                                                          predictions=clf.predict(x_test_val))

                scores_map[model]["f1"].append(f1)
                scores_map[model]["precision"].append(precision)
                scores_map[model]["recall"].append(recall)
                scores_map[model]["accuracy"].append(accuracy) 
        
        cp_scores_map[cp] = scores_map
    return cp_scores_map

@timer
def run_online_setting_atK(X,cluster_2_doc_map,df,cluster_pairs,cosine_mat,user_type="Heterogeneous"):
    """
    """
    cp_scores_map = {}
    results_df_map = {}
    for index,cp in enumerate(cluster_pairs):
        print("Training model for cluster pair : %s" %str(cp))
        x_train,x_test,y_train,y_test = create_train_test(cluster_pair=cp,
                                                          cluster2doc=cluster_2_doc_map,
                                                          X_feats=X,
                                                          df=df,
                                                          user_type=user_type)
        
        # Initial Training on Cluster 1
        estimators = [SGDClassifier()]
        all_param_grids = {0:{"loss":["log"],
                              "penalty":["l1","l2"],
                              "alpha":[0.0001,0.001,0.01],
                              "random_state":[RANDOM_SEED],
                              "n_jobs":[-1]}}
        best_models = []
        for index_est,estimator in enumerate(estimators):
            gcv = GridSearchCV(estimator,all_param_grids[index_est],scoring="f1_macro",n_jobs=-1)
            gcv.fit(x_train,y_train)
            best_model = gcv.best_estimator_
            best_models.append(best_model)
        
        # Online Training on Cluster 2 with updates(partial_fit)
        # for each estimator run 
        scores_map = defaultdict(lambda : defaultdict(list))
        models=["logistic_regression"]
        
        # To measure @K Precision
        # Set N=200 (min cluster size is 500, and min partisan size = 0.4*500=200)
        for index_bm,clf in enumerate(best_models):
            model = models[index_bm]
            N = 200
            total_relevant_docs = Counter(y_test)[1.0]
            print("Total Candidate Pool Size %s" %str(len(y_test)))
            print("Total Rel Docs : %s" %str(total_relevant_docs))
            liked_docs_recommended = 0
            y_true_sub = []
            y_preds = []
            candidate_pool_x = copy.deepcopy(x_test)
            candidate_pool_y = copy.deepcopy(y_test)
            
            
            for i in range(N):
                # recommend article to user
                probas = clf.predict_proba(candidate_pool_x)[:,1]
                rank_indices = np.argsort(probas)
                top_index = rank_indices[-1]
     
                pred_verdict = candidate_pool_y[top_index]
                y_preds.append(pred_verdict)
                    
                
                # update the model 
                
                clf.partial_fit(candidate_pool_x[top_index].reshape(1, -1),np.array([candidate_pool_y[top_index]]))
                
                # update candidate pool
                candidate_pool_x = np.delete(candidate_pool_x, (top_index), axis=0)
                candidate_pool_y = np.delete(candidate_pool_y,(top_index),axis=0)
                
                
                # Recall @K
                recall = Counter(y_preds[:i+1])[1.0]/total_relevant_docs
                # Precision @K
                precision = (Counter(y_preds[:i+1])[1.0])/len(y_preds[:i+1])
                # F1 @K (with zero division handle)
                f1 = None
                if recall+precision == 0.0:
                    f1 = 0.0
                else:
                    f1 = 2 *((recall*precision)/(recall+precision))

                scores_map[model]["f1"].append(f1)
                scores_map[model]["precision"].append(precision)
                scores_map[model]["recall"].append(recall)
        
        df_temp = pd.DataFrame()
        df_temp["Shown at K"] = y_preds
        df_temp["Recall at K"] = scores_map[model]["recall"]
        df_temp["Precision at K"] = scores_map[model]["precision"]
        results_df_map[cp] = df_temp
        cp_scores_map[cp] = scores_map
    return cp_scores_map, results_df_map

@timer
def run_single_cluster_performance(X,cluster_2_doc_map,df,cluster_pairs,cosine_mat,cluster_2_use=2,user_type="Heterogeneous"):
    """
    Measures Systems Performance on a Single Cluster, used to test the performance on cluster 2 just to compare
    how well the recommendation system seems to be able to detect change in topics
    """
    cp_scores_map = {}
    N=200
    for index,cp in enumerate(cluster_pairs):
        print("Training model for cluster pair : %s" %str(index))
        x_train,x_test,y_train,y_test = create_train_test(cluster_pair=cp,
                                                          cluster2doc=cluster_2_doc_map,
                                                          X_feats=X,
                                                          df=df,
                                                          user_type=user_type)
        
        if cluster_2_use == 2:
            x_train = x_test
            y_train = y_test
        
        # Bootstrap data and candidate pool
        x_bootstrap, x_cp, y_bootstrap, y_cp = train_test_split(x_train,
                                                                y_train,
                                                                test_size=N,
                                                                random_state=RANDOM_SEED,
                                                                shuffle=True,
                                                                stratify=y_train)
        
        # Initial Training on Cluster 1
        estimators = [SGDClassifier()]
        all_param_grids = {0:{"loss":["log"],
                              "penalty":["l1","l2"],
                              "alpha":[0.0001,0.001,0.01],
                              "random_state":[RANDOM_SEED],
                              "n_jobs":[-1]}}
        best_models = []
        for index_est,estimator in enumerate(estimators):
            gcv = GridSearchCV(estimator,all_param_grids[index_est],scoring="f1_macro",n_jobs=-1)
            gcv.fit(x_bootstrap,y_bootstrap)
            best_model = gcv.best_estimator_
            best_models.append(best_model)
        
        scores_map = defaultdict(lambda : defaultdict(list))
        models=["logistic_regression"]
        
      
        for index_bm,clf in enumerate(best_models):
            model = models[index_bm]
            total_relevant_docs = Counter(y_cp)[1.0]
            total_interactions = int(N/2)
            print("Total Candidate Pool Size : %s" %str(len(y_cp)))
            print("Total Rel Docs : %s" %str(total_relevant_docs))
            liked_docs_recommended = 0
            y_preds = []
            
            candidate_pool_x = copy.deepcopy(x_cp)
            candidate_pool_y = copy.deepcopy(y_cp)
            
            for i in range(total_interactions):
                # recommend article to user
                probas = clf.predict_proba(candidate_pool_x)[:,1]
                rank_indices = np.argsort(probas)
                top_index = rank_indices[-1]
     
                pred_verdict = candidate_pool_y[top_index]
                y_preds.append(pred_verdict)
                    
                
                # update the model 
                clf.partial_fit(candidate_pool_x[top_index].reshape(1, -1),np.array([candidate_pool_y[top_index]]))
                
                # update candidate pool
                candidate_pool_x = np.delete(candidate_pool_x, (top_index), axis=0)
                candidate_pool_y = np.delete(candidate_pool_y,(top_index),axis=0)
                
                
                # Recall @K
                recall = Counter(y_preds)[1.0]/total_relevant_docs
                # Precision @K
                precision = (Counter(y_preds)[1.0])/len(y_preds)
                # F1 @K (with zero division handle)
                f1 = None
                if recall+precision == 0.0:
                    f1 = 0.0
                else:
                    f1 = 2 *((recall*precision)/(recall+precision))

                scores_map[model]["f1"].append(f1)
                scores_map[model]["precision"].append(precision)
                scores_map[model]["recall"].append(recall)
        
        cp_scores_map[cp] = scores_map
    return cp_scores_map

@timer
def run_regularization_variation(X,cluster_2_doc_map,df,cluster_pairs,cosine_mat,
                                 user_type="Heterogeneous",
                                 reg_constants=[0.0001,0.001,0.01,0.1,1.0]):
    """
    """
    cp_scores_map = {}
    N=100
    for index,cp in enumerate(cluster_pairs):
        print("Training model for cluster pair : %s" %str(index))
        x_train,x_test,y_train,y_test = create_train_test(cluster_pair=cp,
                                                          cluster2doc=cluster_2_doc_map,
                                                          X_feats=X,
                                                          df=df,
                                                          user_type=user_type)
        
        scores_map = defaultdict(lambda : defaultdict(list))
        for regc in reg_constants:
            
            clf = None
            if regc <= 0.0:
                clf = SGDClassifier(loss="log",penalty="l2",alpha=regc,learning_rate="constant",random_state=RANDOM_SEED,eta0=0.001)
            
            else:
                clf = SGDClassifier(loss="log",penalty="l2",alpha=regc,random_state=RANDOM_SEED)
                
            clf.fit(x_train,y_train)
            
            
            total_relevant_docs = Counter(y_test)[1.0]
            total_interactions = N
            print("Total Rel Docs : %s" %str(total_relevant_docs))
            liked_docs_recommended = 0
            y_preds = []
            candidate_pool_x = copy.deepcopy(x_test)
            candidate_pool_y = copy.deepcopy(y_test)
            
            for i in range(total_interactions):
                # recommend article to user
                probas = clf.predict_proba(candidate_pool_x)[:,1]
                rank_indices = np.argsort(probas)
                top_index = rank_indices[-1]
     
                pred_verdict = candidate_pool_y[top_index]
                y_preds.append(pred_verdict)
                    
                
                # update the model 
                clf.partial_fit(candidate_pool_x[top_index].reshape(1, -1),np.array([candidate_pool_y[top_index]]))
                
                # update candidate pool
                candidate_pool_x = np.delete(candidate_pool_x, (top_index), axis=0)
                candidate_pool_y = np.delete(candidate_pool_y,(top_index),axis=0)
                
                
                # Recall @K
                recall = Counter(y_preds)[1.0]/total_relevant_docs
                # Precision @K
                precision = (Counter(y_preds)[1.0])/len(y_preds)
                # F1 @K (with zero division handle)
                f1 = None
                if recall+precision == 0.0:
                    f1 = 0.0
                else:
                    f1 = 2 *((recall*precision)/(recall+precision))

                
                scores_map[str(regc)]["f1"].append(f1)
                scores_map[str(regc)]["precision"].append(precision)
                scores_map[str(regc)]["recall"].append(recall)
                
        cp_scores_map[cp] = scores_map
    return cp_scores_map

@timer
def run_learningrate_variation(X,cluster_2_doc_map,df,cluster_pairs,cosine_mat,
                                 user_type="Heterogeneous",
                                 lr=[0.001,0.01,0.1,1.0,10]):
    """
    """
    cp_scores_map = {}
    N=100
    for index,cp in enumerate(cluster_pairs):
        print("Training model for cluster pair : %s" %str(index))
        x_train,x_test,y_train,y_test = create_train_test(cluster_pair=cp,
                                                          cluster2doc=cluster_2_doc_map,
                                                          X_feats=X,
                                                          df=df,
                                                          user_type=user_type)
        
        scores_map = defaultdict(lambda : defaultdict(list))
        for l_r in lr:
            clf = SGDClassifier(loss="log",penalty="l2",eta0=l_r,learning_rate="constant",random_state=RANDOM_SEED)
            clf.fit(x_train,y_train)
            
            
            total_relevant_docs = Counter(y_test)[1.0]
            total_interactions = N
            print("Total Rel Docs : %s" %str(total_relevant_docs))
            liked_docs_recommended = 0
            y_preds = []
            candidate_pool_x = copy.deepcopy(x_test)
            candidate_pool_y = copy.deepcopy(y_test)
            
            for i in range(total_interactions):
                # recommend article to user
                probas = clf.predict_proba(candidate_pool_x)[:,1]
                rank_indices = np.argsort(probas)
                top_index = rank_indices[-1]
     
                pred_verdict = candidate_pool_y[top_index]
                y_preds.append(pred_verdict)
                    
                
                # update the model 
                clf.partial_fit(candidate_pool_x[top_index].reshape(1, -1),np.array([candidate_pool_y[top_index]]))
                
                # update candidate pool
                candidate_pool_x = np.delete(candidate_pool_x, (top_index), axis=0)
                candidate_pool_y = np.delete(candidate_pool_y,(top_index),axis=0)
                
                
                # Recall @K
                recall = Counter(y_preds)[1.0]/total_relevant_docs
                # Precision @K
                precision = (Counter(y_preds)[1.0])/len(y_preds)
                # F1 @K (with zero division handle)
                f1 = None
                if recall+precision == 0.0:
                    f1 = 0.0
                else:
                    f1 = 2 *((recall*precision)/(recall+precision))

                
                scores_map[str(l_r)]["f1"].append(f1)
                scores_map[str(l_r)]["precision"].append(precision)
                scores_map[str(l_r)]["recall"].append(recall)
                
        cp_scores_map[cp] = scores_map
    return cp_scores_map

@timer
def run_mixeddata_performance_check(X,
                                    cluster_2_doc_map,
                                    df,
                                    cluster_pairs,
                                    cosine_mat,
                                    user_type="Heterogeneous",):
    """
    """
    cp_scores_map = {}
    
    for index,cp in enumerate(cluster_pairs):
        print("Training model for cluster pair : %s" %str(index))
        x_train,x_test,y_train,y_test = create_train_test(cluster_pair=cp,
                                                          cluster2doc=cluster_2_doc_map,
                                                          X_feats=X,
                                                          df=df,
                                                          user_type=user_type)
        
        
        # get candidate pool from train
        x_bootstrap, x_train_cp, y_bootstrap, y_train_cp = train_test_split(x_train,
                                                                y_train,
                                                                test_size=100,
                                                                random_state=RANDOM_SEED,
                                                                shuffle=True,
                                                                stratify=y_train)
        
        # get candidate pool from test
        x_test_cp, _,  y_test_cp,_ = train_test_split(x_test,
                                                    y_test,
                                                    train_size=100,
                                                    random_state=RANDOM_SEED,
                                                    shuffle=True,
                                                    stratify=y_test)
        
        # Initial Training on Cluster 1
        estimators = [SGDClassifier()]
        all_param_grids = {0:{"loss":["log"],
                              "penalty":["l1","l2"],
                              "alpha":[0.0001,0.001,0.01],
                              "random_state":[RANDOM_SEED],
                              "n_jobs":[-1]}}
        best_models = []
        for index_est,estimator in enumerate(estimators):
            gcv = GridSearchCV(estimator,all_param_grids[index_est],scoring="f1_macro",n_jobs=-1)
            gcv.fit(x_bootstrap,y_bootstrap)
            best_model = gcv.best_estimator_
            best_models.append(best_model)
        
        # Online Training on Cluster 2 with updates(partial_fit)
        # for each estimator run 
        scores_map = defaultdict(lambda : defaultdict(list))
        models=["logistic_regression"]
        
        # To measure @K Precision
        for index_bm,clf in enumerate(best_models):
            model = models[index_bm]
            N = 100
            total_relevant_docs = Counter(y_test)[1.0]
            print("Total Rel Docs : %s" %str(total_relevant_docs))
            liked_docs_recommended = 0
            y_true_sub = []
            y_preds = []
            candidate_pool_train_x = copy.deepcopy(x_train_cp)
            candidate_pool_train_y = copy.deepcopy(y_train_cp)
            candidate_pool_test_x = copy.deepcopy(x_test_cp)
            candidate_pool_test_y = copy.deepcopy(y_test_cp)
            candidate_pool_x = np.concatenate([candidate_pool_train_x,candidate_pool_test_x],axis=0)
            candidate_pool_y = np.concatenate([candidate_pool_train_y,candidate_pool_test_y])
            print("Candidate Pool Size : %s" %candidate_pool_x.shape[0])
            
            def unison_shuffled_copies(a, b):
                """
                """
                assert len(a) == len(b)
                p = np.random.permutation(len(a))
                return a[p], b[p]
            
            candidate_pool_x , candidate_pool_y = unison_shuffled_copies(candidate_pool_x,candidate_pool_y)
            
            for i in range(N):
                # recommend article to user
                probas = clf.predict_proba(candidate_pool_x)[:,1]
                rank_indices = np.argsort(probas)
                top_index = rank_indices[-1]
     
                pred_verdict = candidate_pool_y[top_index]
                y_preds.append(pred_verdict)
                    
                
                # update the model 
                
                clf.partial_fit(candidate_pool_x[top_index].reshape(1, -1),np.array([candidate_pool_y[top_index]]))
                
                # update candidate pool
                candidate_pool_x = np.delete(candidate_pool_x, (top_index), axis=0)
                candidate_pool_y = np.delete(candidate_pool_y,(top_index),axis=0)
                
                
                # Recall @K
                recall = Counter(y_preds)[1.0]/total_relevant_docs
                # Precision @K
                precision = (Counter(y_preds)[1.0])/len(y_preds)
                # F1 @K (with zero division handle)
                f1 = None
                if recall+precision == 0.0:
                    f1 = 0.0
                else:
                    f1 = 2 *((recall*precision)/(recall+precision))

                scores_map[model]["f1"].append(f1)
                scores_map[model]["precision"].append(precision)
                scores_map[model]["recall"].append(recall)
        
        cp_scores_map[cp] = scores_map
    return cp_scores_map

