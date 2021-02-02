from general_utils import timer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV,SGDClassifier,PassiveAggressiveClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
import numpy as np
from metrics_utils import get_scores, get_scores_wot
from data_utils import create_train_test
from config import RANDOM_SEED
from collections import defaultdict, Counter
import copy

@timer
def run_model(x_train,x_test,y_train,y_test,seed=RANDOM_SEED):
    """
    """
    clf = LogisticRegressionCV(cv=5,random_state=seed,max_iter=1000,n_jobs=-1,class_weight="balanced").fit(x_train, y_train)
    predicted_probas = clf.predict_proba(x_test)
    return clf,predicted_probas


    
@timer
def run_bs1_train_all(X,cluster_2_doc_map,df,cluster_pairs,cosine_mat,thresholds = [0.5,0.7,0.9],user_type="Heterogeneous"):
    """
    Runs the baseline1 setting where we want to detect how cluster similarity affects the recommender for 
    different user types - Not online setting
    """
    results = defaultdict(list)
    for index,cp in enumerate(cluster_pairs):
            print("Training model for cluster pair : %s" %str(index))
            x_train,x_test,y_train,y_test,cluster_1_doc_indices,cluster_2_doc_indices = create_train_test(cluster_pair=cp,
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
    Performance measurement based on online + Active learning setting
    """
    cp_scores_map = {}
    
    for index,cp in enumerate(cluster_pairs):
        print("Training model for cluster pair : %s" %str(index))
        x_train,x_test,y_train,y_test,cluster_1_doc_indices,cluster_2_doc_indices = create_train_test(cluster_pair=cp,
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
            y_true_sub = []
            y_preds = []
            
            candidate_pool_x = copy.deepcopy(x_test_sub)
            candidate_pool_y = copy.deepcopy(y_test_sub)
            
            for i in range(N):
                # recommend article to user
                probas = clf.predict_proba(candidate_pool_x)[:,1]
                rank_indices = np.argsort(probas)
                top_index = rank_indices[-1]

                pred_verdict = candidate_pool_y[top_index]
                y_preds.append(pred_verdict)
              
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
def run_bs2_train_all(X,cluster_2_doc_map,df,cluster_pairs,cosine_mat,user_type="Heterogeneous"):
    """
    """
    cp_scores_map = {}
    results_df_map = {}
    for index,cp in enumerate(cluster_pairs):
        print("Training model for cluster pair : %s" %str(cp))
        x_train,x_test,y_train,y_test,cluster_1_doc_indices,cluster_2_doc_indices = create_train_test(cluster_pair=cp,
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
def run_bs3_train_all(X,cluster_2_doc_map,df,cluster_pairs,cosine_mat,cluster_2_use=2,user_type="Heterogeneous"):
    """
    Measures Systems Performance on a Single Cluster, used to test the performance on cluster 2 just to compare
    how well the recommendation system seems to be able to detect change in topics
    """
    cp_scores_map = {}
    N=200
    for index,cp in enumerate(cluster_pairs):
        print("Training model for cluster pair : %s" %str(index))
        x_train,x_test,y_train,y_test,cluster_1_doc_indices,cluster_2_doc_indices = create_train_test(cluster_pair=cp,
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
def run_bs4_train_all(X,cluster_2_doc_map,df,cluster_pairs,cosine_mat,
                                 user_type="Heterogeneous",
                                 reg_constants=[0.0001,0.001,0.01,0.1,1.0]):
    """
    """
    cp_scores_map = {}
    N=100
    for index,cp in enumerate(cluster_pairs):
        print("Training model for cluster pair : %s" %str(index))
        x_train,x_test,y_train,y_test,cluster_1_doc_indices,cluster_2_doc_indices = create_train_test(cluster_pair=cp,
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
def run_bs5_train_all(X,sample_df,cluster_2_doc_map,df,cluster_pairs,cosine_mat,
                                 user_type="Heterogeneous",
                                 lr=[0.001,0.01,0.1,1.0,10],debug=False):
    """
    """
    cp_scores_map = {}
    results_df_map_100 = {}
    results_df_map_500 = {}
    for index,cp in enumerate(cluster_pairs):
        print("Training model for cluster pair : %s" %str(index))
        x_train,x_test,y_train,y_test,cluster_1_doc_indices,cluster_2_doc_indices = create_train_test(cluster_pair=cp,
                                                          cluster2doc=cluster_2_doc_map,
                                                          X_feats=X,
                                                          df=df,
                                                          user_type=user_type)
        
        # check 
        scores_map = defaultdict(lambda : defaultdict(list))
        for l_r in lr:
            print("\n*************** CP = %s , LR = %s ****************" %(str(cp),str(l_r)))
            clf = SGDClassifier(loss="log",penalty="l2",eta0=l_r,learning_rate="constant",random_state=RANDOM_SEED)
            print(str(clf))
            clf.fit(x_train,y_train)
            
            
            total_relevant_docs = Counter(y_test)[1.0]
            total_interactions = 200
            liked_docs_recommended = 0
            y_preds = []
            y_pred_text = []
            candidate_pool_x = copy.deepcopy(x_test)
            candidate_pool_y = copy.deepcopy(y_test)

            for i in range(total_interactions):
                
                probas = clf.predict_proba(candidate_pool_x)[:,1]
                rank_indices = np.argsort(probas)
                top_index = rank_indices[-1]
                pred_verdict = candidate_pool_y[top_index]
                y_preds.append(pred_verdict)
                y_pred_text.append(sample_df["processed_text"].iloc[cluster_2_doc_indices[top_index]])
                    
                
                # update the model 
                clf.partial_fit(candidate_pool_x[top_index].reshape(1, -1),np.array([candidate_pool_y[top_index]]))
                
                # update candidate pool
                candidate_pool_x = np.delete(candidate_pool_x, (top_index), axis=0)
                candidate_pool_y = np.delete(candidate_pool_y,(top_index),axis=0)
                cp_original_index = np.delete(cluster_2_doc_indices,(top_index),axis=0)
                
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
                scores_map[str(l_r)]["y_pred_text"] = y_pred_text
                
            clf = None
                
        cp_scores_map[cp] = scores_map
        df_temp = pd.DataFrame()
        df_temp["Shown at K"] = y_preds
        df_temp["Recall at K"] = scores_map["100"]["recall"]
        df_temp["Precision at K"] = scores_map["100"]["precision"]
        df_temp["Top Article"] = scores_map["100"]["y_pred_text"]
        results_df_map_100[cp] = df_temp
        
        df_temp = pd.DataFrame()
        df_temp["Shown at K"] = y_preds
        df_temp["Recall at K"] = scores_map["500"]["recall"]
        df_temp["Precision at K"] = scores_map["500"]["precision"]
        df_temp["Top Article"] = scores_map["500"]["y_pred_text"]
        results_df_map_500[cp] = df_temp
        
    return cp_scores_map,results_df_map_100,results_df_map_500

@timer
def run_bs6_train_all(X,
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
        x_train,x_test,y_train,y_test,cluster_1_doc_indices,cluster_2_doc_indices = create_train_test(cluster_pair=cp,
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
            N = 200
            total_relevant_docs = Counter(y_test)[1.0]
            liked_docs_recommended = 0
            y_true_sub = []
            y_preds = []

            y_pred_c1 = []
            
            y_pred_c2 = []
            
            cwclus = []
            
            candidate_pool_train_x = copy.deepcopy(x_train_cp)
            candidate_pool_train_y = copy.deepcopy(y_train_cp)
            
            c1_total_rel = Counter(candidate_pool_train_y)[1.0]
            
            candidate_pool_test_x = copy.deepcopy(x_test_cp)
            candidate_pool_test_y = copy.deepcopy(y_test_cp)
            
            c2_total_rel = Counter(candidate_pool_test_y)[1.0]
            
            candidate_pool_x = np.concatenate([candidate_pool_train_x,candidate_pool_test_x],axis=0)
            candidate_pool_y = np.concatenate([candidate_pool_train_y,candidate_pool_test_y])
            which_cluster = np.concatenate([np.array([1]*candidate_pool_train_x.shape[0]),
                                            np.array([2]*candidate_pool_test_x.shape[0])])
            
            def unison_shuffled_copies(a, b, c):
                """
                """
                assert len(a) == len(b) == len(c)
                p = np.random.permutation(len(a))
                return a[p], b[p], c[p]
            
            candidate_pool_x , candidate_pool_y, which_cluster = unison_shuffled_copies(candidate_pool_x,candidate_pool_y, which_cluster)
            
            
            for i in range(N):
                # recommend article to user
                probas = clf.predict_proba(candidate_pool_x)[:,1]
                rank_indices = np.argsort(probas)
                top_index = rank_indices[-1]
     
                pred_verdict = candidate_pool_y[top_index]
                y_preds.append(pred_verdict)
                cwclus.append(which_cluster[top_index])
                
                # update the model 
                clf.partial_fit(candidate_pool_x[top_index].reshape(1, -1),np.array([candidate_pool_y[top_index]]))
                
                # update candidate pool
                candidate_pool_x = np.delete(candidate_pool_x, (top_index), axis=0)
                candidate_pool_y = np.delete(candidate_pool_y,(top_index),axis=0)
                
                # record which cluster recommended item came from
                scores_map[model]["which_cluster"].append(which_cluster[top_index])
                
                which_cluster = np.delete(which_cluster,(top_index),axis=0)
                
                # Recall @K
                recall = Counter(y_preds)[1.0]/total_relevant_docs
                recall_c1 = Counter([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 1])[1.0]/c1_total_rel
                recall_c2 = Counter([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 2])[1.0]/c2_total_rel
                
                # Precision @K
                precision = (Counter(y_preds)[1.0])/len(y_preds)
                precision_c1 = 0.0
                precision_c2 = 0.0
                if len([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 1])>0:
                    precision_c1 = Counter([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 1])[1.0]/len([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 1])
                if len([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 2])>0:
                    precision_c2 = Counter([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 2])[1.0]/len([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 2])
                
                # F1 @K (with zero division handle)
                f1 = None
                f1_c1 = None
                f1_c2 = None
                if recall+precision == 0.0:
                    f1 = 0.0
                if recall_c1 + precision_c1 == 0.0:
                    f1_c1 = 0.0
                if recall_c2 + precision_c2 == 0.0:
                    f1_c2 = 0.0
                
                if recall+precision != 0.0: 
                    f1 = 2 *((recall*precision)/(recall+precision))
                if recall_c1 + precision_c1 != 0.0:
                    f1_c1 = 2 *((recall_c1*precision_c1)/(recall_c1+precision_c1))
                if recall_c2 + precision_c2 != 0.0:     
                    f1_c2 = 2 *((recall_c2*precision_c2)/(recall_c2+precision_c2))

                scores_map[model]["f1"].append(f1)
                scores_map[model]["f1_c1"].append(f1_c1)
                scores_map[model]["f1_c2"].append(f1_c2)
                scores_map[model]["precision"].append(precision)
                scores_map[model]["precision_c1"].append(precision_c1)
                scores_map[model]["precision_c2"].append(precision_c2)
                scores_map[model]["recall"].append(recall)
                scores_map[model]["recall_c1"].append(recall_c1)
                scores_map[model]["recall_c2"].append(recall_c2)
        
        cp_scores_map[cp] = scores_map
    return cp_scores_map
        
        
@timer
def run_bs7_train_all(X,cluster_2_doc_map,df,cluster_pairs,cosine_mat,
                                 user_type="Heterogeneous",
                                 lr=[0.001,0.01,0.1,1.0,10]):
    """
    """
    cp_scores_map = {}
    N=200
    for index,cp in enumerate(cluster_pairs):
        print("Training model for cluster pair : %s" %str(index))
        x_train,x_test,y_train,y_test,cluster_1_doc_indices,cluster_2_doc_indices = create_train_test(cluster_pair=cp,
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
        
        scores_map = defaultdict(lambda : defaultdict(list))
        for l_r in lr:
            clf = SGDClassifier(loss="log",penalty="l2",eta0=l_r,learning_rate="constant",random_state=RANDOM_SEED)
            clf.fit(x_bootstrap,y_bootstrap)
            
            
            total_relevant_docs = Counter(y_test)[1.0]
            total_interactions = N
            liked_docs_recommended = 0
            y_preds = []
            
            y_pred_c1 = []
            
            y_pred_c2 = []
            
            cwclus = []
            
            candidate_pool_train_x = copy.deepcopy(x_train_cp)
            candidate_pool_train_y = copy.deepcopy(y_train_cp)
            
            c1_total_rel = Counter(candidate_pool_train_y)[1.0]
            
            candidate_pool_test_x = copy.deepcopy(x_test_cp)
            candidate_pool_test_y = copy.deepcopy(y_test_cp)
            
            c2_total_rel = Counter(candidate_pool_test_y)[1.0]

            candidate_pool_x = np.concatenate([candidate_pool_train_x,candidate_pool_test_x],axis=0)
            candidate_pool_y = np.concatenate([candidate_pool_train_y,candidate_pool_test_y])
            which_cluster = np.concatenate([np.array([1]*candidate_pool_train_x.shape[0]),
                                            np.array([2]*candidate_pool_test_x.shape[0])])
            
            def unison_shuffled_copies(a, b, c):
                """
                """
                assert len(a) == len(b) == len(c)
                p = np.random.permutation(len(a))
                return a[p], b[p], c[p]
            
            candidate_pool_x , candidate_pool_y, which_cluster = unison_shuffled_copies(candidate_pool_x,candidate_pool_y, which_cluster)
            
            for i in range(total_interactions):
                # recommend article to user
                probas = clf.predict_proba(candidate_pool_x)[:,1]
                rank_indices = np.argsort(probas)
                top_index = rank_indices[-1]
     
                pred_verdict = candidate_pool_y[top_index]
                y_preds.append(pred_verdict)
                cwclus.append(which_cluster[top_index])    
                
                # update the model 
                
                clf.partial_fit(candidate_pool_x[top_index].reshape(1, -1),np.array([candidate_pool_y[top_index]]))
                
                # update candidate pool
                candidate_pool_x = np.delete(candidate_pool_x, (top_index), axis=0)
                candidate_pool_y = np.delete(candidate_pool_y,(top_index),axis=0)
                # record which cluster recommended item came from
                scores_map[str(l_r)]["which_cluster"].append(which_cluster[top_index])
                which_cluster = np.delete(which_cluster,(top_index),axis=0)
                
                # Recall @K
                recall = Counter(y_preds)[1.0]/total_relevant_docs
                recall_c1 = Counter([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 1])[1.0]/c1_total_rel
                recall_c2 = Counter([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 2])[1.0]/c2_total_rel
                
                # Precision @K
                precision = (Counter(y_preds)[1.0])/len(y_preds)
                precision_c1 = 0.0
                precision_c2 = 0.0
                if len([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 1])>0:
                    precision_c1 = Counter([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 1])[1.0]/len([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 1])
                if len([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 2])>0:
                    precision_c2 = Counter([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 2])[1.0]/len([pred for ind,pred in enumerate(y_preds) if cwclus[ind] == 2])
                
                # F1 @K (with zero division handle)
                f1 = None
                f1_c1 = None
                f1_c2 = None
                if recall+precision == 0.0:
                    f1 = 0.0
                if recall_c1 + precision_c1 == 0.0:
                    f1_c1 = 0.0
                if recall_c2 + precision_c2 == 0.0:
                    f1_c2 = 0.0
                
                if recall+precision != 0.0: 
                    f1 = 2 *((recall*precision)/(recall+precision))
                if recall_c1 + precision_c1 != 0.0:
                    f1_c1 = 2 *((recall_c1*precision_c1)/(recall_c1+precision_c1))
                if recall_c2 + precision_c2 != 0.0:     
                    f1_c2 = 2 *((recall_c2*precision_c2)/(recall_c2+precision_c2))

                scores_map[str(l_r)]["f1"].append(f1)
                scores_map[str(l_r)]["f1_c1"].append(f1_c1)
                scores_map[str(l_r)]["f1_c2"].append(f1_c2)
                scores_map[str(l_r)]["precision"].append(precision)
                scores_map[str(l_r)]["precision_c1"].append(precision_c1)
                scores_map[str(l_r)]["precision_c2"].append(precision_c2)
                scores_map[str(l_r)]["recall"].append(recall)
                scores_map[str(l_r)]["recall_c1"].append(recall_c1)
                scores_map[str(l_r)]["recall_c2"].append(recall_c2)
                
        cp_scores_map[cp] = scores_map
    return cp_scores_map