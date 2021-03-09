"""
Baseline Descriptions :
-----------------------
* Baseline 1:
* Baseline 2:
* Baseline 3:
* Baseline 4:
* Baseline 5:
* Baseline 6:
* Baseline 7:

To-DO:
-----------
* From all plotting functions remove plt.show()

#1
bsl1_sim_vs_score_all_users(df_results_hetero=df_result,
                                df_results_homo=df_result_homog,
                                threshold=threshold)
                                
# 2
plot_online_learning_cumulative(homo_scores,hetero_scores,active=False)
plot_all_cp_online(homo_scores,user_type="Homogeneous",metric="precision")
plot_all_cp_online(hetero_scores,user_type="Heterogeneous",metric="precision")

# 3
plot_single_cluster_cumulative(homo_scores,hetero_scores)
plot_all_cp_singlecluster(homo_scores)

# 4
plot_regularization_vs_metrics_at_k_cumu(homo_scores,
                                         regularization_constants=[0.0001,0.001,0.01,0.1,0.0,1.0,10.0,20.0,50.0,100.0],
                                         user_type="Homogeneous")
plot_regularization_vs_metrics_at_k_cumu(hetero_scores,
                                         regularization_constants=[0.0001,0.001,0.01,0.1,0.0,1.0,10.0,20.0,50.0,100.0])

#5
plot_lr_vs_metrics_at_k_cumu(homo_scores,
                             lr=[0.001,0.01,0.1,1.0,10,15,20,50,100,500],
                             user_type="Homogeneous")

plot_lr_vs_metrics_at_k_cumu(hetero_scores,
                             lr=[0.001,0.01,0.1,1.0,10,15,20,50,100,500])

#6
plot_mixed_cluster_cumulative_per_cluster(homo_scores,hetero_scores)
plot_mixed_data_all_cp_perf(scores_cp=homo_scores,user_type="Homogeneous",metric="precision")
plot_mixed_data_all_cp_perf(scores_cp=hetero_scores,user_type="Heterogeneous",metric="precision")

#7
plot_lr_vs_metrics_at_k_cumu(homo_scores,
                             lr=[0.001,0.01,0.1,1.0,10,15,20,50,100,500],
                             user_type="Homogeneous",mixed_data=True)

plot_lr_vs_metrics_at_k_cumu(hetero_scores,
                             lr=[0.001,0.01,0.1,1.0,10,15,20,50,100,500],mixed_data=True)
                             
                             
* rename the saved plots to represent the baseline that generated them
* to send string arguments to function have dict {"keyword": argument_value}, then **{} when passing to function
* for the 1st baseline graph update the graph to conside
* finish args4plotters



* bert embeddings loader needs to be created
* google sheet support

* Update config file to contain save paths for graphs and other settings (for cluster, tfidf, params) - DONE
* finish the scorer for bot class - Done
* finish main runner for below class - Done
* all the model runners need a scaling step involved - Done

"""

import config as CONFIG
import model_utils
import numpy as np
import pandas as pd
import model_utils 
import plotters_baseline
import argparse

from bert_embeddings import load_bert_embeddings
from preprocess_utils import tfidf_vectorization, vectorize_text
from clustering_utils import run_clustering, get_cluster_sizes, score_cluster, get_cluster_pairs, get_pairwise_dist, cluster2doc, filter_clusters, get_top_100_clusterpairs
from metrics_utils import calculate_avg_precision_param_variation, calculate_map_param_variation, calculate_avg_precision


class BaselineRunner(object):
    def __init__(self,file_path,baseline_type="all",rep_type="bert_12_100",plot=False):
        
        self.data_path = file_path
        self.baseline_type = baseline_type
        self.rep_type = rep_type
        self.plot_flag = plot
        
        # Initialize our baseline function runners and scores map (scores are precision,recall,f1)
        self.num_baselines = [str(i+1) for i in range(7)]
        self.baseline_func_maps = {str(i+1):getattr(model_utils, 'run_bs%s_train_all'%str(i+1))  for i in range(7)}
        self.scores_map = {str(i+1):{"Heterogeneous_user":None, "Homogeneous_user":None} for i in range(7)}
        self.mean_avg_prec_map = {str(i+1):{"Heterogeneous_user":{}, "Homogeneous_user":{}} for i in range(7)}
        
        
        self.scoring_type = {"1": None,
                             "2":"single",
                             "3":"single",
                             "4":"multiple",
                             "5":"multiple",
                             "6":"multiple",
                             "7":"multiple"}
        
        
        if self.plot_flag:
            self.plot_type = {"1":["comparison"],
                              "2":["comparison","single","single"],
                              "3":["comparison","single"],
                              "4":["single","single"],
                              "5":["single","single"],
                              "6":["comparison","single","single"],
                              "7":["single","single"],}
            
            self.baseline_plotters_func_map = {"1":["bsl1_sim_vs_score_all_users"],
                                               "2":["plot_online_learning_cumulative","plot_all_cp_online","plot_all_cp_online"],
                                               "3":["plot_single_cluster_cumulative","plot_all_cp_singlecluster"],
                                               "4":["plot_regularization_vs_metrics_at_k_cumu","plot_regularization_vs_metrics_at_k_cumu"],
                                               "5":["plot_lr_vs_metrics_at_k_cumu","plot_lr_vs_metrics_at_k_cumu"],
                                               "6":["plot_mixed_cluster_cumulative_per_cluster","plot_mixed_data_all_cp_perf","plot_mixed_data_all_cp_perf"],
                                               "7":["plot_lr_vs_metrics_at_k_cumu","plot_lr_vs_metrics_at_k_cumu"]}
        
        self.baseline_params_map = {"4":CONFIG.rc_params, "5":CONFIG.lr_params, "7":CONFIG.lr_params}
        
        #Initialize other class attributes
        self.data = None
        self.vectors = None
        self.clusters = None
        self.cluster_clf = None
        self.cluster_sizes = None
        self.cluster_pairs_all = None
        self.filtered_cluster_pairs = None
        self.top100_cps = None
        self.cluster_2_doc_map = None
        self.cluster_pair_dist_mat = None
        self.bert_contextualization_perc = None
        pass
    
    def run(self):
        """
        """
        # load data
        self.data = pd.read_csv(self.data_path)
        # get vector representations
        self._get_rep_()
        # cluster
        self._cluster_()
        # get cluster sizes
        self._get_cluster_sizes_()
        # get cluster_2_doc
        self._map_cluster_2_doc_()
        # _gen_cluster_pairs_
        self._gen_cluster_pairs_()
        # filter cluster pairs
        self._filter_clusterpairs_()
        # _get_top_100_cluster_pairs_
        self._get_top_100_cluster_pairs_()
        # run_baselines
        self._run_baselines_(self.baseline_type)
        # calculate maps
        self._calc_map_()
        # save maps
        self._save_scores_()
        
        # generate_plots
        if self.plot_flag:
            self._plot_baseline_results()
            
    
    def _run_baselines_(self,baseline="all"):
        """
        """
        print("\n .......... Running Baselines ..........\n")
        if baseline == "all":
            for baseline_num in self.baseline_func_maps.keys():
                if baseline_num != "1":
                    print("\nRunning Baseline %s"%str(baseline_num))
                    self._run_baseline_(baseline_2_run=baseline_num)
        else:
            print("\nRunning Baseline %s"%str(baseline_num))
            self._run_baseline_(baseline_2_run=baseline)
    
    def _run_baseline_(self,baseline_2_run="2"):
        """
        """
        print("\n .......... Running Baseline - %s..........\n"%str(baseline_2_run))
        kwargs = {"X":self.vectors,
                "cluster_2_doc_map":self.cluster_2_doc_map, 
                "df":self.data,
                "cluster_pairs":self.top100_cps,
                "cosine_mat":self.cluster_pair_dist_mat}
        
        if baseline_2_run in ["5","7"]:
            kwargs["lr"] = self.baseline_params_map[baseline_2_run]
        
        if baseline_2_run == "4":
            kwargs["reg_constants"] = self.baseline_params_map[baseline_2_run]
        
        self.scores_map[baseline_2_run]["Homogeneous_user"] = self.baseline_func_maps[baseline_2_run](user_type="Homogeneous",**kwargs)
        
        self.scores_map[baseline_2_run]["Heterogeneous_user"] = self.baseline_func_maps[baseline_2_run](user_type="Heterogeneous",**kwargs)
        
    
    def _get_rep_(self):
        """
        """
        print("\nRepresentation-Chosen : %s"%self.rep_type )
        if self.rep_type  == "tf-idf":
            vectors,vocab,tfidf_vectorizer = tfidf_vectorization(df=self.data,
                                                                 min_df=CONFIG.min_df,
                                                                 max_df=CONFIG.max_df,
                                                                 seed=CONFIG.RANDOM_SEED)
            print("Vocab Size : %s" %str(len(vocab)))
            self.vectors = vectors.todense()
        
        elif self.rep_type  == "glove":
            self.vectors = vectorize_text(df=self.data,mode="glove")
            print("Glove Dimensions : %s" %str(self.vectors.shape))
        
        elif "bert" in self.rep_type :
            layer = int(self.rep_type .split("_")[1])
            perc_vc = int(self.rep_type .split("_")[-1])
            print("\nLoading Bert with %s %% contextualization"%str(perc_vc))
            self.vectors = load_bert_embeddings(df=self.data,
                                                saved_path="/media/karthikshivaram/SABER_4TB/bert_embeddings",
                                                batch_size=50,
                                                layer=layer,
                                                context_var=perc_vc,
                                                aggregation="mean")
    
    def _cluster_(self):
        """
        """
        print("\nRunning Clustering .....")
        clusters,cluster_clf = run_clustering(vectors=self.vectors,
                                              seed=CONFIG.RANDOM_SEED,
                                              num_clusters=CONFIG.num_clusters,
                                              clus_type="kmeans")
        self.clusters = clusters
        self.cluster_clf = cluster_clf
        pass
    
    def _get_cluster_sizes_(self):
        """
        """
        self.cluster_sizes = get_cluster_sizes(self.cluster_clf)
    
    def _gen_cluster_pairs_(self):
        """
        """
        cluster_pairs = get_cluster_pairs(num_clusters=CONFIG.num_clusters)
        self.cluster_pairs_all = cluster_pairs
    
    def _get_top_100_cluster_pairs_(self):
        """
        """
        if len(self.filtered_cluster_pairs) > 100:
            print("\nNumber of Filtered Cluster Pairs are greater 100, picking top 100 most similar cluster pairs")
            self.cluster_pair_dist_mat = get_pairwise_dist(self.cluster_clf,dist_type="cosine")
            top100 = get_top_100_clusterpairs(cluster_pairs=self.filtered_cluster_pairs,dist_matrix=self.cluster_pair_dist_mat,reverse=True)
            self.top100_cps = top100 
        
        else:
            self.top100_cps = self.filtered_cluster_pairs
            print("\nNumber of Filtered Cluster Pairs is less than 100 so skipping top 100 selection")
    
    def _map_cluster_2_doc_(self):
        """
        """
        doc_2_cluster_map = cluster2doc(num_texts=self.data.shape[0],cluster_labels=self.cluster_clf.labels_)
        self.cluster_2_doc_map = doc_2_cluster_map
    
    def _filter_clusterpairs_(self):
        """
        """
        print("\nNum of Possible Cluster Pairs : %s" %str(len(self.cluster_pairs_all)))
        
        filtered_cluster_pairs = filter_clusters(cluster_pairs=self.cluster_pairs_all,
                                        doc_2_cluster_map=self.cluster_2_doc_map,
                                        cluster_sizes=self.cluster_sizes,
                                        partisan_scores=self.data["binary_ps"].tolist(),
                                        min_size=CONFIG.min_cluster_size,
                                        max_size=CONFIG.max_cluster_size,
                                        min_partisan_size=CONFIG.min_partisan_size)

        print("Num of Cluster Pairs After Filtering : %s" %str(len(filtered_cluster_pairs)))
        
        self.filtered_cluster_pairs = filtered_cluster_pairs
    
    def _calc_map_(self):
        """
        """
        for baseline in self.scores_map.keys():
            
            if baseline in ["2","3","6"]:
                
                if baseline == "6":
                    avg_prescision, c1_avg_precision, c2_avg_precision = calculate_avg_precision(self.scores_map[baseline]["Heterogeneous_user"],mode="mixed")
                    self.mean_avg_prec_map[baseline]["Heterogeneous_user"]["map"] = {"avg_precision":np.mean(avg_prescision),"c1_avg_precision":np.mean(c1_avg_precision),"c2_avg_precision":np.mean(c2_avg_precision)}
                    avg_prescision, c1_avg_precision, c2_avg_precision = calculate_avg_precision(self.scores_map[baseline]["Homogeneous_user"],mode="mixed")
                    self.mean_avg_prec_map[baseline]["Homogeneous_user"]["map"] = {"avg_precision":np.mean(avg_prescision),"c1_avg_precision":np.mean(c1_avg_precision),"c2_avg_precision":np.mean(c2_avg_precision)}
                else:
                    avg_prescision = calculate_avg_precision(self.scores_map[baseline]["Heterogeneous_user"],mode="single")
                    self.mean_avg_prec_map[baseline]["Heterogeneous_user"]["map"]={"avg_precision":np.mean(avg_prescision),"c1_avg_precision":np.nan,"c2_avg_precision":np.nan}
                    avg_prescision = calculate_avg_precision(self.scores_map[baseline]["Homogeneous_user"],mode="single")
                    self.mean_avg_prec_map[baseline]["Homogeneous_user"]["map"]={"avg_precision": np.mean(avg_prescision),"c1_avg_precision":np.nan,"c2_avg_precision":np.nan}
            
            if baseline in ["4","5","7"]:
                
                if baseline == "7":
                    param_results = calculate_avg_precision_param_variation(self.scores_map[baseline]["Heterogeneous_user"],params=self.baseline_params_map[baseline],mode="mixed")
                    self.mean_avg_prec_map[baseline]["Heterogeneous_user"]["map"] = calculate_map_param_variation(param_results,mode="mixed")
                    param_results = calculate_avg_precision_param_variation(self.scores_map[baseline]["Homogeneous_user"],params=self.baseline_params_map[baseline],mode="mixed")
                    self.mean_avg_prec_map[baseline]["Homogeneous_user"]["map"] = calculate_map_param_variation(param_results,mode="mixed")
                else:
                    param_results = calculate_avg_precision_param_variation(self.scores_map[baseline]["Heterogeneous_user"],params=self.baseline_params_map[baseline],mode="single")
                    self.mean_avg_prec_map[baseline]["Heterogeneous_user"]["map"] = calculate_map_param_variation(param_results,mode="single")
                    param_results = calculate_avg_precision_param_variation(self.scores_map[baseline]["Homogeneous_user"],params=self.baseline_params_map[baseline],mode="single")
                    self.mean_avg_prec_map[baseline]["Homogeneous_user"]["map"] = calculate_map_param_variation(param_results,mode="single")

    
    def _save_scores_(self):
        """
        df with columsn = ["Baseline", "Param_Setting","MAP","MAP_C1","MAP_C2"]
        """
        baselines = ["2","3","4","5","6","7"]
        map_res_homo = []
        map_res_hetero = []
        param_val = []
        baseline_col = []
        
        # saving results without param variation
        
        for baseline in ["2","3","6"]:
            map_res_homo.append(self.mean_avg_prec_map[baseline]["Homogeneous_user"]["map"])
#             map_res_homo += list(self.mean_avg_prec_map[baseline]["Homogeneous_user"]["map"].items())
            map_res_hetero.append(self.mean_avg_prec_map[baseline]["Heterogeneous_user"]["map"])
            baseline_col.append(baseline)
            param_val.append("NA")
        
        # saving results with param variation
        for baseline in ["4","5","7"]:
            for param in self.mean_avg_prec_map[baseline]["Homogeneous_user"]["map"].keys():
                baseline_col.append(baseline)
                param_val.append(str(param))
                map_res_homo.append(self.mean_avg_prec_map[baseline]["Homogeneous_user"]["map"][param])
                map_res_hetero.append(self.mean_avg_prec_map[baseline]["Heterogeneous_user"]["map"][param])
        
        # from our lists of dictionaries (ie {map,map_c1,map_c2})
        score_df_homo = pd.DataFrame(map_res_homo)
        score_df_homo["Baseline"] = baseline_col
        score_df_homo["Param_setting"] = param_val
        score_df_homo.rename({"avg_precision":"MAP","c1_avg_precision":"MAP_C1","c2_avg_precision":"MAP_C2"},inplace=True)
        
        score_df_hetero = pd.DataFrame(map_res_hetero)
        score_df_hetero["Baseline"] = baseline_col
        score_df_hetero["Param_setting"] = param_val
        score_df_hetero.rename({"avg_precision":"MAP","c1_avg_precision":"MAP_C1","c2_avg_precision":"MAP_C2"},inplace=True)
        
#         score_df_homo = score_df_homo[["Baseline", "Param_Setting","MAP","MAP_C1","MAP_C2"]]
#         score_df_hetero = score_df_hetero[["Baseline", "Param_Setting","MAP","MAP_C1","MAP_C2"]]
        
        score_df_homo.to_csv("Baseline_results_%s_homo.csv"%self.rep_type ,index=False)
        score_df_hetero.to_csv("Baseline_results_%s_hetero.csv"%self.rep_type ,index=False)
        print("\nFinished Saving Results ..............")
                
                
                
    def _plot_baseline_results(self):
        """
        """
        pass
        
    
    def _save_2_sheets_(self,):
        """
        """
        pass


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p",
                        "--path", 
                        type=str,
                        default="../sampled_articles_from_relevant_data.csv",
                        help= "The path where the data csv is saved")
    
    parser.add_argument("-b",
                        "--baseline", 
                        type=str,
                        default="all",
                        help= "The baselines to run")
    
    parser.add_argument("-r",
                        "--rep_type",
                        type=str,
                        default="tf-idf",
                        help= "The representation type to run the baselines, (eg: tf-idf,glove, bert_12_100)")
    
    parser.add_argument("-pl",
                        "--plot", 
                        type=bool, 
                        default=False,
                        help="Do you want the plots for the baselines")
    
    args = parser.parse_args()
    
    print("Settings")
    for c in CONFIG.__dict__:
        if "__" not in c and c not in ["np","os","copy"]:
            print("%s :: %s"%(c,CONFIG.__dict__[c]))
    
    br = BaselineRunner(file_path=args.path,
                        baseline_type=args.baseline,
                        rep_type=args.rep_type,
                        plot=args.plot)
    
    br.run()