import pickle
import pandas as pd
from config import RANDOM_SEED
from data_utils import sample_data
from preprocess_utils import preprocess_texts


def load_pkl_file(file_path):
    """
    """
    article_dict = pickle.load(open(file_path,"rb"))
    articles = []
    for article in article_dict.keys():
        articles.append(article_dict[article])
    
    print("\n ********** Original Dataset ******************")
    print("\nNumber of Articles : %s"%str(len(articles)))
    articles_df = pd.DataFrame(articles)
    
    # We want the distribution of stances and sources
    print("\nStance Distribution :")
    print(articles_df["source_partisan_score"].value_counts())
    
    print("\nNew's Source Distribution :")
    print(articles_df["source"].value_counts())
    
    print("Shape Before Processing : %s" %str(articles_df.shape))
    #drop columns
    articles_df.drop(columns=["article_id",
                              "url",
                              "tweet_id",
                              "tweet_text",
                              "kws_label",
                              "cls_label",
                              "tweet_screen_name",
                              "tweet_created_at"],inplace=True)
    #reset index
    articles_df.reset_index(inplace=True,drop=True)
    #drop partisan of 0.0
    articles_df = articles_df.loc[articles_df["source_partisan_score"] != 0.0]
    
    sampled_df = sample_data(df=articles_df,sample_size=100000,seed=RANDOM_SEED)
    
    print("\n ********* Sampled Dataset ********************")
    print("\nNumber of Articles : %s"%str(sampled_df.shape[0]))
    print("\nStance Distribution :")
    print(sampled_df["source_partisan_score"].value_counts())
    print("\nNew's Source Distribution :")
    print(sampled_df["source"].value_counts())
    
    return sampled_df
    
#     articles_df["binary_ps"] = articles_df["source_partisan_score"].apply(lambda x: 1 if x>0 else 0)
    return articles_df

def convert_stances_extreme(articles_df):
    """
    """
    articles_df.drop(articles_df.loc[articles_df['source_partisan_score']==1.0].index, inplace=True)
    articles_df.drop(articles_df.loc[articles_df['source_partisan_score']==-1.0].index, inplace=True)
    
    articles_df["binary_ps"] = articles_df["source_partisan_score"].apply(lambda x: 1 if x>0 else 0)
    print("\n ********* Sampled Dataset Extreme Stances Only ********************")
    print("\nNumber of Articles : %s"%str(articles_df.shape[0]))
    print("\nStance Distribution :")
    print(articles_df["source_partisan_score"].value_counts())
    print("\nNew's Source Distribution :")
    print(articles_df["source"].value_counts())
    
    articles_df["processed_text"] = preprocess_texts(text_lists=articles_df["text"].tolist())
    
    articles_df.reset_index(inplace=True,drop=True)
    
    return articles_df


if __name__ == "__main__":
    
    sampled_df = load_pkl_file("../labeled_political_articles.pkl")
    article_df = convert_stances_extreme(sampled_df)
    
    article_df.to_csv("../sampled_articles_from_relevant_data_extreme.csv",index=False)