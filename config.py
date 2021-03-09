import numpy as np
import copy

print("....... Initializing Settings ..... ")

np.random.seed(24101990)
RANDOM_SEED = np.random.randint(low=1, high=100000, size=1)[0]
print("Random_Seed Chosen : %s" %str(RANDOM_SEED))

ROOT_FOLDER = ""

#Clustering Settings
num_clusters = 100
min_cluster_size = 300
max_cluster_size = 10000
min_partisan_size = 0.4

# TFIDF Settings
min_df = 30
max_df = 0.75

lr_params = [0.001,0.01,0.1,1.0,10,15,20,50,100,500]
rc_params = [0.0001,0.001,0.01,0.1,0.0,1.0,10.0,20.0,50.0,100.0]

# EMBEDDING FILES
GLOVE_PATH = ROOT_FOLDER + "Embeddings/glove.6B.300d.magnitude"
W2V_PATH = ROOT_FOLDER + "Embeddings/GoogleNews-vectors-negative300.magnitude"
FASTTEXT_PATH = ROOT_FOLDER + "Embeddings/wiki-news-300d-1M-subword.magnitude"
ELMO_PATH = ROOT_FOLDER + "Embeddings/elmo_2x2048_256_2048cnn_1xhighway_weights_GoogleNews_vocab.magnitude"
