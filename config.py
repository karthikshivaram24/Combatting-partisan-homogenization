import numpy as np

print("....... Initializing Settings ..... ")

np.random.seed(24101990)
RANDOM_SEED = np.random.randint(low=1, high=100000, size=1)[0]
print("Random_Seed Chosen : %s" %str(RANDOM_SEED))

ROOT_FOLDER = ""
EMBEDDING_TYPE = "Graphs/TFIDF/"


#Clustering Settings
NUM_CLUSTERS = 1000
CLUSTER_MIN_SIZE = 500
CLUSTER_MAX_SIZE = 5000
MIN_PARTISAN_SIZE = 0.4

# TFIDF Settings
MIN_DF = 30
MAX_DF = 0.75

# EMBEDDING FILES
GLOVE_PATH = ROOT_FOLDER + "Embeddings/glove.6B.300d.magnitude"
W2V_PATH = ROOT_FOLDER + "Embeddings/GoogleNews-vectors-negative300.magnitude"
FASTTEXT_PATH = ROOT_FOLDER + "Embeddings/wiki-news-300d-1M-subword.magnitude"
ELMO_PATH = ROOT_FOLDER + "Embeddings/elmo_2x2048_256_2048cnn_1xhighway_weights_GoogleNews_vocab.magnitude"