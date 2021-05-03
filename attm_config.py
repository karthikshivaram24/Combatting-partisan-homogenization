import numpy as np
np.random.seed(24)

def random_search_lr(n,max_lr=0.001,min_lr=0.0000001):
    """
    """
    range_size = (max_lr - min_lr)  # 2
    lrs = (np.random.rand(n) * range_size + min_lr).tolist()
    return lrs

# lr = [0.001,0.0001,0.00001]
lr = [0.001,0.0001]
word_weights = [0.1,0.4,0.8]
print("Learning Rates Used : \n%s"%str(lr))
epochs = [10]
batch_size = [16,32]
dropouts = [0.1,0.3,0.5]
pickle_file_path = "att_pickle_objs_input"
save_pickle_results_path = "att_pickle_objs_results"     