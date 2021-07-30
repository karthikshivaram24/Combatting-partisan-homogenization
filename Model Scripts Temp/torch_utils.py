import torch
import torch.nn as nn
import pickle
from time import time
import functools
import time

def timer(func):
    """
    Decorator to time a given function

    Parameters
    ----------
    func : generic
        The function to time

    Raises
    ------
    No Exceptions

    Returns
    -------
    value : generic
        The return value from func

    """
    @functools.wraps(func)
    def wrapper_timer(*args,**kwargs):
        start = time.perf_counter()
        value = func(*args,**kwargs)
        stop = time.perf_counter()
        run_time = stop - start
        print(f"\nFinished running {func.__name__!r} in {run_time/60.0:.4f} mins\n")
        return value
    return wrapper_timer


def load_pickle(file):
    """
    """
    obj = None
    with open(file,'rb') as handle:
        obj = pickle.load(handle)
    return obj

def save_obj(path,obj):
    """
    """
    with open(path,"wb") as wb:
        pickle.dump(obj,wb)

class PolarLoss(nn.Module):
    
    def __init__(self,bad_term_embeds):
        """
        """
        super(PolarLoss,self).__init__()
        self.bad_terms = bad_term_embeds.detach() # have to detach here to make sure this embedding term requires no gradient update as it is constant throughout the training phase
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,att_vec):
        """
        """
        # type 1: Professors code
#         batch_dot_prods = []
#         for attv in att_vec:
#             #print(attv.size()) # 1,768 vector
#             #print(self.bad_terms.size()) # 200,768 matrix
#             #print(torch.mul(attv,self.bad_terms).size()) # 200,768 matrix
#             #print(torch.mul(attv,self.bad_terms).sum(axis=1).size()) # 200 x1 vector
#             #print(torch.mul(attv,self.bad_terms).sum(axis=1).mean().size()) # 1x1 vector
            
#             dot_prods = self.sigmoid(torch.mul(attv,self.bad_terms).sum(axis=1).mean())
#             batch_dot_prods.append(dot_prods)
            
#         batch_dot_prods = torch.vstack(batch_dot_prods).mean()
        
        # type 2: My corrections
        batch_dot_prods = []
        
        for attv in att_vec:
            dot_prods = self.sigmoid(torch.mul(attv,self.bad_terms).sum(axis=1)).mean()
            batch_dot_prods.append(dot_prods)
        
        batch_dot_prods = torch.vstack(batch_dot_prods).mean()
        
        return batch_dot_prods