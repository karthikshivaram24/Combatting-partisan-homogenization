import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_name,env_name="Frames",timeout_flag=-1):
    """
    """
    with open(notebook_name) as rp:
        nb = nbformat.read(rp, as_version=4)
        ep = ExecutePreprocessor(timeout=timeout_flag, kernel_name=env_name)
        ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})
        
        with open(notebook_name, 'w', encoding='utf-8') as wp:
            nbformat.write(nb, wp)


if __name__=="__main__()":
    
    notebooks = ["Bert_layer_2_results.ipynb","Bert_layer_3_results.ipynb","Bert_layer_10_results.ipynb"]
    
    for n in notebooks:
        run_notebook(notebook_name=n)
