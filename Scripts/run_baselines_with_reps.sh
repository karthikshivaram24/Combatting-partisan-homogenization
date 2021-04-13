#!/bin/bash

# rerun glove and bert_1_100
# conda activate frames

python baseline_runner.py -r tf-idf 

# python baseline_runner.py -r glove -p ../sampled_articles_from_relevant_data_extreme.csv
python baseline_runner.py -r bert_12_100 
python baseline_runner.py -r bert_12_0 
python baseline_runner.py -r bert_11_100 
# python baseline_runner.py -r bert_1_100 -p ../sampled_articles_from_relevant_data_extreme.csv

# combining vectors
python baseline_runner.py -r bert_12_100_tf_idf 
python baseline_runner.py -r bert_1_100_tf_idf 
python baseline_runner.py -r bert_11_100_tf_idf 
python baseline_runner.py -r bert_12_0_tf_idf 