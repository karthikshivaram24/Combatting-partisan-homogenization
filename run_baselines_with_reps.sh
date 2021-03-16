#!/bin/bash

# conda activate frames

# python baseline_runner.py -r tf-idf
python baseline_runner.py -r glove
python baseline_runner.py -r bert_12_100
python baseline_runner.py -r bert_12_0
python baseline_runner.py -r bert_11_100
python baseline_runner.py -r bert_1_100

# combining vectors
python baseline_runner.py -r bert_12_100_tf_idf
python baseline_runner.py -r bert_1_100_tf_idf
python baseline_runner.py -r bert_11_100_tf_idf
python baseline_runner.py -r bert_12_0_tf_idf