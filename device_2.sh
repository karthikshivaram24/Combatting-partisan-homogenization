#!/bin/bash

# device 2
python attm_run_ssda.py -cp "(22, 58)" -d "1" -t "single" -a True > log2_1.txt
python attm_run_ssda.py -cp "(22, 58)" -d "1" -t "single" -a False > log2_2.txt
python attm_run_ssda.py -cp "(22, 58)" -d "1" -t "multi" -a True > log2_3.txt