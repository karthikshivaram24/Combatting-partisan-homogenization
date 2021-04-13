#!/bin/bash

# device 0
python attm_run_ssda.py -cp "(14, 44)" -d "0" -t "single" -a True 
python attm_run_ssda.py -cp "(14, 44)" -d "0" -t "single" -a False 
python attm_run_ssda.py -cp "(14, 44)" -d "0" -t "multi" -a True 