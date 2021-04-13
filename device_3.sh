#!/bin/bash

# device 3
python attm_run_ssda.py -cp "(22, 36)" -d "3" -t "single" -a True > log3_1.txt
python attm_run_ssda.py -cp "(22, 36)" -d "3" -t "single" -a False > log3_2.txt
python attm_run_ssda.py -cp "(22, 36)" -d "3" -t "multi" -a True > log3_3.txt