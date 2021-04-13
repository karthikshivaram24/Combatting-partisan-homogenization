#!/bin/bash

# device 0
# screen -dmS device_0 bash -c 'device_0.sh'
# screen -dmS device_1 bash -c 'device_1.sh'
# screen -dmS device_2 bash -c 'device_2.sh'
# screen -dmS device_3 bash -c 'device_3.sh'

# make sure scripts are executable

screen -S device_0  -dm ./device_0.sh
screen -S device_1  -dm ./device_1.sh
screen -S device_2  -dm ./device_2.sh
screen -S device_3  -dm ./device_3.sh