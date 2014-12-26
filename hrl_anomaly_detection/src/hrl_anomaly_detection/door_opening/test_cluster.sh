#!/bin/bash                                                                                                                      
nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
sleep 3.0

nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
sleep 3.0

nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
sleep 3.0

#nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
#sleep 3.0
