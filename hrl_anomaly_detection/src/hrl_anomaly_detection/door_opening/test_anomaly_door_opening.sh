#!/bin/bash                                                                                                       
#su dpark -c ". ~/.bashrc ; rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv" 
                      
#echo "uid is ${UID}"                                                                                        
#echo "user is ${USER}"                                                                                        
#echo "username is ${USERNAME}"
                                                                                                                 
 
#su dpark -c 'echo "uidr is ${UID}"'                                                                             
#su dpark -c 'echo "user is ${USER}"'                                                                            
#su dpark -c 'echo "username is ${USERNAME}"'                                         
                            
#su dpark -c 'source ~/.profile ; touch ~/hrl_file_server/dpark_data/anomaly/RSS2015/test.txt'     
               
sleep 5

su dpark -c 'source ~/.profile ; rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv' 

sleep 5.0

su dpark -c 'source ~/.profile ; rosrun hrl_anomaly_detection test_anomaly_door_opening.py --fig_roc_human' 

#sleep 5.0
#su dpark -c 'source ~/.profile ; rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &' 
#sleep 5.0
#su dpark -c 'source ~/.profile ; rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv ' 
#sleep 5.0

