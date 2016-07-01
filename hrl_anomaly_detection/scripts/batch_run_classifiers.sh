#!/bin/bash

#kill -9 `ps aux | grep gazebo | awk '{print $2}'`
#sleep 1

#SAVE_PATH=N1=/home/$USER/hrl_file_server/dpark_data/

#if [ "$1" -eq "renew" ] ; then
    
#fi

for ((I=2;I<=5;I++)); do
    echo ${I}
    if [ $I -eq 5 ]; then 
        if ["$1" -eq "scooping"] || ["$1" -eq "feeding"]; then
            return;;
        fi 
    fi

    python ../src/hrl_anomaly_detection/rss2016test.py --task $1 --dim $I --ea --hr --np;

    if [ $I -eq 2 ]; then
        METHOD=('svm' 'hmmosvm');
    elif [ $I -eq 3 ]; then        
        METHOD=('svm' 'hmmosvm' );
    elif [ $I -eq 4 ]; then        
        METHOD=('svm' 'hmmosvm' 'hmmsvm_diag' 'hmmsvm_dL' 'hmmsvm_LSLS' );
    elif [ $I -eq 5 ]; then        
        METHOD=('svm' 'hmmosvm' );
    fi

    for method in "${METHOD[@]}"
    do
        python ../src/hrl_anomaly_detection/classifiers/run_classifier_aws.py --task $1 --dim $I --save --method ${method} \;
    done

done
