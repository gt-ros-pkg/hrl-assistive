#!/bin/bash

#kill -9 `ps aux | grep gazebo | awk '{print $2}'`
#sleep 1

#SAVE_PATH=N1=/home/$USER/hrl_file_server/dpark_data/

#if [ "$1" -eq "renew" ] ; then    
#fi

if [ $# -lt 2 ]; then
    HMM_RENEW=0
elif [ "$2" = "hmm_renew" ]; then
    HMM_RENEW=1
else
    HMM_RENEW=0
fi
echo "HMM renew is " $HMM_RENEW
#exit 2

for ((I=2;I<=5;I++)); do
    echo ${I}
    if [ $I -eq 5 ]; then 
        if ["$1" = "scooping"] || ["$1" = "feeding"]; then
            return
        fi 
    fi

    if [ $HMM_RENEW -eq 1 ]; then
        python ../src/hrl_anomaly_detection/rss2016test.py --task $1 --dim $I --ea --hr --np;
    fi

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
