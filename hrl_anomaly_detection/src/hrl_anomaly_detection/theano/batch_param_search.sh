#!/bin/bash -x


eta_list=(1e-3 1e-4 1e-5)
etadecay_list=(1e-5)
momentum_list=(1e-6 1e-5 1e-4 1e-3)
lambda_list=(1e-6 1e-5 1e-4 1e-3)
timewindow_list=(2 4)
batch_list=(1)

layersize='[20,10,5]'
FOLDER_NAME=/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/pushing_data/${layersize}
mkdir -p $FOLDER_NAME

for i in "${eta_list[@]}"
do
    for j in "${momentum_list[@]}"
    do
        for k in "${lambda_list[@]}"
        do
            for l in "${timewindow_list[@]}"
            do
                for m in "${batch_list[@]}"
                do

                    FILENAME=${FOLDER_NAME}/E_${i}_M_${j}_L_${k}_TW_${l}_b_${m}.log

                    python test.py --train --ls ${layersize} --lr ${i} --m ${j} --lambda ${k} --tw ${l} --batch_size ${m} --mi 1500 >> $FILENAME

                done
            done
        done
    done
done


