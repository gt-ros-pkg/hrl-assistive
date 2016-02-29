#!/bin/bash -x


lr_list=(1e-3 1e-4 1e-5)
lrdecay_list=(1e-5)
momentum_list=(1e-6)
dampening_list=(1e-6)
lambda_list=(1e-6)
timewindow_list=(4)
batch_list=(1)
layersize_list=('[256,128]')

for n in "${layersize_list[@]}"
do
    FOLDER_NAME=/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/pushing_data/${n}
    mkdir -p $FOLDER_NAME

    for i in "${lr_list[@]}"
    do
        for ii in "${lrdecay_list[@]}"
        do
            for j in "${momentum_list[@]}"
            do
                for jj in "${dampening_list[@]}"
                do
                    for k in "${lambda_list[@]}"
                    do
                        for l in "${timewindow_list[@]}"
                        do
                            for m in "${batch_list[@]}"
                            do

                                FILENAME=${FOLDER_NAME}/E_${i}_${ii}_M_${j}_${jj}_L_${k}_TW_${l}_b_${m}.log

                                THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python test.py --train --ls ${n} --lr ${i} --lrd ${ii} --m ${j} --d ${jj} --lambda ${k} --tw ${l} --batch_size ${m} --mi 3000 >> $FILENAME
                                #python test.py --train --ls ${n} --lr ${i} --m ${j} --lambda ${k} --tw ${l} --batch_size ${m} --mi 300 >> $FILENAME

                            done
                        done
                    done
                done
            done
        done
    done
done


