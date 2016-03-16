#!/bin/bash -x


lr_list=(1e-6)
lrdecay_list=(1e-6)
momentum_list=(1e-7)
dampening_list=(1e-7)
lambda_list=(1e-6 1e-7)
timewindow_list=(4)
batch_list=(1)
layersize_list=('[256,128,16]' '[256,128,8]')
maxiter=10000

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

                                PARAMS=E_${i}_${ii}_M_${j}_${jj}_L_${k}_TW_${l}_b_${m}
                                FILENAME=${FOLDER_NAME}/${PARAMS}.log

                                IN=$(THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py --train --ls ${n} --lr ${i} --lrd ${ii} --m ${j} --d ${jj} --lambda ${k} --tw ${l} --batch_size ${m} --mi $maxiter 2>&1 > $FILENAME )
                                #python test.py --train --ls ${n} --lr ${i} --m ${j} --lambda ${k} --tw ${l} --batch_size ${m} --mi 300 >> $FILENAME

                                set -- "$IN" 
                                IFS=">"; declare -a Array=($*) 
                                echo "${Array[1]}" 


                                NEWFILENAME=${FOLDER_NAME}/${Array[1]}_${PARAMS}.log
                                mv $FILENAME $NEWFILENAME

                            done
                        done
                    done
                done
            done
        done
    done
done


