#!/bin/bash 

source ../zero_shot_config.sh

export OMP_NUM_THREADS=1

export model_path="$PRIZM_PATH/ModellerModule/checkpoints/UniRep/1900_weights"
export output_dir=${DMS_output_score_folder_subs}/UniRep

cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i

    start_time=$(date +%s.%N)

    python baselines/unirep/unirep_inference.py \
                --model_path $model_path \
                --data_path $DMS_data_folder_subs \
                --output_dir $output_dir \
                --mapping_path $DMS_reference_file_path_subs \
                --DMS_index $DMS_index \
                --batch_size 8
    
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i: $elapsed_time seconds"

done
