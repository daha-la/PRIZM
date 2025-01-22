#!/bin/bash 

source ../zero_shot_config.sh

export ProtGPT2_model_name_or_path="/data/checkpoints/ProtGPT2"
export output_scores_folder="${DMS_output_score_folder_subs}/ProtGPT2"
cd ../../proteingym

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i

    start_time=$(date +%s.%N)
    
    python baselines/protgpt2/compute_fitness.py \
                --ProtGPT2_model_name_or_path ${ProtGPT2_model_name_or_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --output_scores_folder ${output_scores_folder}

    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i: $elapsed_time seconds"
done

