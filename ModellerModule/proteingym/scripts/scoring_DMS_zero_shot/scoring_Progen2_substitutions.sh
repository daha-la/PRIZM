#!/bin/bash 

source ../zero_shot_config.sh

cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i

    start_time=$(date +%s.%N)
    export Progen2_model_name_or_path="$checkpoint_folder/Progen2/progen2-small"
    export output_scores_folder="${DMS_output_score_folder_subs}/Progen2/small"
    
    python baselines/progen2/compute_fitness.py \
                --Progen2_model_name_or_path ${Progen2_model_name_or_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --output_scores_folder ${output_scores_folder} 
    
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with Small: $elapsed_time seconds"
    
    start_time=$(date +%s.%N)
    export Progen2_model_name_or_path="$checkpoint_folder/Progen2/progen2-medium"
    export output_scores_folder="${DMS_output_score_folder_subs}/Progen2/medium"
    
    python baselines/progen2/compute_fitness.py \
                --Progen2_model_name_or_path ${Progen2_model_name_or_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --output_scores_folder ${output_scores_folder} 
    
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with Medium: $elapsed_time seconds"

    start_time=$(date +%s.%N)
    export Progen2_model_name_or_path="$checkpoint_folder/Progen2/progen2-base"
    export output_scores_folder="${DMS_output_score_folder_subs}/Progen2/base"
    
    python baselines/progen2/compute_fitness.py \
                --Progen2_model_name_or_path ${Progen2_model_name_or_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --output_scores_folder ${output_scores_folder} 
    
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with Base: $elapsed_time seconds"

    start_time=$(date +%s.%N)
    export Progen2_model_name_or_path="$checkpoint_folder/Progen2/progen2-large"
    export output_scores_folder="${DMS_output_score_folder_subs}/Progen2/large"
    
    python baselines/progen2/compute_fitness.py \
                --Progen2_model_name_or_path ${Progen2_model_name_or_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --output_scores_folder ${output_scores_folder} 
    
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with Large: $elapsed_time seconds"

done
