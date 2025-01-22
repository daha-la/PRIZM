#!/bin/bash 

source ../zero_shot_config.sh
#source activate proteingym_env

cd ../../proteingym/

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i
#"Experiment index to run (e.g. 0,1,2,...216)" >216 = Inhouse

    start_time=$(date +%s.%N)


    export RITA_model_path="/data/checkpoints/RITA/RITA_s"
    export output_scores_folder="${DMS_output_score_folder_subs}/RITA/small"
    
    python baselines/rita/compute_fitness.py \
                --RITA_model_name_or_path ${RITA_model_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --output_scores_folder ${output_scores_folder}
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with Small: $elapsed_time seconds"
 
    start_time=$(date +%s.%N)
    export RITA_model_path="/data/checkpoints/RITA/RITA_m"
    export output_scores_folder="${DMS_output_score_folder_subs}/RITA/medium"
    
    python baselines/rita/compute_fitness.py \
                --RITA_model_name_or_path ${RITA_model_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --output_scores_folder ${output_scores_folder}
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with Medium: $elapsed_time seconds"

    start_time=$(date +%s.%N)
    export RITA_model_path="/data/checkpoints/RITA/RITA_l"
    export output_scores_folder="${DMS_output_score_folder_subs}/RITA/large"
    
    python baselines/rita/compute_fitness.py \
                --RITA_model_name_or_path ${RITA_model_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --output_scores_folder ${output_scores_folder}
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with Large: $elapsed_time seconds"

    start_time=$(date +%s.%N)
    export RITA_model_path="/data/checkpoints/RITA/RITA_xl"
    export output_scores_folder="${DMS_output_score_folder_subs}/RITA/xlarge"
    
    python baselines/rita/compute_fitness.py \
                --RITA_model_name_or_path ${RITA_model_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --output_scores_folder ${output_scores_folder}

    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with Xlarge: $elapsed_time seconds"
done

