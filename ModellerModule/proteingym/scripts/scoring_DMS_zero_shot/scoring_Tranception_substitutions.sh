#!/bin/bash 

source ../zero_shot_config.sh

cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i

    start_time=$(date +%s.%N)
    export checkpoint="$checkpoint_folder/Tranception/Tranception_Small"
    export output_scores_folder=${DMS_output_score_folder_subs}/Tranception/Tranception_S
    
    python baselines/tranception/score_tranception_proteingym.py \
                    --checkpoint ${checkpoint} \
                    --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                    --DMS_data_folder ${DMS_data_folder_subs} \
                    --DMS_index ${DMS_index} \
                    --output_scores_folder ${output_scores_folder} \
                    --inference_time_retrieval \
                    --MSA_folder ${DMS_MSA_data_folder} \
                    --MSA_weights_folder ${DMS_MSA_weights_folder}
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with Small: $elapsed_time seconds"

    start_time=$(date +%s.%N)
    
    export checkpoint="$checkpoint_folders/Tranception/Tranception_Medium"
    export output_scores_folder=${DMS_output_score_folder_subs}/Tranception/Tranception_M
    
    python baselines/tranception/score_tranception_proteingym.py \
                    --checkpoint ${checkpoint} \
                    --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                    --DMS_data_folder ${DMS_data_folder_subs} \
                    --DMS_index ${DMS_index} \
                    --output_scores_folder ${output_scores_folder} \
                    --inference_time_retrieval \
                    --MSA_folder ${DMS_MSA_data_folder} \
                    --MSA_weights_folder ${DMS_MSA_weights_folder}
    
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with Medium: $elapsed_time seconds"

    start_time=$(date +%s.%N)

    export checkpoint="$checkpoint_folder/Tranception/Tranception_Large"
    export output_scores_folder=${DMS_output_score_folder_subs}/Tranception/Tranception_L
    
    python baselines/tranception/score_tranception_proteingym.py \
                    --checkpoint ${checkpoint} \
                    --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                    --DMS_data_folder ${DMS_data_folder_subs} \
                    --DMS_index ${DMS_index} \
                    --output_scores_folder ${output_scores_folder} \
                    --inference_time_retrieval \
		    --batch_size_inference 5 \
                    --MSA_folder ${DMS_MSA_data_folder} \
                    --MSA_weights_folder ${DMS_MSA_weights_folder}
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with Large: $elapsed_time seconds"
    
done
    
