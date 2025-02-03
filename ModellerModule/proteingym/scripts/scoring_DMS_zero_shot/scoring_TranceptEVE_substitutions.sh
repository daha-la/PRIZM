#!/bin/bash 

source ../zero_shot_config.sh

export inference_time_retrieval_type="TranceptEVE"
export EVE_num_samples_log_proba=200000 
export EVE_model_parameters_location="$PRIZM_PATH/ModellerModule/proteingym/baselines/trancepteve/trancepteve/utils/eve_model_default_params.json"
export EVE_seeds="42"
#"0 1 2 3 4"

export scoring_window="optimal" 

cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i

    start_time=$(date +%s.%N)
    export checkpoint="$PRIZM_PATH/ModellerModule/checkpoints/Tranception/Tranception_Small"
    export output_scores_folder=${DMS_output_score_folder_subs}/TranceptEVE/TranceptEVE_S
    
    python baselines/trancepteve/score_trancepteve.py \
                    --checkpoint ${checkpoint} \
                    --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                    --DMS_data_folder ${DMS_data_folder_subs} \
                    --DMS_index ${DMS_index} \
                    --output_scores_folder ${output_scores_folder} \
                    --inference_time_retrieval_type ${inference_time_retrieval_type} \
                    --MSA_folder ${DMS_MSA_data_folder} \
                    --MSA_weights_folder ${DMS_MSA_weights_folder} \
                    --EVE_num_samples_log_proba ${EVE_num_samples_log_proba} \
                    --EVE_model_parameters_location ${EVE_model_parameters_location} \
                    --EVE_model_folder ${DMS_EVE_model_folder} \
                    --scoring_window ${scoring_window} \
                    --EVE_seeds ${EVE_seeds} \
                    --EVE_recalibrate_probas
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with Small: $elapsed_time seconds"

    start_time=$(date +%s.%N)
    export checkpoint="$PRIZM_PATH/ModellerModule/checkpoints/Tranception/Tranception_Medium"
    export output_scores_folder=${DMS_output_score_folder_subs}/TranceptEVE/TranceptEVE_M
    
    python baselines/trancepteve/score_trancepteve.py \
                    --checkpoint ${checkpoint} \
                    --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                    --DMS_data_folder ${DMS_data_folder_subs} \
                    --DMS_index ${DMS_index} \
                    --output_scores_folder ${output_scores_folder} \
                    --inference_time_retrieval_type ${inference_time_retrieval_type} \
                    --MSA_folder ${DMS_MSA_data_folder} \
                    --MSA_weights_folder ${DMS_MSA_weights_folder} \
                    --EVE_num_samples_log_proba ${EVE_num_samples_log_proba} \
                    --EVE_model_parameters_location ${EVE_model_parameters_location} \
                    --EVE_model_folder ${DMS_EVE_model_folder} \
                    --scoring_window ${scoring_window} \
                    --EVE_seeds ${EVE_seeds} \
                    --EVE_recalibrate_probas
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with Medium: $elapsed_time seconds"

    start_time=$(date +%s.%N)
    export checkpoint="$PRIZM_PATH/ModellerModule/checkpoints/Tranception/Tranception_Large"
    export output_scores_folder=${DMS_output_score_folder_subs}/TranceptEVE/TranceptEVE_L
    
    python baselines/trancepteve/score_trancepteve.py \
                    --checkpoint ${checkpoint} \
                    --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                    --DMS_data_folder ${DMS_data_folder_subs} \
                    --DMS_index ${DMS_index} \
                    --output_scores_folder ${output_scores_folder} \
                    --inference_time_retrieval_type ${inference_time_retrieval_type} \
                    --MSA_folder ${DMS_MSA_data_folder} \
                    --MSA_weights_folder ${DMS_MSA_weights_folder} \
                    --EVE_num_samples_log_proba ${EVE_num_samples_log_proba} \
                    --EVE_model_parameters_location ${EVE_model_parameters_location} \
                    --EVE_model_folder ${DMS_EVE_model_folder} \
                    --scoring_window ${scoring_window} \
                    --EVE_seeds ${EVE_seeds} \
                    --EVE_recalibrate_probas \
                    --batch_size_inference 10
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with Large: $elapsed_time seconds"
    
done
