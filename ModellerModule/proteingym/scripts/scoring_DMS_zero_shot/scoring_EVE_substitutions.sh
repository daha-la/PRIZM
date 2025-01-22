#!/bin/bash

source ../zero_shot_config.sh
#source activate proteingym_env

export model_parameters_location='/home/dahala/mnt/ZeroShot/ProteinGym_code/proteingym/baselines/EVE/EVE/default_model_params.json'
export training_logs_location='/home/dahala/mnt/ZeroShot/ProteinGym_code/proteingym/baselines/EVE/logs/'
export computation_mode='DMS'
export output_score_folder="${DMS_output_score_folder_subs}/EVE/"
export num_samples_compute_evol_indices=20000
export batch_size=1024  # Pushing batch size to limit of GPU memory
export random_seeds="42"
#"0 1 2 3 4"

cd ../../proteingym/

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i
#"Experiment index to run (e.g. 0,1,2,...216)" >216 = Inhouse

    start_time=$(date +%s.%N)



    python baselines/EVE/compute_evol_indices_DMS.py \
        --MSA_data_folder ${DMS_MSA_data_folder} \
        --DMS_reference_file_path ${DMS_reference_file_path_subs} \
        --protein_index ${DMS_index} \
        --VAE_checkpoint_location ${DMS_EVE_model_folder} \
        --model_parameters_location ${model_parameters_location} \
        --DMS_data_folder ${DMS_data_folder_subs} \
        --output_scores_folder ${output_score_folder} \
        --num_samples_compute_evol_indices ${num_samples_compute_evol_indices} \
        --batch_size ${batch_size} \
        --aggregation_method "full" \
        --threshold_focus_cols_frac_gaps 1 \
        --MSA_weights_location ${DMS_MSA_weights_folder} \
        --random_seeds ${random_seeds} 
    #changed from --output_evol_indices_location

    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i: $elapsed_time seconds"
done

