#!/bin/bash

source ../zero_shot_config.sh

export seed=42
#"random seed value"

export model_parameters_location='/home/dahala/mnt/ZeroShot/ProteinGym_code/proteingym/baselines/EVE/EVE/default_model_params.json'
export training_logs_location='/home/dahala/mnt/ZeroShot/ProteinGym_code/proteingym/baselines/EVE/logs'
export DMS_reference_file_path=$DMS_reference_file_path_subs
# export DMS_reference_file_path=$DMS_reference_file_path_indels
export overwrite=True
echo $DMS_reference_file_path_subs

cd ../../proteingym/

for ((i=$1; i<=$2; i++))
do
    echo "Training on DMS index $i"
    export DMS_index=$i
#"Experiment index to run (e.g. 0,1,2,...216)" >216 = Inhouse

    start_time=$(date +%s.%N)

    python baselines/EVE/train_VAE.py \
        --MSA_data_folder ${DMS_MSA_data_folder} \
        --DMS_reference_file_path ${DMS_reference_file_path} \
        --protein_index "${DMS_index}" \
        --MSA_weights_location ${DMS_MSA_weights_folder} \
        --VAE_checkpoint_location ${DMS_EVE_model_folder} \
        --model_parameters_location ${model_parameters_location} \
        --training_logs_location ${training_logs_location} \
        --threshold_focus_cols_frac_gaps 1 \
        --seed ${seed} \
        --skip_existing \
        --experimental_stream_data \
        #--overwrite_weights ${overwrite}
        #--force_load_weights
    
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i: $elapsed_time seconds"
done

