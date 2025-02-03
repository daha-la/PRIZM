#!/bin/bash 

source ../zero_shot_config.sh

export model_checkpoint="$PRIZM_PATH/ModellerModule/checkpoints/esm/esm1b_t33_650M_UR50S.pt"

export dms_output_folder="${DMS_output_score_folder_subs}/ESM1b/"

export model_type="ESM1b"
export scoring_strategy="wt-marginals"
export scoring_window="overlapping"

cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i

    start_time=$(date +%s.%N)

    python baselines/esm/compute_fitness.py --model-location ${model_checkpoint} --model_type ${model_type} --dms_index ${DMS_index} --dms_mapping ${DMS_reference_file_path_subs} --dms-input ${DMS_data_folder_subs} --dms-output ${dms_output_folder} --scoring-strategy ${scoring_strategy} --scoring-window ${scoring_window}
    
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i: $elapsed_time seconds"
done
