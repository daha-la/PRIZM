#!/bin/bash 

source ../zero_shot_config.sh
#source activate proteingym_env

# ESM-1v parameters 
# Five checkpoints for ESM-1v
export model_checkpoint1="/data/checkpoints/esm/esm1v_t33_650M_UR90S_1.pt"
export model_checkpoint2="/data/checkpoints/esm/esm1v_t33_650M_UR90S_2.pt"
export model_checkpoint3="/data/checkpoints/esm/esm1v_t33_650M_UR90S_3.pt"
export model_checkpoint4="/data/checkpoints/esm/esm1v_t33_650M_UR90S_4.pt"
export model_checkpoint5="/data/checkpoints/esm/esm1v_t33_650M_UR90S_5.pt"
# combine all five into one string 
export model_checkpoint="${model_checkpoint1} ${model_checkpoint2} ${model_checkpoint3} ${model_checkpoint4} ${model_checkpoint5}"
export dms_output_folder="${DMS_output_score_folder_subs}/ESM1v/"

export model_type="ESM1v"
export scoring_strategy="masked-marginals"  # MSATransformer only uses masked-marginals
export scoring_window="optimal"
#"Experiment index to run (e.g. 0,1,2,...216)" >216 = Inhouse
cd ../../proteingym

echo $DMS_reference_file_path_subs
for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i
#"Experiment index to run (e.g. 0,1,2,...216)" >216 = Inhouse

    start_time=$(date +%s.%N)


    python baselines/esm/compute_fitness.py \
        --model-location ${model_checkpoint} \
        --model_type ${model_type} \
        --dms_index ${DMS_index} \
        --dms_mapping ${DMS_reference_file_path_subs} \
        --dms-input ${DMS_data_folder_subs} \
        --dms-output ${dms_output_folder} \
        --scoring-strategy ${scoring_strategy} \
        --scoring-window ${scoring_window}
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i: $elapsed_time seconds"
done

