#!/bin/bash 

source ../zero_shot_config.sh

export model_type="ESM2"
export scoring_strategy="masked-marginals"

cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i

    start_time=$(date +%s.%N)

    export model_checkpoint="$checkpoint_folder/esm/esm2_t6_8M_UR50D.pt"
    export dms_output_folder=${DMS_output_score_folder_subs}/ESM2/8M
    
    python baselines/esm/compute_fitness.py \
        --model-location ${model_checkpoint} \
        --dms_index $DMS_index \
        --dms_mapping ${DMS_reference_file_path_subs} \
        --dms-input ${DMS_data_folder_subs} \
        --dms-output ${dms_output_folder} \
        --scoring-strategy ${scoring_strategy} \
        --model_type ${model_type} 

    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with 8M: $elapsed_time seconds"
    
    start_time=$(date +%s.%N)
    export model_checkpoint="$checkpoint_folder/esm/esm2_t12_35M_UR50D.pt"
    export dms_output_folder=${DMS_output_score_folder_subs}/ESM2/35M
    
    python baselines/esm/compute_fitness.py \
        --model-location ${model_checkpoint} \
        --dms_index $DMS_index \
        --dms_mapping ${DMS_reference_file_path_subs} \
        --dms-input ${DMS_data_folder_subs} \
        --dms-output ${dms_output_folder} \
        --scoring-strategy ${scoring_strategy} \
        --model_type ${model_type}
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with 35M: $elapsed_time seconds"

    start_time=$(date +%s.%N)
    export model_checkpoint="$checkpoint_folder/esm/esm2_t30_150M_UR50D.pt"
    export dms_output_folder=${DMS_output_score_folder_subs}/ESM2/150M
    
    python baselines/esm/compute_fitness.py \
        --model-location ${model_checkpoint} \
        --dms_index $DMS_index \
        --dms_mapping ${DMS_reference_file_path_subs} \
        --dms-input ${DMS_data_folder_subs} \
        --dms-output ${dms_output_folder} \
        --scoring-strategy ${scoring_strategy} \
        --model_type ${model_type}

    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with 150M: $elapsed_time seconds"

    start_time=$(date +%s.%N)
    export model_checkpoint="$checkpoint_folder/esm/esm2_t33_650M_UR50D.pt"
    export dms_output_folder=${DMS_output_score_folder_subs}/ESM2/650M
    
    python baselines/esm/compute_fitness.py \
        --model-location ${model_checkpoint} \
        --dms_index $DMS_index \
        --dms_mapping ${DMS_reference_file_path_subs} \
        --dms-input ${DMS_data_folder_subs} \
        --dms-output ${dms_output_folder} \
        --scoring-strategy ${scoring_strategy} \
        --model_type ${model_type}
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with 650M: $elapsed_time seconds"

    start_time=$(date +%s.%N)
    export model_checkpoint="$checkpoint_folder/esm/esm2_t36_3B_UR50D.pt"
    export dms_output_folder=${DMS_output_score_folder_subs}/ESM2/3B
    
    python baselines/esm/compute_fitness.py \
        --model-location ${model_checkpoint} \
        --dms_index $DMS_index \
        --dms_mapping ${DMS_reference_file_path_subs} \
        --dms-input ${DMS_data_folder_subs} \
        --dms-output ${dms_output_folder} \
        --scoring-strategy ${scoring_strategy} \
        --model_type ${model_type}
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i with 3B: $elapsed_time seconds"

done

