#!/bin/bash

source ../zero_shot_config.sh

export foldseek_path="$installation_folder/foldseek/bin/foldseek"

cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i

    start_time=$(date +%s.%N)
    export MULAN_model_path="$checkpoint_folder/MULAN/MULAN-small"
    python baselines/mulan/compute_fitness.py \
                --MULAN_model_name_or_path ${MULAN_model_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --structure_data_folder ${DMS_structure_folder} \
                --DMS_index $DMS_index \
                --foldseek_path $foldseek_path \
                --output_scores_folder "${DMS_output_score_folder_subs}/MULAN/MULAN_small" \
                --use_foldseek 

    end_time=$(date +%s.%N)
    elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
    echo "Time taken for $i with MULAN-small: $elapsed_time seconds"

    # Removing temporary files
    rm -rf "${DMS_data_folder_subs}/test_angles.npy.gz"
    rm -rf "${DMS_data_folder_subs}/test_foldseek_masked_sequences.json"
    rm -rf "${DMS_data_folder_subs}/test_names.json"
    rm -rf "${DMS_data_folder_subs}/test_sequences.json"
done