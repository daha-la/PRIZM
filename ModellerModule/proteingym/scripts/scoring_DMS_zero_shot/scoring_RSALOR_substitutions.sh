#!/bin/bash 

source ../zero_shot_config.sh
#pip install rsalor

export output_scores_folder=${DMS_output_score_folder_subs}/RSALOR/

cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i # Currently not used in RSALOR, implement this before using RSALOR

    start_time=$(date +%s.%N)

    python baselines/RSALOR/run_rsalor.py \
        --DMS_reference_file_path ${DMS_reference_file_path_subs} \
        --DMS_data_folder ${DMS_data_folder_subs} \
        --MSA_folder ${DMS_MSA_data_folder} \
        --DMS_index $DMS_index \
        --DMS_structure_folder ${DMS_structure_folder} \
        --output_scores_folder ${output_scores_folder}

    end_time=$(date +%s.%N)
    elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
    echo "Time taken for $i with RSALOR: $elapsed_time seconds"

done