#!/bin/bash

source ../zero_shot_config.sh

export SaProt_model_path="$checkpoint_folder/SaProt/SaProt_650M_AF2" #Path where you have downloaded all SaProt model/tokenizer files from the HF hub (https://huggingface.co/westlake-repl/SaProt_650M_AF2)
export output_scores_folder="${DMS_output_score_folder_subs}/SaProt/SaProt_650M_AF2"
export foldseek_bin="$installation_folder/foldseek/bin/foldseek" #(Download from here: https://github.com/steineggerlab/foldseek?tab=readme-ov-file)

cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i

    start_time=$(date +%s.%N)

    python baselines/saprot/compute_fitness.py \
                --foldseek_bin ${foldseek_bin} \
                --SaProt_model_name_or_path ${SaProt_model_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --structure_data_folder ${DMS_structure_folder} \
                --DMS_index $DMS_index \
                --output_scores_folder ${output_scores_folder}

    end_time=$(date +%s.%N)
    elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
    echo "Time taken for $i with SaProt: $elapsed_time seconds"

done