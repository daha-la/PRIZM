#!/bin/bash 

source ../zero_shot_config.sh

export GEMME_LOCATION="$installation_folder/GEMME"
export JET2_LOCATION="$installation_folder/JET2"
export TEMP_FOLDER="./gemme_tmp/"
export DMS_output_score_folder="${DMS_output_score_folder_subs}/GEMME/"

cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i

    start_time=$(date +%s.%N)

    python baselines/gemme/compute_fitness.py --DMS_index=$DMS_index --DMS_reference_file_path=$DMS_reference_file_path_subs \
    --DMS_data_folder=$DMS_data_folder_subs --MSA_folder=$DMS_MSA_data_folder --output_scores_folder=$DMS_output_score_folder \
    --GEMME_path=$GEMME_LOCATION --JET_path=$JET2_LOCATION --temp_folder=$TEMP_FOLDER

    end_time=$(date +%s.%N)
    elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
    echo "Time taken for $i: $elapsed_time seconds"

done