#!/bin/bash

source ../zero_shot_config.sh
#source activate protssn

# please download and unzip the following files to a folder: https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/ProteinGym_substitutions_pdb-csv_checked.zip
# eg. data/mutant_example/ProteinGym_substitutions_pdb-csv_checked
export DMS_and_structure_folder="$PRIZM_PATH/data/special_folders/ProtSSN"

# model checkpoint is at: https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/ProtSSN.model.tar
# please download and unzip the files to a folder
export model_checkpoint="$checkpoint_folder/ProtSSN/model"

# To ensemble all models use: "k10_h512 k20_h512 k30_h512 k10_h768 k20_h768 k30_h768 k10_h1280 k20_h1280 k30_h1280"
# To use a single model, only reference a single model string via the model_name argument (eg., k20_h512)
export model_name="k10_h512 k20_h512 k30_h512 k10_h768 k20_h768 k30_h768 k10_h1280 k20_h1280 k30_h1280"
export gnn_config="$PRIZM_PATH/ModellerModule/proteingym/baselines/protssn/src/config/egnn.yaml"
#export score_info=../../protssn_scores.csv

export DMS_output_score_folder="$DMS_output_score_folder_subs/ProtSSN"

cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i

    start_time=$(date +%s.%N)

    python baselines/protssn/compute_fitness.py \
        --gnn_config ${gnn_config} \
        --gnn_model_dir ${model_checkpoint} \
        --gnn_model_name ${model_name} \
        --use_ensemble \
        --mutant_dataset_dir ${DMS_and_structure_folder} \
        --DMS_reference_file_path $DMS_reference_file_path_subs \
        --DMS_index $DMS_index \
        --output_scores_folder ${DMS_output_score_folder} \
        --repo_path $PRIZM_PATH/ModellerModule
        #--score_info ${score_info}

    end_time=$(date +%s.%N)
    elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
    echo "Time taken for $i: $elapsed_time seconds"

done