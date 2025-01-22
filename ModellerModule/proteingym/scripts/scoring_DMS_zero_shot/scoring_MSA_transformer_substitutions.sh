#!/bin/bash 

source ../zero_shot_config.sh
#source activate proteingym_env

# MSA transformer checkpoint 
export model_checkpoint="/data/checkpoints/MSA_Transformer/esm_msa1b_t12_100M_UR50S.pt"
export DMS_index=1
#"Experiment index to run (e.g. 1,2,...217)"
export dms_output_folder="${DMS_output_score_folder_subs}/MSA_Transformer/"
export scoring_strategy=masked-marginals # MSA transformer only supports "masked-marginals" #"wt-marginals"
export scoring_window=overlapping
export model_type=MSA_transformer
export random_seeds="1 2 3 4 5"


cd ../../proteingym

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
        --scoring-window ${scoring_window} \
        --msa-path ${DMS_MSA_data_folder} \
        --msa-weights-folder ${DMS_MSA_weights_folder} \
        --seeds ${random_seeds} #\
        #--nogpu	

    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i: $elapsed_time seconds"
done
