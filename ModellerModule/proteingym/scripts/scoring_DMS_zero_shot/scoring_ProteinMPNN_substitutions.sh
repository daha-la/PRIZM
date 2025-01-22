#!/bin/bash 

source ../zero_shot_config.sh
#source activate proteingym_env

export output_scores_folder=${DMS_output_score_folder_subs}/ProteinMPNN

# Vanilla Models
#export model_checkpoint="/home/dahala/mnt/ZeroShot/checkpoints/ProteinMPNN/vanilla_model_weights/v_48_002.pt"
#export model_checkpoint="/home/dahala/mnt/ZeroShot/checkpoints/ProteinMPNN/vanilla_model_weights/v_48_010.pt"
export model_checkpoint="/data/checkpoints/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
#export model_checkpoint="/home/dahala/mnt/ZeroShot/checkpoints/ProteinMPNN/vanilla_model_weights/v_48_030.pt"

#Soluble Models
#export model_checkpoint="/home/dahala/mnt/ZeroShot/checkpoints/ProteinMPNN/soluble_model_weights/v_48_002.pt"
#export model_checkpoint="/home/dahala/mnt/ZeroShot/checkpoints/ProteinMPNN/soluble_model_weights/v_48_010.pt"
#export model_checkpoint="/home/dahala/mnt/ZeroShot/checkpoints/ProteinMPNN/soluble_model_weights/v_48_020.pt"
#export model_checkpoint="/home/dahala/mnt/ZeroShot/checkpoints/ProteinMPNN/soluble_model_weights/v_48_030.pt"



cd ../../proteingym/

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i
#"Experiment index to run (e.g. 0,1,2,...216)" >216 = Inhouse

    start_time=$(date +%s.%N)


    python baselines/protein_mpnn/compute_fitness.py \
        --checkpoint ${model_checkpoint} \
        --structure_folder ${DMS_structure_folder} \
        --DMS_index $DMS_index \
        --DMS_reference_file_path ${DMS_reference_file_path_subs} \
        --DMS_data_folder ${DMS_data_folder_subs} \
        --output_scores_folder ${output_scores_folder}
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i: $elapsed_time seconds"
done

