#!/bin/bash 

source ../zero_shot_config.sh
#source activate protein_fitness_prediction_hsu

export OMP_NUM_THREADS=1

export model_path="/home/dahala/mnt/ZeroShot/results/DMS_EvoUniRep_models"
export output_dir=${DMS_output_score_folder_subs}/UniRep_evotuned

cd ../../proteingym/
for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i
#"Experiment index to run (e.g. 0,1,2,...216)" >216 = Inhouse

    start_time=$(date +%s.%N)

    python baselines/unirep/unirep_inference.py \
                --model_path $model_path \
                --data_path $DMS_data_folder_subs \
                --output_dir $output_dir \
                --mapping_path $DMS_reference_file_path_subs \
                --DMS_index $DMS_index \
                --batch_size 32 \
                --evotune
    
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i: $elapsed_time seconds"
done
