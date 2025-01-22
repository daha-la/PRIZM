#!/bin/bash 

source ../zero_shot_config.sh
#source activate protein_fitness_prediction_hsu

export OMP_NUM_THREADS=1

export savedir="/home/dahala/mnt/ZeroShot/results/DMS_EvoUniRep_models"
export initial_weights_dir="/data/checkpoints/UniRep/1900_weights_random"
export DMS_reference_file_path=$DMS_reference_file_path_subs
# uncomment below to run for indels 
# export DMS_reference_file_path=$DMS_reference_file_path_indels
export steps=13000 #Same as Unirep paper

cd ../../proteingym/

for ((i=$1; i<=$2; i++))
do
    echo "Evotuning Unirep index $i"
    export DMS_index=$i
#"Experiment index to run (e.g. 0,1,2,...216)" >216 = Inhouse

    #start_time=$(date +%s.%N)

    python baselines/unirep/unirep_evotune.py \
        --seqs_fasta_path $DMS_MSA_data_folder \
        --save_weights_dir $savedir \
        --initial_weights_dir $initial_weights_dir \
        --num_steps $steps \
        --batch_size 128 \
        --mapping_path $DMS_reference_file_path \
        --DMS_index $DMS_index \
        --max_seq_len 500

    #end_time=$(date +%s.%N)
    #elapsed_time=$(echo "$end_time - $start_time" | bc)
    #echo "Time taken for $i: $elapsed_time seconds"
done

