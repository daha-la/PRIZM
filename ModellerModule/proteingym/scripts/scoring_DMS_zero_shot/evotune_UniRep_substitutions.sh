#!/bin/bash 

source ../zero_shot_config.sh

export OMP_NUM_THREADS=1

export savedir="$PRIZM_PATH/finetuned_models/eUniRep"
export initial_weights_dir="$checkpoint_folder/UniRep/1900_weights_random"
export DMS_reference_file_path=$DMS_reference_file_path_subs
export steps=13000 #Same as Unirep paper

cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evotuning Unirep index $i"
    export DMS_index=$i

    start_time=$(date +%s.%N)

    python baselines/unirep/unirep_evotune.py \
        --seqs_fasta_path $DMS_MSA_data_folder \
        --save_weights_dir $savedir \
        --initial_weights_dir $initial_weights_dir \
        --num_steps $steps \
        --batch_size 128 \
        --mapping_path $DMS_reference_file_path \
        --DMS_index $DMS_index \
        --max_seq_len 500

    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken for $i: $elapsed_time seconds"
done

