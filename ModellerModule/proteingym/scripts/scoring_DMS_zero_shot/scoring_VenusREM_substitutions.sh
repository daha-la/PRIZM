source ../zero_shot_config.sh
#source activate venusrem

# the structure pdb files can be found in ProtSSN: https://github.com/tyang816/ProtSSN
# ProteinGym a2m homology sequences (EVCouplings): https://huggingface.co/datasets/tyang816/VenusREM/blob/main/aa_seq_aln_a2m.tar.gz. 
# The original a2m files are downloaded at [ProteinGym](https://github.com/OATML-Markslab/ProteinGym).
export DMS_folder="$PRIZM_PATH/data/special_folders/VenusREM"
export DMS_residue_folder="${DMS_folder}/aa_seq"
#export DMS_residue_alignment_folder="${DMS_folder}/aa_seq_aln_a2m"
export DMS_structure_seq_folder="${DMS_folder}/struc_seq"
#export DMS_data_folder_subs="${DMS_folder}/substitutions"
export DMS_output_score_folder="${DMS_output_score_folder_subs}/VenusREM"
# 128 512 1024 2048 4096
cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i
    for model_hyp in 20 128 512 1024 2048 4096; do
        
        export model_name="AI4Protein/ProSST-$model_hyp"

        echo "Using Model: $model_name"

        start_time=$(date +%s.%N)

        python baselines/venusrem/compute_fitness.py \
            --model_name ${model_name} \
            --base_dir ${DMS_folder} \
            --mutant_dir $DMS_data_folder_subs \
            --aa_seq_dir $DMS_residue_folder \
            --aa_seq_aln_dir $DMS_MSA_data_folder \
            --struc_seq_dir $DMS_structure_seq_folder \
            --pdb_dir $DMS_structure_folder \
            --DMS_reference_file_path $DMS_reference_file_path_subs \
            --DMS_index $DMS_index \
            --output_scores_folder ${DMS_output_score_folder}

        end_time=$(date +%s.%N)
        elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
        echo "Time taken for $model_hyp: $elapsed_time seconds"
    done
done