
import torch
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForMaskedLM
from argparse import ArgumentParser

from baselines.prosst.prosst.structure.quantizer import PdbQuantizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_seq(fasta):
    for record in SeqIO.parse(fasta, "fasta"):
        return str(record.seq)


def tokenize_structure_sequence(structure_sequence):
    shift_structure_sequence = [i + 3 for i in structure_sequence]
    shift_structure_sequence = [1, *shift_structure_sequence, 2]
    return torch.tensor(
        [
            shift_structure_sequence,
        ],
        dtype=torch.long,
    )


@torch.no_grad()
def score_protein(model, tokenizer, sequence, structure_fasta, mutant_df):
    #sequence = read_seq(residue_fasta)
    structure_sequence = read_seq(structure_fasta)
    print('seq len;',len(sequence),'struc seq len;', len(structure_sequence))
    structure_sequence = [int(i) for i in structure_sequence.split(",")]
    ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)
    tokenized_results = tokenizer([sequence], return_tensors="pt")
    input_ids = tokenized_results["input_ids"].to(device)
    attention_mask = tokenized_results["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=ss_input_ids,
        labels=input_ids,
    )

    logits = outputs.logits
    logits = torch.log_softmax(logits[:, 1:-1, :], dim=-1)

    mutants = mutant_df["mutant"].tolist()
    scores = []
    vocab = tokenizer.get_vocab()
    for mutant in tqdm(mutants):
        pred_score = 0
        for sub_mutant in mutant.split(":"):
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
            score = logits[0, idx, vocab[mt]] - logits[0, idx, vocab[wt]]
            pred_score += score.item()
        scores.append(pred_score)

    return scores
    


def read_names(fasta_dir):
    files = Path(fasta_dir).glob("*.fasta")
    names = [file.stem for file in files]
    return names


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="AI4Protein/ProSST-2048", nargs="+", required=True)
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory containing all data",)
    parser.add_argument("--residue_dir", type=str, default=None, help="Directory containing FASTA files of residue sequences",)
    parser.add_argument("--struc_seq_dir", type=str, default=None, help="Directory containing FASTA files of structure sequences",)
    parser.add_argument("--mutant_dir", type=str, default=None, help="Directory containing CSV files with mutants",)
    parser.add_argument("--pdb_dir", type=str, default=None, help="Directory containing PDB files",)
    parser.add_argument("--output_scores_folder", default=None, help="Directory to save scores")

    # reference file
    parser.add_argument("--DMS_reference_file_path", type=str, help="Datasets reference '.csv' file.")
    parser.add_argument("--DMS_index", type=int, default=None,help="DMS Index of protein of interest")
    args = parser.parse_args()

    print("Scoring proteins...")
    os.makedirs(args.output_scores_folder, exist_ok=True)

    '''
    if args.base_dir:
        protein_names = read_names(f"{args.base_dir}/residue_sequence")
    if args.residue_dir:
        protein_names = read_names(args.residue_dir)
    protein_names = sorted(protein_names)
    print(protein_names)
    print(f">>> total proteins: {len(protein_names)}")'
    '''
    
    dataset_reference_path = args.DMS_reference_file_path
    dataset_reference = pd.read_csv(dataset_reference_path)
    dataset_entry = dataset_reference.loc[args.DMS_index]
    protein_name = dataset_entry["DMS_id"]
    sequence = dataset_entry["target_seq"]

    for model_name in args.model_name:
        print(f">>> Loading model {model_name}...")
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        #for idx, protein_name in enumerate(protein_names):
        #print(f">>> Scoring {protein_name}, current {idx+1}/{len(protein_names)}...")
        print(f">>> Scoring {protein_name}...")
        # load data
        #residue_fasta = f"{args.aa_seq_dir}/{protein_name}.fasta"
        if os.path.exists(f"{args.struc_seq_dir}/{model_name.split('-')[-1]}/{dataset_entry['pdb_file'].split('.')[0]}.fasta"):
            structure_fasta = f"{args.struc_seq_dir}/{model_name.split('-')[-1]}/{dataset_entry['pdb_file'].split('.')[0]}.fasta"
        else:
            print(f">>> Structure sequence not found for {protein_name}, extracting from PDB file...")
            pdb_file = f"{args.pdb_dir}/{dataset_entry['pdb_file']}"
            processor = PdbQuantizer(structure_vocab_size=int(model_name.split('-')[-1])) # model_name.split('-')[-1] is the model size
            structure_sequence = processor(pdb_file)
            structure_sequence = [i + 3 for i in structure_sequence]
            structure_sequence = ', '.join(map(str, structure_sequence))
            print(f">>> Saving structure sequence for {protein_name}...")
            with open(f"{args.struc_seq_dir}/{model_name.split('-')[-1]}/{dataset_entry['pdb_file'].split('.')[0]}.fasta", "w") as f:
                f.write(f'>{protein_name}\n')
                f.write(structure_sequence)
            structure_fasta = f"{args.struc_seq_dir}/{model_name.split('-')[-1]}/{dataset_entry['pdb_file'].split('.')[0]}.fasta"

        mutant_file = f"{args.mutant_dir}/{dataset_entry['DMS_filename']}"
        '''
        if args.base_dir:
            residue_fasta = f"{args.base_dir}/residue_sequence/{protein_name}.fasta"
            structure_fasta = f"{args.base_dir}/structure_sequence/{model_name.split('-')[-1]}/{protein_name}.fasta"
            mutant_file = f"{args.base_dir}/substitutions/{protein_name}.csv"
        if args.residue_dir:
            residue_fasta = f"{args.residue_dir}/{protein_name}.fasta"
        if args.structure_dir:
            structure_fasta = f"{args.structure_dir}/{protein_name}.fasta"
        if args.mutant_dir:
            mutant_file = f"{args.mutant_dir}/{protein_name}.csv"
        if os.path.exists(f"{args.output_scores_folder}/{protein_name}.csv"):
            mutant_file = f"{args.output_scores_folder}/{protein_name}.csv"'
        '''
        mutant_df = pd.read_csv(mutant_file)
        
        scores = score_protein(
                model=model,
                tokenizer=tokenizer,
                #residue_fasta=residue_fasta,
                sequence=sequence,
                structure_fasta=structure_fasta,
                mutant_df=mutant_df,
            )
        
        # remove the path from model_name
        model_name = model_name.split("/")[-1]
        mutant_df[model_name] = scores
        #corr = spearmanr(mutant_df["DMS_score"], mutant_df[model_name]).correlation
        #print(f">>> {model_name} on {protein_name}: {corr}")
            
        mutant_df.to_csv(f"{args.output_scores_folder}/{protein_name}.csv", index=False)


if __name__ == "__main__":
    main()