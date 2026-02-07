import os
import torch
from sequence_models.collaters import SimpleCollater, StructureCollater, BGCCollater
from sequence_models.pretrained import load_carp, load_gnn, MIF
from sequence_models.constants import PROTEIN_ALPHABET

# Define local paths
CARP_PATH = '/data/checkpoints/CARP/'
MIF_PATH = '/data/checkpoints/MIF/'
BIG_PATH = '/data/checkpoints/BIG/'

def load_model_and_alphabet(model_name, model_dir=None):
    if not model_name.endswith(".pt"): 
        if 'big' in model_name:
            model_file = os.path.join(BIG_PATH, f"{model_name}.pt")
        elif 'carp' in model_name:
            model_file = os.path.join(CARP_PATH, f"{model_name}.pt")
        elif 'mif' in model_name:
            model_file = os.path.join(MIF_PATH, f"{model_name}.pt")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file {model_file} not found.")
        
        model_data = torch.load(model_file, map_location="cpu")
    else:
        model_data = torch.load(model_name, map_location="cpu")
        
    if 'big' in model_data['model']:
        pfam_to_domain = model_data['pfam_to_domain']
        tokens = model_data['tokens']
        collater = BGCCollater(tokens, pfam_to_domain)
    else:
        collater = SimpleCollater(PROTEIN_ALPHABET, pad=True)
    
    if 'carp' in model_data['model']:
        model = load_carp(model_data)
    elif model_data['model'] in ['mif', 'mif-st']:
        gnn = load_gnn(model_data)
        cnn = None
        if model_data['model'] == 'mif-st':
            # Load CNN directly from the local file instead of downloading
            cnn_file = os.path.join(CARP_PATH, 'carp_640M.pt')
            if not os.path.exists(cnn_file):
                raise FileNotFoundError(f"CNN file {cnn_file} not found.")
            cnn_data = torch.load(cnn_file, map_location="cpu")
            cnn = load_carp(cnn_data)
        
        collater = StructureCollater(collater, n_connections=30)
        model = MIF(gnn, cnn=cnn)

    return model, collater