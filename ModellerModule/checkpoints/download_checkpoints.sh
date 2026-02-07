#!/usr/bin/env bash
set -euo pipefail

###############################################
# PRIZM checkpoint downloader
#
# Usage:
#   bash download_checkpoints.sh            # save into folder containing this script
#   bash download_checkpoints.sh /my/path   # save into /my/path
#
# Requirements (depending on what you download):
#   - curl or wget
#   - tar, unzip
#   - git + git-lfs (ProtGPT2 + RITA)
#   - awscli with anonymous access (UniRep)
###############################################

# -------- configuration --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_ROOT="${1:-$SCRIPT_DIR}"   # change this default if you want a fixed path

echo "Saving checkpoints under: $CHECKPOINT_ROOT"
mkdir -p "$CHECKPOINT_ROOT"

ESM_DIR="$CHECKPOINT_ROOT/esm"
MIF_DIR="$CHECKPOINT_ROOT/MIF"
PROGEN2_DIR="$CHECKPOINT_ROOT/Progen2"
PROTGPT2_DIR="$CHECKPOINT_ROOT/ProtGPT2"
RITA_DIR="$CHECKPOINT_ROOT/RITA"
TRANCEPTION_DIR="$CHECKPOINT_ROOT/Tranception"
UNIREP_DIR="$CHECKPOINT_ROOT/UniRep"
CARP_DIR="$CHECKPOINT_ROOT/CARP"
MSA_TRANSFORMER_DIR="$CHECKPOINT_ROOT/MSA_Transformer"
MULAN_DIR="$CHECKPOINT_ROOT/MULAN"
PROTSSN_DIR="$CHECKPOINT_ROOT/ProtSSN"
SAPROT_DIR="$CHECKPOINT_ROOT/SaProt"

mkdir -p "$ESM_DIR" "$PROGEN2_DIR" \
         "$PROTGPT2_DIR" "$RITA_DIR" "$TRANCEPTION_DIR" "$UNIREP_DIR" \
         "$CARP_DIR" "$MSA_TRANSFORMER_DIR" "$MULAN_DIR" "$PROTSSN_DIR" "$SAPROT_DIR"

# -------- helper functions --------
have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

download() {
    local url="$1"
    local dest="$2"

    if [ -f "$dest" ]; then
        echo "  - Exists, skipping: $dest"
        return
    fi
    echo "  - Downloading: $url"
    if have_cmd curl; then
        curl -L "$url" -o "$dest"
    elif have_cmd wget; then
        wget -O "$dest" "$url"
    else
        echo "ERROR: need curl or wget installed." >&2
        exit 1
    fi
}

# -------- CARP --------
echo ""
echo "== Downloading CARP checkpoints =="

CARP_BASE="https://zenodo.org/records/6564798/files"

CARP_MODELS=(
  "carp_600K.pt"
  "carp_38M.pt"
  "carp_76M.pt"
  "carp_640M.pt"
)

for f in "${CARP_MODELS[@]}"; do
    download "${CARP_BASE}/${f}" "$CARP_DIR/${f}"
done

# -------- ESM --------
echo ""
echo "== Downloading ESM checkpoints =="

ESM_BASE="https://dl.fbaipublicfiles.com/fair-esm/models"

ESM_MODELS=(
  "esm1b_t33_650M_UR50S"
  "esm1v_t33_650M_UR90S_1"
  "esm1v_t33_650M_UR90S_2"
  "esm1v_t33_650M_UR90S_3"
  "esm1v_t33_650M_UR90S_4"
  "esm1v_t33_650M_UR90S_5"
  "esm_if1_gvp4_t16_142M_UR50"
  "esm2_t6_8M_UR50D"
  "esm2_t12_35M_UR50D"
  "esm2_t30_150M_UR50D"
  "esm2_t33_650M_UR50D"
  "esm2_t36_3B_UR50D"
)

for model in "${ESM_MODELS[@]}"; do
    dest="$ESM_DIR/${model}.pt"
    download "${ESM_BASE}/${model}.pt" "$dest"
done

# -------- MIF / MIFST --------
echo ""
echo "== Downloading MIF / MIFST checkpoints =="

MIF_BASE="https://zenodo.org/records/6573779/files"

download "${MIF_BASE}/mif.pt"   "$MIF_DIR/mif.pt"
download "${MIF_BASE}/mifst.pt" "$MIF_DIR/mifst.pt"

# -------- MSA Transformer --------
echo ""
echo "== Downloading MSA Transformer checkpoint =="

MSA_MODEL="esm_msa1b_t12_100M_UR50S"
download \
  "https://dl.fbaipublicfiles.com/fair-esm/models/${MSA_MODEL}.pt" \
  "$MSA_TRANSFORMER_DIR/${MSA_MODEL}.pt"

# -------- MULAN --------
echo ""
echo "== Downloading MULAN-small checkpoint =="

MULAN_REPO="DFrolova/MULAN-small"
MULAN_MODEL_DIR="$MULAN_DIR/MULAN-small"
mkdir -p "$MULAN_MODEL_DIR"

MULAN_FILES=(
  "config.json"
  "model.safetensors"
)

for f in "${MULAN_FILES[@]}"; do
    url="https://huggingface.co/${MULAN_REPO}/resolve/main/${f}"
    dest="$MULAN_MODEL_DIR/${f}"
    download "$url" "$dest"
done


# -------- ProGen2 --------
echo ""
echo "== Downloading ProGen2 checkpoints (Small / Medium / Base / Large / XLarge) =="

# Official URLs from ProGen2 README
declare -A PROGEN2_URLS
PROGEN2_URLS[progen2-small]="https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-small.tar.gz"
PROGEN2_URLS[progen2-medium]="https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-medium.tar.gz"
PROGEN2_URLS[progen2-base]="https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-base.tar.gz"
PROGEN2_URLS[progen2-large]="https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-large.tar.gz"
PROGEN2_URLS[progen2-xlarge]="https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-xlarge.tar.gz"

for model in progen2-small progen2-medium progen2-base progen2-large progen2-xlarge; do
    url="${PROGEN2_URLS[$model]}"
    model_dir="$PROGEN2_DIR/${model}"

    # If model directory already contains a checkpoint, skip
    if [ -d "$model_dir" ] && [ -f "$model_dir/pytorch_model.bin" ]; then
        echo "  - $model already present in $model_dir, skipping download."
        continue
    fi

    mkdir -p "$model_dir"
    tmp_tar="$model_dir/${model}.tar.gz"

    download "$url" "$tmp_tar"

    echo "  - Extracting $model into $model_dir"
    tar -xzf "$tmp_tar" -C "$model_dir"
    # Optionally remove the tarball:
    rm -f "$tmp_tar"
done

# -------- ProtGPT2 --------
echo ""
echo "== Downloading ProtGPT2 files from Hugging Face =="

mkdir -p "$PROTGPT2_DIR"

PROTGPT2_FILES=(
  "config.json"
  "merges.txt"
  "pytorch_model.bin"
  "special_tokens_map.json"
  "tokenizer.json"
  "vocab.json"
)

for f in "${PROTGPT2_FILES[@]}"; do
    url="https://huggingface.co/nferruz/ProtGPT2/resolve/main/${f}"
    dest="$PROTGPT2_DIR/${f}"
    download "$url" "$dest"
done

# -------- ProtSSN --------
echo ""
echo "== Downloading ProtSSN checkpoints (direct HF download) =="

PROTSSN_MODEL_DIR="$PROTSSN_DIR/model"
mkdir -p "$PROTSSN_MODEL_DIR"

PROTSSN_BASE="https://huggingface.co/tyang816/ProtSSN/resolve/main"

PROTSSN_FILES=(
  "protssn_k10_h512.pt"
  "protssn_k10_h768.pt"
  "protssn_k10_h1280.pt"
  "protssn_k20_h512.pt"
  "protssn_k20_h768.pt"
  "protssn_k20_h1280.pt"
  "protssn_k30_h512.pt"
  "protssn_k30_h768.pt"
  "protssn_k30_h1280.pt"
)

for f in "${PROTSSN_FILES[@]}"; do
    url="${PROTSSN_BASE}/${f}"
    dest="${PROTSSN_MODEL_DIR}/${f}"
    download "$url" "$dest"
done

# -------- RITA --------
echo ""
echo "== Downloading RITA models (s / m / l / xl) =="

if ! have_cmd git; then
    echo "ERROR: git is required for RITA download." >&2
else
    if have_cmd git-lfs; then
        git lfs install >/dev/null 2>&1 || true
    fi

    for size in s m l xl; do
        repo="lightonai/RITA_${size}"
        outdir="$RITA_DIR/RITA_${size}"
        if [ -d "$outdir/.git" ]; then
            echo "  - RITA_${size} already present, skipping clone."
        else
            rm -rf "$outdir"
            echo "  - Cloning $repo into $outdir"
            git clone "https://huggingface.co/${repo}" "$outdir"
        fi
    done
fi

# -------- SaProt --------
echo ""
echo "== Downloading SaProt_650M_AF2 checkpoint (Hugging Face direct download) =="

SAPROT_MODEL_DIR="$SAPROT_DIR/SaProt_650M_AF2"
mkdir -p "$SAPROT_MODEL_DIR"

SAPROT_FILES=(
  "SaProt_650M_AF2.pt"
  "config.json"
  "pytorch_model.bin"
  "special_tokens_map.json"
  "tokenizer_config.json"
  "vocab.txt"
)

for f in "${SAPROT_FILES[@]}"; do
    url="https://huggingface.co/westlake-repl/SaProt_650M_AF2/resolve/main/${f}"
    dest="$SAPROT_MODEL_DIR/${f}"
    download "$url" "$dest"
done

# -------- Tranception --------
echo ""
echo "== Downloading Tranception checkpoints (Small / Medium / Large) =="

TRANCEPTION_BASE="https://marks.hms.harvard.edu/tranception"

pushd "$TRANCEPTION_DIR" >/dev/null

for size in Small Medium Large; do
    zip_name="Tranception_${size}_checkpoint.zip"
    url="${TRANCEPTION_BASE}/${zip_name}"

    # If the final folder already exists with a config and model, skip
    if [ -d "Tranception_${size}" ] && [ -f "Tranception_${size}/pytorch_model.bin" ]; then
        echo "  - Tranception_${size} already present, skipping download."
        continue
    fi

    download "$url" "$zip_name"

    echo "  - Extracting Tranception $size"
    unzip -o "$zip_name"
    rm -f "$zip_name"
done

popd >/dev/null

# -------- UniRep --------
echo ""
echo "== Downloading UniRep weights =="

UNIREP_FILES=(
  "embed_matrix:0.npy"
  "fully_connected_biases:0.npy"
  "fully_connected_weights:0.npy"
  "mlstm_layer_norm:0.npy"
  "rnn_mlstm_mlstm_b:0.npy"
  "rnn_mlstm_mlstm_b_norm:0.npy"
  "rnn_mlstm_mlstm_wmh:0.npy"
  "rnn_mlstm_mlstm_wmh_norm:0.npy"
  "rnn_mlstm_mlstm_wmx:0.npy"
  "rnn_mlstm_mlstm_wmx_norm:0.npy"
  "rnn_mlstm_mlstm_wx:0.npy"
  "rnn_mlstm_mlstm_wh:0.npy"
  "rnn_mlstm_mlstm_gx:0.npy"
  "rnn_mlstm_mlstm_gh:0.npy"
  "rnn_mlstm_mlstm_gmx:0.npy"
  "rnn_mlstm_mlstm_gmh:0.npy"
)

download_unirep_folder() {
    local folder="$1"
    local target="$UNIREP_DIR/$folder"
    mkdir -p "$target"

    echo "  - Downloading UniRep $folder"
    for f in "${UNIREP_FILES[@]}"; do
        url="https://unirep-public.s3.amazonaws.com/${folder}/${f}"
        dest="${target}/${f}"

        download "$url" "$dest"
    done
}

download_unirep_folder "1900_weights"
download_unirep_folder "1900_weights_random"


# -------- .gitignore --------

GITIGNORE_FILE="$CHECKPOINT_ROOT/.gitignore"

CHECKPOINT_DIRS=(
  "Progen2/"
  "ProtGPT2/"
  "RITA/"
  "Tranception/"
  "UniRep/"
  "CARP/"
  "MULAN/"
  "MIF/"
  "ProtSSN/"
  "SaProt/"
)

# ESM: ignore only the large model checkpoints downloaded by this script,
# but NOT the contact-regression files you already ship with the repo.
ESM_CHECKPOINTS=(
  "esm/esm1b_t33_650M_UR50S.pt"
  "esm/esm1v_t33_650M_UR90S_1.pt"
  "esm/esm1v_t33_650M_UR90S_2.pt"
  "esm/esm1v_t33_650M_UR90S_3.pt"
  "esm/esm1v_t33_650M_UR90S_4.pt"
  "esm/esm1v_t33_650M_UR90S_5.pt"
  "esm/esm_if1_gvp4_t16_142M_UR50.pt"
  "esm/esm2_t6_8M_UR50D.pt"
  "esm/esm2_t12_35M_UR50D.pt"
  "esm/esm2_t30_150M_UR50D.pt"
  "esm/esm2_t33_650M_UR50D.pt"
  "esm/esm2_t36_3B_UR50D.pt"
)

MSA_TRANSFORMER_CHECKPOINTS=(
  "MSA_Transformer/esm_msa1b_t12_100M_UR50S.pt"
)

echo ""
echo "== Updating .gitignore =="

# Ensure file exists
touch "$GITIGNORE_FILE"

add_pattern_if_missing() {
    local pattern="$1"
    if ! grep -qxF "$pattern" "$GITIGNORE_FILE"; then
        echo "  - Adding $pattern to .gitignore"
        echo "$pattern" >> "$GITIGNORE_FILE"
    else
        echo "  - $pattern already in .gitignore"
    fi
}

# Add folder-level ignores
for dir in "${CHECKPOINT_DIRS[@]}"; do
    add_pattern_if_missing "$dir"
done

# Add ESM checkpoints (but not contact-regression)
for f in "${ESM_CHECKPOINTS[@]}"; do
    add_pattern_if_missing "$f"
done

# Add MSA Transformer checkpoints (but not contact-regression)
for f in "${MSA_TRANSFORMER_CHECKPOINTS[@]}"; do
    add_pattern_if_missing "$f"
done

echo "Updated .gitignore at: $GITIGNORE_FILE"

# -------- done --------
echo ""
echo "All requested checkpoints processed."
echo "If any step failed (missing tools, etc.), you can rerun the script after fixing it."

