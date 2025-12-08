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
PROGEN2_DIR="$CHECKPOINT_ROOT/Progen2"
PROTGPT2_DIR="$CHECKPOINT_ROOT/ProtGPT2"
RITA_DIR="$CHECKPOINT_ROOT/RITA"
TRANCEPTION_DIR="$CHECKPOINT_ROOT/Tranception"
UNIREP_DIR="$CHECKPOINT_ROOT/UniRep"

mkdir -p "$ESM_DIR" "$PROGEN2_DIR" \
         "$PROTGPT2_DIR" "$RITA_DIR" "$TRANCEPTION_DIR" "$UNIREP_DIR"

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

# -------- ProGen2 --------
echo ""
echo "== Downloading ProGen2 checkpoints (Small / Medium / Base / Large) =="

declare -A PROGEN2_URLS
PROGEN2_URLS[progen2-small]="https://storage.googleapis.com/anon-progen-research/checkpoints/progen2-small.tar.gz"
PROGEN2_URLS[progen2-medium]="https://storage.googleapis.com/anon-progen-research/checkpoints/progen2-medium.tar.gz"
PROGEN2_URLS[progen2-base]="https://storage.googleapis.com/anon-progen-research/checkpoints/progen2-base.tar.gz"
PROGEN2_URLS[progen2-large]="https://storage.googleapis.com/anon-progen-research/checkpoints/progen2-large.tar.gz"

for model in progen2-small progen2-medium progen2-base progen2-large; do
    url="${PROGEN2_URLS[$model]}"
    tmp_tar="$PROGEN2_DIR/${model}.tar.gz"

    download "$url" "$tmp_tar"

    echo "  - Extracting $model"
    # Extract into Progen2 directory; tar contains its own subfolder.
    tar -xzf "$tmp_tar" -C "$PROGEN2_DIR"
    # You can delete the tarballs if you like:
    # rm -f "$tmp_tar"
done

# -------- ProtGPT2 --------
echo ""
echo "== Downloading ProtGPT2 from Hugging Face (git-lfs clone) =="

if ! have_cmd git; then
    echo "ERROR: git is required for ProtGPT2 download." >&2
else
    if ! have_cmd git-lfs; then
        echo "WARNING: git-lfs not found; large files may not download correctly."
        echo "         Install git-lfs for a full clone."
    else
        git lfs install >/dev/null 2>&1 || true
    fi

    if [ -d "$PROTGPT2_DIR/.git" ]; then
        echo "  - ProtGPT2 repo already present, skipping clone."
    else
        rm -rf "$PROTGPT2_DIR"
        echo "  - Cloning nferruz/ProtGPT2 into $PROTGPT2_DIR"
        git clone https://huggingface.co/nferruz/ProtGPT2 "$PROTGPT2_DIR"
    fi
fi

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

# -------- Tranception --------
echo ""
echo "== Downloading Tranception checkpoints (Small / Medium / Large) =="

TRANCEPTION_BASE="https://marks.hms.harvard.edu/tranception"

pushd "$TRANCEPTION_DIR" >/dev/null

for size in Small Medium Large; do
    zip_name="Tranception_${size}_checkpoint.zip"
    url="${TRANCEPTION_BASE}/${zip_name}"

    download "$url" "$zip_name"

    echo "  - Extracting Tranception $size"
    unzip -o "$zip_name"
    # rm -f "$zip_name"   # uncomment to remove zip files after extraction
done

popd >/dev/null

# -------- UniRep --------
echo ""
echo "== Downloading UniRep weights (1900_weights / 1900_weights_random) =="

if ! have_cmd aws; then
    echo "ERROR: aws CLI is required for UniRep weights (anonymous S3 sync)." >&2
    echo "       Install awscli or download UniRep weights manually."
else
    pushd "$UNIREP_DIR" >/dev/null
    echo "  - Syncing 1900_weights"
    aws s3 sync --no-sign-request s3://unirep-public/1900_weights 1900_weights
    echo "  - Syncing 1900_weights_random"
    aws s3 sync --no-sign-request s3://unirep-public/1900_weights_random 1900_weights_random
    popd >/dev/null
fi


# -------- .gitignore --------

GITIGNORE_FILE="$CHECKPOINT_ROOT/.gitignore"

CHECKPOINT_DIRS=(
  "Progen2/"
  "ProtGPT2/"
  "RITA/"
  "Tranception/"
  "UniRep/"
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

# Add ESM checkpoint file patterns (but not contact-regression)
for f in "${ESM_CHECKPOINTS[@]}"; do
    add_pattern_if_missing "$f"
done

echo "Updated .gitignore at: $GITIGNORE_FILE"

# -------- done --------
echo ""
echo "All requested checkpoints processed."
echo "If any step failed (missing tools, etc.), you can rerun the script after fixing it."
