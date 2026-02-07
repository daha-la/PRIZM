#!/usr/bin/env bash
set -euo pipefail

# Installer for PRIZM tools:
# - GEMME
# - JET2
# - Foldseek
#
# It installs into the directory where the script lives.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$SCRIPT_DIR"
CACHE_DIR="$INSTALL_DIR/.cache"

FORCE=0
SKIP_GEMME=0
SKIP_JET2=0
SKIP_FOLDSEEK=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --force            Reinstall (delete existing tool dirs first)
  --skip-gemme       Skip GEMME
  --skip-jet2        Skip JET2
  --skip-foldseek    Skip Foldseek
  -h, --help         Show help

Installs to:
  $INSTALL_DIR

Expected outputs:
  GEMME/gemme.py
  JET2/jet
  foldseek/bin/foldseek
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --skip-gemme) SKIP_GEMME=1; shift ;;
    --skip-jet2) SKIP_JET2=1; shift ;;
    --skip-foldseek) SKIP_FOLDSEEK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: Missing required command: $1"
    exit 1
  }
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

download() {
  # download <url> <output_path>
  local url="$1"
  local out="$2"

  if command -v curl >/dev/null 2>&1; then
    # Try secure first
    if curl -L --fail --retry 3 --retry-delay 2 -o "$out" "$url"; then
      return 0
    fi

    echo "WARNING: TLS certificate verification failed for:"
    echo "  $url"
    echo "Retrying download with --insecure (no certificate verification)."
    echo "If this concerns you, download the file manually and place it in:"
    echo "  $out"
    echo

    curl -L --fail --retry 3 --retry-delay 2 --insecure -o "$out" "$url"
    return 0
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
    return 0
  fi

  echo "ERROR: Neither curl nor wget found."
  exit 1
}


rm_if_force() {
  local path="$1"
  if [[ "$FORCE" == "1" ]] && [[ -e "$path" ]]; then
    echo "Removing existing: $path"
    rm -rf "$path"
  fi
}

extract_tgz_into() {
  # extract_tgz_into <tgz_file> <dest_dir>
  local tgz="$1"
  local dest="$2"
  mkdir -p "$dest"

  # Extract to a temp dir first so we can handle different archive layouts.
  local tmp
  tmp="$(mktemp -d)"
  tar -xzf "$tgz" -C "$tmp"

  # If exactly one top-level directory exists, move its contents into dest; else move all.
  local top_count
  top_count="$(find "$tmp" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')"
  if [[ "$top_count" == "1" ]] && [[ -d "$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)" ]]; then
    local top_dir
    top_dir="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    shopt -s dotglob
    mv "$top_dir"/* "$dest"/
    shopt -u dotglob
  else
    shopt -s dotglob
    mv "$tmp"/* "$dest"/
    shopt -u dotglob
  fi

  rm -rf "$tmp"
}

echo "== PRIZM tools installer =="
echo "Install dir: $INSTALL_DIR"
echo

need_cmd tar
mkdir -p "$CACHE_DIR"

########################################
# GEMME
########################################
if [[ "$SKIP_GEMME" == "0" ]]; then
  GEMME_DIR="$INSTALL_DIR/GEMME"
  rm_if_force "$GEMME_DIR"
  if [[ -d "$GEMME_DIR" ]]; then
    echo "[GEMME] already present: $GEMME_DIR (use --force to reinstall)"
  else
    echo "[GEMME] downloading..."
    GEMME_URL="http://www.lcqb.upmc.fr/GEMME/package/GEMME.tgz"
    GEMME_TGZ="$CACHE_DIR/GEMME.tgz"
    download "$GEMME_URL" "$GEMME_TGZ"
    echo "[GEMME] extracting..."
    extract_tgz_into "$GEMME_TGZ" "$GEMME_DIR"
  fi

  if [[ ! -f "$GEMME_DIR/gemme.py" ]]; then
    echo "ERROR: GEMME install finished but gemme.py not found at:"
    echo "  $GEMME_DIR/gemme.py"
    echo "Check the archive contents/structure."
    exit 1
  fi
  echo "[GEMME] OK: $GEMME_DIR/gemme.py"
  echo
fi

########################################
# JET2
########################################
if [[ "$SKIP_JET2" == "0" ]]; then
  JET2_DIR="$INSTALL_DIR/JET2"
  rm_if_force "$JET2_DIR"
  if [[ -d "$JET2_DIR" ]]; then
    echo "[JET2] already present: $JET2_DIR (use --force to reinstall)"
  else
    echo "[JET2] downloading..."
    JET2_URL="http://www.lcqb.upmc.fr/JET2/package/JET2.tgz"
    JET2_TGZ="$CACHE_DIR/JET2.tgz"
    download "$JET2_URL" "$JET2_TGZ"
    echo "[JET2] extracting..."
    extract_tgz_into "$JET2_TGZ" "$JET2_DIR"
  fi

  if [[ ! -e "$JET2_DIR/jet" ]]; then
    echo "ERROR: JET2 install finished but 'jet' not found at:"
    echo "  $JET2_DIR/jet"
    echo "Check the archive contents/structure."
    exit 1
  fi
  # Make it executable if needed
  chmod +x "$JET2_DIR/jet" 2>/dev/null || true
  echo "[JET2] OK: $JET2_DIR/jet"
  echo
fi

########################################
# Patch GEMME and JET2 default.conf to point to local JET2 matrix folder
########################################

GEMME_CONF="$INSTALL_DIR/GEMME/default.conf"
JET2_CONF="$INSTALL_DIR/JET2/default.conf"

# Replace /opt/JET/ -> <INSTALL_DIR>/JET2/
sed -i "s|/opt/JET/|$INSTALL_DIR/JET2/|g" "$GEMME_CONF"
sed -i "s|/opt/JET/|$INSTALL_DIR/JET2/|g" "$JET2_CONF"

# Replace /opt/JET2/ -> <INSTALL_DIR>/JET2/
sed -i "s|/opt/JET2/|$INSTALL_DIR/JET2/|g" "$GEMME_CONF"
sed -i "s|/opt/JET2/|$INSTALL_DIR/JET2/|g" "$JET2_CONF"

echo "[patch] Updated JET2 paths in:"
echo "  $GEMME_CONF"
echo "  $JET2_CONF"

########################################
# Foldseek
########################################
if [[ "$SKIP_FOLDSEEK" == "0" ]]; then
  FOLDSEEK_DIR="$INSTALL_DIR/foldseek"
  rm_if_force "$FOLDSEEK_DIR"
  if [[ -d "$FOLDSEEK_DIR" ]]; then
    echo "[Foldseek] already present: $FOLDSEEK_DIR (use --force to reinstall)"
  else
    echo "[Foldseek] downloading precompiled binary..."

    OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
    ARCH="$(uname -m)"
    URL=""

    if [[ "$OS" == "linux" ]]; then
      if [[ "$ARCH" == "x86_64" || "$ARCH" == "amd64" ]]; then
        # Prefer AVX2 build when available (many servers/workstations have it)
        if grep -q -m1 -w avx2 /proc/cpuinfo 2>/dev/null; then
          URL="https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz"
        else
          echo "ERROR: No AVX2 detected on this CPU."
          echo "Foldseek's common precompiled Linux build is AVX2; please install via conda/bioconda or build from source."
          exit 1
        fi
      elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
        URL="https://mmseqs.com/foldseek/foldseek-linux-arm64.tar.gz"
      else
        echo "ERROR: Unsupported Linux arch for this installer: $ARCH"
        exit 1
      fi
    elif [[ "$OS" == "darwin" ]]; then
      URL="https://mmseqs.com/foldseek/foldseek-osx-universal.tar.gz"
    else
      echo "ERROR: Unsupported OS for Foldseek precompiled install: $OS"
      exit 1
    fi

    TGZ="$CACHE_DIR/foldseek.tar.gz"
    download "$URL" "$TGZ"

    # The tarball contains a top-level "foldseek/" directory; extract at INSTALL_DIR.
    tar -xzf "$TGZ" -C "$INSTALL_DIR"
  fi

  if [[ ! -x "$FOLDSEEK_DIR/bin/foldseek" ]]; then
    echo "ERROR: Foldseek install finished but binary not found/executable at:"
    echo "  $FOLDSEEK_DIR/bin/foldseek"
    exit 1
  fi
  echo "[Foldseek] OK: $FOLDSEEK_DIR/bin/foldseek"
  echo
fi

########################################
# Update .gitignore (local to installation/)
########################################
GITIGNORE_FILE="$INSTALL_DIR/.gitignore"

add_to_gitignore() {
  local entry="$1"
  # Ensure file exists
  touch "$GITIGNORE_FILE"
  # Add entry only if not present
  if ! grep -Fxq "$entry" "$GITIGNORE_FILE"; then
    echo "$entry" >> "$GITIGNORE_FILE"
    echo "[gitignore] added: $entry"
  fi
}

echo "Updating .gitignore to exclude installed tools..."

add_to_gitignore ".cache/"
add_to_gitignore "GEMME/"
add_to_gitignore "JET2/"
add_to_gitignore "foldseek/"

echo "== Done =="
echo "Tools installed under: $INSTALL_DIR"
echo "PRIZM will use them via your config variable: installation_folder=\"$INSTALL_DIR\""
