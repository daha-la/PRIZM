#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# PRIZM global submission script
#
# Integrates:
# - CSV → MSA_filename mapping via Python helper (tmp TSV, auto-cleanup)
# - DONE markers keyed by MSA identity (msa_id = MSA_filename stripped after first dot)
#   (reuse training across datasets/libraries)
# - Conda env selection per model (conda run -n <env>)
# - Training runs sequentially (EVE then eUniRep), only if needed by selected models
# - Scoring runs sequential or background, strict order, GPU capacity scheduling
# - --only, --list-models, --dry-run
###############################################################################

# ============================ Defaults =======================================
FIRST_INDEX=2
LAST_INDEX=2
MODE="sequential"            # sequential | background
ONLY_MODELS=""
OVERWRITE_TRAINING=false

# Background scheduling (scoring only)
GPU_CAPACITY_UNITS=16
SLEEP_BETWEEN=2
WAIT_INTERVAL=20

# UX flags
SUBMIT_NOOP=false
LIST_MODELS=false
DRY_RUN=false
KEEP_TMP=false

usage() {
  cat <<EOF
Usage: $0 --first N --last M [options]

Options:
  --first N                 First DMS index (0-based)
  --last M                  Last DMS index (inclusive)
  --mode sequential|background
  --only m1,m2,...           Run only a subset of SCORING models (comma-separated keys)
                             Training will run ONLY if required by selected models.
  --overwrite-training       Force retraining (ignore DONE markers)
  --gpu-capacity N           GPU capacity units for background scoring (default: 16)
  --submit-noop              Submit scoring jobs as no-op commands (tests submission/logging/envs)
  --list-models              Print available model keys and exit
  --dry-run                  Print the plan and exit (no commands run)
  --keep-tmp                 Keep the generated tmp TSV (debugging)
EOF
}

die(){ echo "ERROR: $*" >&2; exit 1; }

# ============================ Parse CLI ======================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --first) FIRST_INDEX="$2"; shift 2;;
    --last)  LAST_INDEX="$2"; shift 2;;
    --mode)  MODE="$2"; shift 2;;
    --only)  ONLY_MODELS="$2"; shift 2;;
    --gpu-capacity) GPU_CAPACITY_UNITS="$2"; shift 2;;
    --overwrite-training) OVERWRITE_TRAINING=true; shift;;
    --submit-noop) SUBMIT_NOOP=true; shift;;
    --list-models) LIST_MODELS=true; shift;;
    --dry-run) DRY_RUN=true; shift;;
    --keep-tmp) KEEP_TMP=true; shift;;
    -h|--help) usage; exit 0;;
    *) die "Unknown argument: $1";;
  esac
done

[[ "$MODE" == "sequential" || "$MODE" == "background" ]] || die "--mode must be sequential|background"
[[ "$GPU_CAPACITY_UNITS" =~ ^[0-9]+$ ]] || die "--gpu-capacity must be an integer"
(( GPU_CAPACITY_UNITS >= 0 )) || die "--gpu-capacity must be >= 0"

# ============================ Paths & config =================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logfiles"
SCORING_DIR="${ROOT_DIR}/proteingym/scripts/scoring_DMS_zero_shot"
CONFIG_SH="${ROOT_DIR}/proteingym/scripts/zero_shot_config.sh"
HELPER_PY="${SCRIPT_DIR}/utils/ref_msa_map.py"

mkdir -p "$LOG_DIR"

TMP_DIR="${SCRIPT_DIR}/tmp"
mkdir -p "$TMP_DIR"

# shellcheck disable=SC1090
source "$CONFIG_SH"

[[ -f "$HELPER_PY" ]] || die "Python helper not found: $HELPER_PY"

# ============================ Model registry =================================
declare -A MODEL_TO_SCRIPT=(
  ["eve_train"]="training_EVE_models.sh"
  ["unirep_evotune"]="evotune_UniRep_substitutions.sh"

  ["carp"]="scoring_CARP_substitutions.sh"
  ["esm_if1"]="scoring_ESM_IF1_substitutions.sh"
  ["esm1b"]="scoring_ESM1b_substitutions.sh"
  ["esm1v"]="scoring_ESM1v_substitutions.sh"
  ["esm2"]="scoring_ESM2_substitutions.sh"
  ["eve"]="scoring_EVE_substitutions.sh"
  ["gemme"]="scoring_GEMME_substitutions.sh"
  ["mif"]="scoring_MIF_substitutions.sh"
  ["mifst"]="scoring_MIFST_substitutions.sh"
  ["msa_transformer"]="scoring_MSA_transformer_substitutions.sh"
  ["mulan"]="scoring_MULAN_substitutions.sh"
  ["poet"]="scoring_PoET_substitutions.sh"
  ["progen2"]="scoring_Progen2_substitutions.sh"
  ["prosst"]="scoring_ProSST_substitutions.sh"
  ["proteinmpnn"]="scoring_ProteinMPNN_substitutions.sh"
  ["protgpt2"]="scoring_ProtGPT2_substitutions.sh"
  ["protssn"]="scoring_ProtSSN_substitutions.sh"
  ["rita"]="scoring_RITA_substitutions.sh"
  ["rsalor"]="scoring_RSALOR_substitutions.sh"
  ["saprot"]="scoring_SaProt_substitutions.sh"
  ["tranception"]="scoring_Tranception_substitutions.sh"
  ["tranception_no_retrieval"]="scoring_Tranception_substitutions_no_retrieval.sh"
  ["trancepteve"]="scoring_TranceptEVE_substitutions.sh"
  ["unirep"]="scoring_UniRep_substitutions.sh"
  ["unirep_evotune_scoring"]="scoring_UniRep_evotune_substitutions.sh"
  ["venusrem"]="scoring_VenusREM_substitutions.sh"
)

model_env() {
  local m="$1"
  case "$m" in
    venusrem) echo "venusrem" ;;
    prosst) echo "prosst" ;;
    mulan) echo "mulan" ;;
    unirep|unirep_evotune|unirep_evotune_scoring) echo "unirep_evotune" ;;
    protssn) echo "protssn" ;;
    rsalor) echo "rsalor" ;;
    *) echo "PRIZM" ;;
  esac
}

# Dependencies (used to decide whether to run training at all)
EVE_DEPENDENTS=(
  "eve"
  "gemme"
  "msa_transformer"
  "rsalor"
  "trancepteve"
  "tranception"
  "venusrem"
  "unirep_evotune"
  "unirep_evotune_scoring"
)

EUNIREP_DEPENDENTS=(
  "unirep_evotune_scoring"
)

is_in_list() {
  local x="$1"; shift
  for y in "$@"; do [[ "$x" == "$y" ]] && return 0; done
  return 1
}

# ============================ GPU cost model =================================
declare -A GPU_COST=(
  ["gemme"]=0
  ["rsalor"]=1
  ["proteinmpnn"]=2
  ["mif"]=2
  ["mifst"]=2
  ["eve"]=3
  ["eve_train"]=3
  ["carp"]=3
  ["msa_transformer"]=3.5
  ["esm1v"]=4
  ["esm1b"]=4
  ["protgpt2"]=4
  ["mulan"]=4

  ["protssn"]=4.5
  ["venusrem"]=5
  ["prosst"]=5
  ["saprot"]=5
  ["tranception"]=8
  ["trancepteve"]=8
  ["tranception_no_retrieval"]=8

  ["rita"]=10
  ["progen2"]=12
  ["esm2"]=12
  ["esm_if1"]=14
  ["unirep"]=15
  ["unirep_evotune"]=15
  ["unirep_evotune_scoring"]=15
)

# ============================ Default scoring order ===========================
SCORING_ORDER_DEFAULT=(
  "gemme"
  "rsalor"
  "proteinmpnn"
  "mif"
  "mifst"
  "eve"
  "carp"
  "msa_transformer"
  "esm1b"
  "esm1v"
  "protgpt2"
  "mulan"
  "protssn"
  "prosst"
  "venusrem"
  "saprot"
  "tranception"
  "trancepteve"
  "tranception_no_retrieval"
  "rita"
  "progen2"
  "esm2"
  "esm_if1"
  "unirep"
  "unirep_evotune_scoring"
)

SCORING_ORDER=("${SCORING_ORDER_DEFAULT[@]}")
if [[ -n "$ONLY_MODELS" ]]; then
  IFS=',' read -r -a SCORING_ORDER <<< "$ONLY_MODELS"
fi

# ============================ Fast exits =====================================
if $LIST_MODELS; then
  echo "Available model keys:"
  for k in "${!MODEL_TO_SCRIPT[@]}"; do
    echo "  - $k"
  done | sort
  echo
  echo "Notes:"
  echo "  - Training keys are: eve_train, unirep_evotune"
  echo "  - --only controls scoring models; training runs only if needed by selected scoring models."
  exit 0
fi

# Validate indices for real runs and dry-run
[[ "$FIRST_INDEX" =~ ^[0-9]+$ ]] || die "--first must be an integer"
[[ "$LAST_INDEX"  =~ ^[0-9]+$ ]] || die "--last must be an integer"
(( FIRST_INDEX <= LAST_INDEX )) || die "--first must be <= --last"

# ============================ tmux safeguard (sequential) =====================
if [[ "$MODE" == "sequential" && -z "${TMUX:-}" && "$DRY_RUN" == "false" ]]; then
  echo
  echo "WARNING:"
  echo "  You are running SEQUENTIAL mode outside tmux."
  echo "  If SSH disconnects, the run will stop."
  echo
  echo "  Recommended:"
  echo "    tmux new -s prizm"
  echo "    bash $0 --first ${FIRST_INDEX} --last ${LAST_INDEX} [other args]"
  echo
fi

# ============================ Determine training needs ========================
NEED_EVE=false
NEED_EUNIREP=false

for m in "${SCORING_ORDER[@]}"; do
  is_in_list "$m" "${EVE_DEPENDENTS[@]}" && NEED_EVE=true
  is_in_list "$m" "${EUNIREP_DEPENDENTS[@]}" && NEED_EUNIREP=true
done

# eUniRep implicitly requires EVE
if $NEED_EUNIREP; then
  NEED_EVE=true
fi

# ============================ Tmp TSV generation =============================
TMP_TSV=""
cleanup_tmp() {
  if $KEEP_TMP; then
    [[ -n "$TMP_TSV" ]] && echo "Keeping tmp TSV: $TMP_TSV"
    return
  fi
  [[ -n "$TMP_TSV" && -f "$TMP_TSV" ]] && rm -f "$TMP_TSV"
}
trap cleanup_tmp EXIT

generate_tmp_tsv() {
  TMP_TSV="$(mktemp --tmpdir="$TMP_DIR" prizm_msa_map_XXXXXX.tsv)"
  python "$HELPER_PY" \
    --ref "$DMS_reference_file_path_subs" \
    --first "$FIRST_INDEX" \
    --last "$LAST_INDEX" \
    > "$TMP_TSV"
}

# DONE marker dirs keyed by MSA identity (msa_id = strip after first dot)
EVE_DONE_BASE="${PRIZM_PATH}/finetuned_models/EVE/done"
EUNIREP_DONE_BASE="${PRIZM_PATH}/finetuned_models/eUniRep/done"
mkdir -p "$EVE_DONE_BASE" "$EUNIREP_DONE_BASE"

eve_done_file() {
  local msa_id="$1"
  echo "${EVE_DONE_BASE}/${msa_id}.txt"
}

eunirep_done_file() {
  local msa_id="$1"
  echo "${EUNIREP_DONE_BASE}/${msa_id}.txt"
}


write_done() {
  local donefile="$1"
  local msa="$2"
  local rep_idx="$3"
  local phase="$4"
  local tmp="${donefile}.tmp"

  {
    echo "ok $(date -Iseconds)"
    echo "phase=${phase}"
    echo "msa=${msa}"
    echo "rep_index=${rep_idx}"
    echo "ref=${DMS_reference_file_path_subs}"
    echo "host=$(hostname)"
  } > "$tmp"
  mv "$tmp" "$donefile"
}

# ============================ Conda runner ===================================
run_in_env() {
  local env="$1"; shift
  conda run -n "$env" "$@"
}

# ============================ Background scheduler (scoring) ==================
PIDS=()
PIDS_COST=()

prune_jobs() {
  local np=() nc=()
  for i in "${!PIDS[@]}"; do
    if kill -0 "${PIDS[$i]}" 2>/dev/null; then
      np+=("${PIDS[$i]}")
      nc+=("${PIDS_COST[$i]}")
    fi
  done
  PIDS=("${np[@]}")
  PIDS_COST=("${nc[@]}")
}

current_gpu_usage() {
  local sum=0
  for c in "${PIDS_COST[@]}"; do sum=$((sum + c)); done
  echo "$sum"
}

wait_for_capacity() {
  local cost="$1"
  while true; do
    prune_jobs
    local used; used="$(current_gpu_usage)"
    if (( used + cost <= GPU_CAPACITY_UNITS )); then
      return
    fi
    echo "Waiting for GPU capacity: used=${used}/${GPU_CAPACITY_UNITS}, need=${cost} (sleep ${WAIT_INTERVAL}s)"
    sleep "$WAIT_INTERVAL"
  done
}

timestamp(){ date +'%Y%m%d_%H%M%S'; }

# ============================ Execution: training =============================
run_training_phase() {
  local phase="$1"      # "EVE" or "eUniRep"
  local model_key="$2"  # eve_train or unirep_evotune

  echo
  echo "=== Training phase: ${phase} ==="

  local env; env="$(model_env "$model_key")"
  local script="${MODEL_TO_SCRIPT[$model_key]}"
  [[ -f "${SCORING_DIR}/${script}" ]] || die "Training script not found: ${SCORING_DIR}/${script}"

  [[ -n "$TMP_TSV" && -f "$TMP_TSV" ]] || die "Internal error: TMP_TSV missing"

  while IFS=$'\t' read -r msa rep_idx; do
    [[ -n "$msa" ]] || continue

    # MSA identity used for DONE marker folder
    local msa_id="${msa%%.*}"

    local donefile
    if [[ "$phase" == "EVE" ]]; then
      donefile="$(eve_done_file "$msa_id")"
    else
      donefile="$(eunirep_done_file "$msa_id")"
    fi

    if ! $OVERWRITE_TRAINING && [[ -f "$donefile" ]]; then
      echo "✓ ${phase} skip (DONE exists): msa=${msa} (id=${msa_id})"
      continue
    fi

    echo "→ ${phase} train: msa=${msa} (id=${msa_id}, rep_index=${rep_idx}, env=${env})"

    if $SUBMIT_NOOP; then
      echo "[NOOP] would run: ${script} ${rep_idx} ${rep_idx} (env=${env})"
      run_in_env "$env" bash -lc \
        "echo \"[NOOP] training phase=$phase msa=$msa id=$msa_id rep_index=$rep_idx env=$env host=\$(hostname) time=\$(date -Iseconds)\"; \
        python -c 'import sys; print(\"python:\", sys.executable)'; \
        echo \"[NOOP] done\""
      echo "[NOOP] not writing DONE marker in noop mode: $donefile"
    else
      ( cd "$SCORING_DIR" && run_in_env "$env" bash "$script" "$rep_idx" "$rep_idx" )
      write_done "$donefile" "$msa" "$rep_idx" "$phase"
      echo "  Wrote DONE: $donefile"
    fi
  done < "$TMP_TSV"

  echo "=== Training phase complete: ${phase} ==="
}

# ============================ Execution: scoring ==============================
run_scoring_model() {
  local key="$1"
  local script="${MODEL_TO_SCRIPT[$key]:-}"
  [[ -n "$script" ]] || die "Unknown model key: '$key'"
  [[ -f "${SCORING_DIR}/${script}" ]] || die "Script not found: ${SCORING_DIR}/${script}"

  local env; env="$(model_env "$key")"
  local cost="${GPU_COST[$key]:-4}"
  local ts out err
  ts="$(timestamp)"
  out="${LOG_DIR}/${key}_${FIRST_INDEX}-${LAST_INDEX}_${ts}.out"
  err="${LOG_DIR}/${key}_${FIRST_INDEX}-${LAST_INDEX}_${ts}.err"

  echo "→ score $key (env=${env}, cost=${cost})"

  if [[ "$MODE" == "background" ]]; then
    wait_for_capacity "$cost"

    (
      cd "$SCORING_DIR"

      if $SUBMIT_NOOP; then
        # Run a harmless command inside the env; evaluate hostname/date INSIDE the job.
        nohup conda run -n "$env" bash -lc \
          "echo \"[NOOP] model=$key env=$env host=\$(hostname) time=\$(date -Iseconds)\"; \
           python -c 'import sys; print(\"python:\", sys.executable)'; \
           echo \"[NOOP] done\"" \
          > "$out" 2> "$err" &
      else
        nohup conda run -n "$env" bash "$script" "$FIRST_INDEX" "$LAST_INDEX" \
          > "$out" 2> "$err" &
      fi

      echo $!
    ) > "${out}.pid"

    local pid; pid="$(cat "${out}.pid")"
    PIDS+=("$pid")
    PIDS_COST+=("$cost")
    echo "  PID=$pid | used=$(current_gpu_usage)/${GPU_CAPACITY_UNITS}"
    sleep "$SLEEP_BETWEEN"

  else
    if $SUBMIT_NOOP; then
      conda run -n "$env" bash -lc \
        "echo \"[NOOP] model=$key env=$env host=\$(hostname) time=\$(date -Iseconds)\"; \
         python -c 'import sys; print(\"python:\", sys.executable)'; \
         echo \"[NOOP] done\"" \
        > "$out" 2> "$err"
    else
      ( cd "$SCORING_DIR" && conda run -n "$env" bash "$script" "$FIRST_INDEX" "$LAST_INDEX" \
        > "$out" 2> "$err" )
    fi
  fi
}


# ============================ Dry-run ========================================
if $DRY_RUN; then
  echo "DRY RUN (no commands executed)"
  echo "Indices: ${FIRST_INDEX}-${LAST_INDEX}"
  echo "Mode: $MODE"
  echo "GPU capacity units: $GPU_CAPACITY_UNITS"
  echo "Ref CSV: $DMS_reference_file_path_subs"
  echo

  echo "Selected scoring models:"
  for m in "${SCORING_ORDER[@]}"; do
    env="$(model_env "$m")"
    cost="${GPU_COST[$m]:-4}"
    need_eve_note=""
    need_eunirep_note=""
    is_in_list "$m" "${EVE_DEPENDENTS[@]}" && need_eve_note=" needs_eve"
    is_in_list "$m" "${EUNIREP_DEPENDENTS[@]}" && need_eunirep_note=" needs_eunirep"
    echo "  - $m (env=${env}, cost=${cost})${need_eve_note}${need_eunirep_note}"
  done

  echo
  echo "Training required?"
  echo "  - EVE:     $NEED_EVE"
  echo "  - eUniRep: $NEED_EUNIREP"
  echo
  echo "Training reuse policy:"
  if $OVERWRITE_TRAINING; then
    echo "  - overwrite-training: ENABLED (DONE markers will be ignored)"
  else
    echo "  - overwrite-training: disabled (existing DONE markers will be reused)"
  fi

  if $NEED_EVE || $NEED_EUNIREP; then
    echo
    echo "MSAs selected for training (from reference file):"
    generate_tmp_tsv
    echo "  (tmp TSV: $TMP_TSV)"

    while IFS=$'\t' read -r msa rep_idx; do
      [[ -n "$msa" ]] || continue
      msa_id="${msa%%.*}"
      echo "  - ${msa} (id=${msa_id}, rep_index=${rep_idx})"
    done < "$TMP_TSV"
  fi

  if $NEED_EVE || $NEED_EUNIREP; then
    echo
    echo "DONE markers (per MSA id):"

    while IFS=$'\t' read -r msa rep_idx; do
      [[ -n "$msa" ]] || continue
      msa_id="${msa%%.*}"

      if $NEED_EVE; then
        echo "  - EVE:     ${EVE_DONE_BASE}/${msa_id}.txt"
      fi

      if $NEED_EUNIREP; then
        echo "  - eUniRep: ${EUNIREP_DONE_BASE}/${msa_id}.txt"
      fi

    done < "$TMP_TSV"
  fi

  echo
  echo "Helper:"
  echo "  - $HELPER_PY  (CSV -> tmp TSV: MSA_filename <tab> rep_index)"

  exit 0
fi

# ============================ Generate MSA map TSV (only if training needed) ==
if $NEED_EVE || $NEED_EUNIREP; then
  echo "Generating tmp TSV mapping (MSA_filename -> rep_index) from reference CSV..."
  generate_tmp_tsv
  echo "Tmp TSV: $TMP_TSV"
else
  echo "No training required by selected models; skipping TSV generation."
fi

# ============================ Phase 1: training (sequential) ==================
if $NEED_EVE; then
  run_training_phase "EVE" "eve_train"
fi

if $NEED_EUNIREP; then
  run_training_phase "eUniRep" "unirep_evotune"
fi

# ============================ Phase 2: scoring ===============================
echo
echo "=== Scoring phase (mode=$MODE) ==="
for m in "${SCORING_ORDER[@]}"; do
  if [[ "$m" == "eve_train" || "$m" == "unirep_evotune" ]]; then
    echo "Skipping '$m' (training key) in scoring phase."
    continue
  fi
  run_scoring_model "$m"
done

echo "All jobs submitted/executed."
