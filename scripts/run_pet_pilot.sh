#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Train one seeded Oxford Pet model and compare damage-aware compression with
simple SVD-energy choices. All selection uses validation; test runs last.

Usage: scripts/run_pet_pilot.sh --data-root PATH [options]
  --config PATH       Study template (default: examples/configs/pet-pilot.yaml)
  --data-root PATH    Existing torchvision Oxford-IIIT Pet directory
  --output-dir PATH   Persistent parent directory (default: runs/pet-pilot)
  --run-id ID         Reuse this ID to resume; otherwise a UTC timestamp is used
  --device DEVICE     cpu or cuda[:N] (default: cpu)
  --seeds CSV         One or three model seeds (default: 7; later: 7,17,27)
  --with-recovery     Give each selected model and dense control the same extra training
  --dry-run           Print stages without writing or loading model/dataset assets
  --help

Prepare the public dataset separately. The runner never downloads assets.
Failed candidates remain in logs and selection.json; completion is not proof of
better quality or latency. Use the same run ID and options to resume.
EOF
}

CONFIG=examples/configs/pet-pilot.yaml
DATA_ROOT=
OUTPUT_DIR=runs/pet-pilot
RUN_ID=
DEVICE=cpu
SEEDS=7
WITH_RECOVERY=0
DRY_RUN=0
PYTHON=${PYTHON:-python}
while (($#)); do
  case "$1" in
    --config) CONFIG=${2:?missing config}; shift 2 ;;
    --data-root) DATA_ROOT=${2:?missing data root}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?missing output directory}; shift 2 ;;
    --run-id) RUN_ID=${2:?missing run ID}; shift 2 ;;
    --device) DEVICE=${2:?missing device}; shift 2 ;;
    --seeds) SEEDS=${2:?missing seeds}; shift 2 ;;
    --with-recovery) WITH_RECOVERY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$CONFIG" ]] || { echo "Missing template: $CONFIG" >&2; exit 2; }
[[ -n "$DATA_ROOT" ]] || { echo "--data-root is required" >&2; exit 2; }
[[ "$DEVICE" =~ ^(cpu|cuda(:[0-9]+)?)$ ]] || { echo "Invalid device: $DEVICE" >&2; exit 2; }
[[ "$SEEDS" =~ ^[0-9]+(,[0-9]+){0,2}$ ]] || { echo "Use one or three integer model seeds." >&2; exit 2; }
IFS=, read -r -a SEED_LIST <<< "$SEEDS"
[[ ${#SEED_LIST[@]} == 1 || ${#SEED_LIST[@]} == 3 ]] || { echo "Use one or three seeds." >&2; exit 2; }
for ((i=0; i<${#SEED_LIST[@]}; i++)); do
  for ((j=i+1; j<${#SEED_LIST[@]}; j++)); do
    [[ ${SEED_LIST[i]} != "${SEED_LIST[j]}" ]] || { echo "Model seeds must be distinct." >&2; exit 2; }
  done
done
[[ -n "$RUN_ID" ]] || RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Run ID must be file-safe." >&2; exit 2; }

RUN_DIR="$OUTPUT_DIR/$RUN_ID"
COMMIT=$(git rev-parse HEAD)
TEMPLATE_HASH=$(sha256sum "$CONFIG" | awk '{print $1}')
ENERGY_OUTPUT=$("$PYTHON" -m tn_compression.pet_pilot energies --config "$CONFIG")
mapfile -t ENERGY_NAMES <<< "${ENERGY_OUTPUT//$'\r'/}"

show() { printf '+ '; printf '%q ' "$@"; printf '\n'; }
run_stage() {
  local name=$1; shift
  local marker="$SEED_DIR/.stages/$name.done"
  local log="$SEED_DIR/logs/$name.log"
  local signature saved
  signature=$(printf '%s\0' "$COMMIT" "$TEMPLATE_HASH" "$DATA_ROOT" "$DEVICE" "$WITH_RECOVERY" "$@" | sha256sum | awk '{print $1}')
  if [[ -f "$marker" ]]; then
    saved=$(sed -n 's/^signature=//p' "$marker")
    [[ "$saved" == "$signature" ]] || { echo "Changed stage $name; use a new run ID." >&2; exit 2; }
    echo "Skipping completed $name"
    return 0
  fi
  if [[ "$DRY_RUN" == 1 ]]; then show "$@"; return 0; fi
  mkdir -p "$SEED_DIR/.stages" "$SEED_DIR/logs"
  show "$@" | tee "$log"
  if "$@" 2>&1 | tee -a "$log"; then
    printf 'signature=%s\ngit_commit=%s\ntemplate_sha256=%s\n' "$signature" "$COMMIT" "$TEMPLATE_HASH" > "$marker"
    rm -f "$SEED_DIR/.stages/$name.failed"
  else
    printf 'signature=%s\nlog=%s\n' "$signature" "$log" > "$SEED_DIR/.stages/$name.failed"
    return 1
  fi
}

if [[ "$DRY_RUN" == 0 ]]; then
  [[ -z $(git status --porcelain) ]] || {
    echo "Commit or remove working-tree changes before a research run; the Git SHA must identify the code." >&2
    exit 2
  }
  [[ -d "$DATA_ROOT" ]] || { echo "Dataset directory does not exist: $DATA_ROOT" >&2; exit 2; }
  mkdir -p "$RUN_DIR/environment"
  if [[ -f "$RUN_DIR/environment/git-commit.txt" ]]; then
    [[ $(cat "$RUN_DIR/environment/git-commit.txt") == "$COMMIT" ]] || {
      echo "Git revision changed; resume with the recorded revision or a new run ID." >&2; exit 2;
    }
  else
    printf '%s\n' "$COMMIT" > "$RUN_DIR/environment/git-commit.txt"
    git status --short --branch > "$RUN_DIR/environment/git-status.txt"
    "$PYTHON" -m pip freeze > "$RUN_DIR/environment/packages.txt"
    "$PYTHON" --version > "$RUN_DIR/environment/python.txt" 2>&1
    if command -v nvidia-smi >/dev/null; then nvidia-smi -q > "$RUN_DIR/environment/nvidia-smi.txt"; fi
  fi
fi

for seed in "${SEED_LIST[@]}"; do
  SEED_DIR="$RUN_DIR/seed-$seed"
  CONFIG_PATH="$SEED_DIR/resolved-config.json"
  RECOVERY_CONFIG="$SEED_DIR/recovery-config.json"
  if [[ "$DRY_RUN" == 1 ]]; then
    show "$PYTHON" -m tn_compression.pet_pilot prepare --config "$CONFIG" --data-root "$DATA_ROOT" --seed "$seed" --root "$SEED_DIR"
  else
    "$PYTHON" -m tn_compression.pet_pilot prepare --config "$CONFIG" --data-root "$DATA_ROOT" --seed "$seed" --root "$SEED_DIR"
  fi
  run_stage split-ids "$PYTHON" -m tn_compression split-ids --config "$CONFIG_PATH" --output-dir "$SEED_DIR/splits" --device "$DEVICE"
  run_stage train "$PYTHON" -m tn_compression train --config "$CONFIG_PATH" --output-dir "$SEED_DIR/train" --device "$DEVICE"
  DENSE="$SEED_DIR/train/bundle"
  run_stage dense-validation "$PYTHON" -m tn_compression evaluate --config "$CONFIG_PATH" --checkpoint "$DENSE" --role validation --output-dir "$SEED_DIR/dense-validation" --device "$DEVICE"
  run_stage dense-benchmark "$PYTHON" -m tn_compression benchmark --config "$CONFIG_PATH" --checkpoint "$DENSE" --output-dir "$SEED_DIR/dense-benchmark" --device "$DEVICE"

  if run_stage analyzer-analysis "$PYTHON" -m tn_compression analyze --config "$CONFIG_PATH" --checkpoint "$DENSE" --level validated --output-dir "$SEED_DIR/analysis" --device "$DEVICE"; then
    if [[ "$DRY_RUN" == 1 ]] || "$PYTHON" -m tn_compression.pet_pilot has-plan --root "$SEED_DIR"; then
      run_stage analyzer-compress "$PYTHON" -m tn_compression compress --config "$CONFIG_PATH" --checkpoint "$DENSE" --plan "$SEED_DIR/analysis/compression_plan.json" --output-dir "$SEED_DIR/candidates/analyzer/compress" --device "$DEVICE" || true
    fi
  fi

  for name in "analyzer" "${ENERGY_NAMES[@]}"; do
    CANDIDATE="$SEED_DIR/candidates/$name"
    if [[ "$name" != analyzer ]]; then
      TRIAL_CONFIG="$SEED_DIR/configs/$name.json"
      run_stage "$name-compress" "$PYTHON" -m tn_compression compress --config "$TRIAL_CONFIG" --checkpoint "$DENSE" --output-dir "$CANDIDATE/compress" --device "$DEVICE" || true
    fi
    if [[ "$DRY_RUN" == 1 || -f "$CANDIDATE/compress/bundle/manifest.json" ]]; then
      run_stage "$name-validation" "$PYTHON" -m tn_compression evaluate --config "$CONFIG_PATH" --checkpoint "$CANDIDATE/compress/bundle" --role validation --output-dir "$CANDIDATE/validation" --device "$DEVICE" || true
      run_stage "$name-benchmark" "$PYTHON" -m tn_compression benchmark --config "$CONFIG_PATH" --checkpoint "$CANDIDATE/compress/bundle" --output-dir "$CANDIDATE/benchmark" --device "$DEVICE" || true
    fi
  done
  run_stage select "$PYTHON" -m tn_compression.pet_pilot select --root "$SEED_DIR"

  if [[ "$WITH_RECOVERY" == 1 ]]; then
    run_stage dense-control "$PYTHON" -m tn_compression finetune --config "$RECOVERY_CONFIG" --checkpoint "$DENSE" --output-dir "$SEED_DIR/dense-control" --device "$DEVICE"
    run_stage dense-control-validation "$PYTHON" -m tn_compression evaluate --config "$CONFIG_PATH" --checkpoint "$SEED_DIR/dense-control/bundle" --role validation --output-dir "$SEED_DIR/dense-control-validation" --device "$DEVICE"
    run_stage dense-control-benchmark "$PYTHON" -m tn_compression benchmark --config "$CONFIG_PATH" --checkpoint "$SEED_DIR/dense-control/bundle" --output-dir "$SEED_DIR/dense-control-benchmark" --device "$DEVICE"
  fi
  if [[ "$DRY_RUN" == 1 ]]; then
    SELECTED=(analyzer "${ENERGY_NAMES[0]}")
  else
    SELECTED_OUTPUT=$("$PYTHON" -m tn_compression.pet_pilot selected --root "$SEED_DIR")
    SELECTED=()
    if [[ -n "$SELECTED_OUTPUT" ]]; then mapfile -t SELECTED <<< "${SELECTED_OUTPUT//$'\r'/}"; fi
  fi
  for name in "${SELECTED[@]}"; do
    CANDIDATE="$SEED_DIR/candidates/$name"
    if [[ "$WITH_RECOVERY" == 1 ]]; then
      run_stage "$name-recovery" "$PYTHON" -m tn_compression finetune --config "$RECOVERY_CONFIG" --checkpoint "$CANDIDATE/compress/bundle" --output-dir "$CANDIDATE/recovery" --device "$DEVICE"
      run_stage "$name-recovery-validation" "$PYTHON" -m tn_compression evaluate --config "$CONFIG_PATH" --checkpoint "$CANDIDATE/recovery/bundle" --role validation --output-dir "$CANDIDATE/recovery-validation" --device "$DEVICE"
      run_stage "$name-recovery-benchmark" "$PYTHON" -m tn_compression benchmark --config "$CONFIG_PATH" --checkpoint "$CANDIDATE/recovery/bundle" --output-dir "$CANDIDATE/recovery-benchmark" --device "$DEVICE"
    fi
  done
  run_stage dense-test "$PYTHON" -m tn_compression evaluate --config "$CONFIG_PATH" --checkpoint "$DENSE" --role test --output-dir "$SEED_DIR/dense-test" --device "$DEVICE"
  if [[ "$WITH_RECOVERY" == 1 ]]; then
    run_stage dense-control-test "$PYTHON" -m tn_compression evaluate --config "$CONFIG_PATH" --checkpoint "$SEED_DIR/dense-control/bundle" --role test --output-dir "$SEED_DIR/dense-control-test" --device "$DEVICE"
  fi
  for name in "${SELECTED[@]}"; do
    CANDIDATE="$SEED_DIR/candidates/$name"
    run_stage "$name-test" "$PYTHON" -m tn_compression evaluate --config "$CONFIG_PATH" --checkpoint "$CANDIDATE/compress/bundle" --role test --output-dir "$CANDIDATE/test" --device "$DEVICE"
    if [[ "$WITH_RECOVERY" == 1 ]]; then
      run_stage "$name-recovery-test" "$PYTHON" -m tn_compression evaluate --config "$CONFIG_PATH" --checkpoint "$CANDIDATE/recovery/bundle" --role test --output-dir "$CANDIDATE/recovery-test" --device "$DEVICE"
    fi
  done
  run_stage report "$PYTHON" -m tn_compression.pet_pilot report --root "$SEED_DIR"
  if [[ "$DRY_RUN" == 0 ]]; then echo "Inspect $SEED_DIR/selection.json and $SEED_DIR/final.json"; fi
done
