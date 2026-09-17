#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the damage-aware analyzer experiments from explicit, pinned configurations.

Usage:
  scripts/run_analyzer_experiments.sh [options]

Options:
  --suite NAME          smoke, vision, language or all (default: smoke)
  --device DEVICE       PyTorch device passed to tn-compress (default: cpu)
  --output-dir PATH     Parent directory for unique runs (default: runs/analyzer)
  --run-id ID           Resume or create this run ID; omitted IDs use a timestamp
  --model-cache PATH    Existing Hugging Face/Torch cache; network stays disabled
  --data-cache PATH     Existing Hugging Face datasets cache; network stays disabled
  --smoke-config PATH   Smoke workflow YAML
  --vision-config PATH  Vision workflow YAML
  --language-config PATH Language workflow YAML
  --with-recovery       Run finetune after compression (config must support it)
  --dry-run             Print commands without creating files or running work
  --help                Show this help

The script never enables downloads. Prepare all model and dataset assets first.
Vision can take tens of minutes on CPU. The language suite normally needs a GPU
and several GB of model, activation and candidate-factor storage. Benchmarking is
run only after a plan has been applied and a reconstructible bundle exists.
EOF
}

SUITE=smoke
DEVICE=cpu
OUTPUT_DIR=runs/analyzer
RUN_ID=
MODEL_CACHE=
DATA_CACHE=
SMOKE_CONFIG=examples/configs/analyzer-smoke.yaml
VISION_CONFIG=examples/configs/analyzer-vision.yaml
LANGUAGE_CONFIG=examples/configs/analyzer-language.yaml
WITH_RECOVERY=0
DRY_RUN=0

while (($#)); do
  case "$1" in
    --suite) SUITE=${2:?missing suite}; shift 2 ;;
    --device) DEVICE=${2:?missing device}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?missing output directory}; shift 2 ;;
    --run-id) RUN_ID=${2:?missing run ID}; shift 2 ;;
    --model-cache) MODEL_CACHE=${2:?missing model cache}; shift 2 ;;
    --data-cache) DATA_CACHE=${2:?missing data cache}; shift 2 ;;
    --smoke-config) SMOKE_CONFIG=${2:?missing smoke config}; shift 2 ;;
    --vision-config) VISION_CONFIG=${2:?missing vision config}; shift 2 ;;
    --language-config) LANGUAGE_CONFIG=${2:?missing language config}; shift 2 ;;
    --with-recovery) WITH_RECOVERY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SUITE" in
  smoke|vision|language|all) ;;
  *) echo "Unknown suite: $SUITE" >&2; exit 2 ;;
esac

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Run ID may contain only letters, numbers, dots, underscores and hyphens." >&2
  exit 2
fi
RUN_ROOT="$OUTPUT_DIR/$RUN_ID"
CLI=(python -m tn_compression)

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
[[ -n "$MODEL_CACHE" ]] && export HF_HOME="$MODEL_CACHE" TORCH_HOME="$MODEL_CACHE/torch"
[[ -n "$DATA_CACHE" ]] && export HF_DATASETS_CACHE="$DATA_CACHE"

print_command() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
}

run_stage() {
  local suite_dir=$1
  local name=$2
  shift 2
  local marker="$suite_dir/.stages/$name.done"
  local log="$suite_dir/logs/$name.log"
  if [[ "$DRY_RUN" == 1 ]]; then
    print_command "$@"
    return
  fi
  if [[ -f "$marker" ]]; then
    echo "Skipping completed stage $name"
    return
  fi
  mkdir -p "$suite_dir/.stages" "$suite_dir/logs"
  print_command "$@" | tee "$log"
  "$@" 2>&1 | tee -a "$log"
  touch "$marker"
}

record_environment() {
  if [[ "$DRY_RUN" == 1 ]]; then
    print_command git rev-parse HEAD
    print_command python -m pip freeze
    print_command nvidia-smi
    return
  fi
  mkdir -p "$RUN_ROOT/environment"
  if [[ ! -f "$RUN_ROOT/environment/git-commit.txt" ]]; then
    git rev-parse HEAD > "$RUN_ROOT/environment/git-commit.txt"
    git status --short --branch > "$RUN_ROOT/environment/git-status.txt"
    python --version > "$RUN_ROOT/environment/python.txt" 2>&1
    python -m pip freeze > "$RUN_ROOT/environment/packages.txt"
    uname -a > "$RUN_ROOT/environment/uname.txt"
    command -v lscpu >/dev/null && lscpu > "$RUN_ROOT/environment/cpu.txt"
    if command -v nvidia-smi >/dev/null; then
      nvidia-smi -q > "$RUN_ROOT/environment/nvidia-smi.txt"
    else
      echo "nvidia-smi unavailable" > "$RUN_ROOT/environment/nvidia-smi.txt"
    fi
  fi
}

run_suite() {
  local name=$1
  local config=$2
  local suite_dir="$RUN_ROOT/$name"
  local resolved="$suite_dir/resolved-config.yaml"
  if [[ ! -f "$config" ]]; then
    echo "Configuration does not exist: $config" >&2
    exit 2
  fi
  if [[ "$DRY_RUN" == 1 ]]; then
    print_command cp "$config" "$resolved"
  else
    mkdir -p "$suite_dir"
    if [[ -f "$resolved" ]] && ! cmp -s "$config" "$resolved"; then
      echo "Refusing to resume $name with a different configuration." >&2
      exit 2
    fi
    [[ -f "$resolved" ]] || cp "$config" "$resolved"
  fi

  # Bounded activation collection and local reconstruction screening.
  run_stage "$suite_dir" calibrated-screening "${CLI[@]}" analyze \
    --config "$resolved" --output-dir "$suite_dir/calibrated" \
    --level calibrated --device "$DEVICE"

  # Full task interventions, greedy selection and cumulative rollback.
  run_stage "$suite_dir" validated-analysis "${CLI[@]}" analyze \
    --config "$resolved" --output-dir "$suite_dir/analysis" \
    --level validated --device "$DEVICE"

  run_stage "$suite_dir" compression "${CLI[@]}" compress \
    --config "$resolved" --plan "$suite_dir/analysis/compression_plan.json" \
    --output-dir "$suite_dir/compression" --device "$DEVICE"

  local checkpoint="$suite_dir/compression/bundle"
  if [[ "$WITH_RECOVERY" == 1 ]]; then
    run_stage "$suite_dir" recovery "${CLI[@]}" finetune \
      --config "$resolved" --checkpoint "$checkpoint" \
      --output-dir "$suite_dir/recovery" --device "$DEVICE"
    checkpoint="$suite_dir/recovery/bundle"
  elif [[ "$DRY_RUN" == 0 ]]; then
    mkdir -p "$suite_dir/.stages"
    echo "Recovery intentionally disabled; pass --with-recovery to enable it." \
      > "$suite_dir/.stages/recovery.skipped"
  fi

  run_stage "$suite_dir" evaluation "${CLI[@]}" evaluate \
    --config "$resolved" --checkpoint "$checkpoint" \
    --output-dir "$suite_dir/evaluation" --role test --device "$DEVICE"

  run_stage "$suite_dir" benchmark "${CLI[@]}" benchmark \
    --config "$resolved" --checkpoint "$checkpoint" \
    --output-dir "$suite_dir/benchmark" --device "$DEVICE"
}

record_environment
case "$SUITE" in
  smoke) run_suite smoke "$SMOKE_CONFIG" ;;
  vision) run_suite vision "$VISION_CONFIG" ;;
  language) run_suite language "$LANGUAGE_CONFIG" ;;
  all)
    run_suite smoke "$SMOKE_CONFIG"
    run_suite vision "$VISION_CONFIG"
    run_suite language "$LANGUAGE_CONFIG"
    ;;
esac

if [[ "$DRY_RUN" == 0 ]]; then
  touch "$RUN_ROOT/complete"
  echo "Run complete: $RUN_ROOT"
fi
