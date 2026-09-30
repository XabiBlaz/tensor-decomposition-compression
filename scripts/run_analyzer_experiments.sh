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
  --vision-checkpoint PATH Trained vision bundle (required for the vision suite)
  --allow-synthetic-baseline Explicitly allow an untrained vision baseline; report as synthetic
  --language-config PATH Language workflow YAML
  --with-recovery       Run finetune after compression (config must support it)
  --allow-code-change   Resume across commits while retaining per-stage revisions
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
ALLOW_CODE_CHANGE=0
DRY_RUN=0
VISION_CHECKPOINT=
ALLOW_SYNTHETIC_BASELINE=0

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
    --vision-checkpoint) VISION_CHECKPOINT=${2:?missing vision checkpoint}; shift 2 ;;
    --allow-synthetic-baseline) ALLOW_SYNTHETIC_BASELINE=1; shift ;;
    --with-recovery) WITH_RECOVERY=1; shift ;;
    --allow-code-change) ALLOW_CODE_CHANGE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SUITE" in
  smoke|vision|language|all) ;;
  *) echo "Unknown suite: $SUITE" >&2; exit 2 ;;
esac

if [[ "$SUITE" == vision || "$SUITE" == all ]]; then
  if [[ -z "$VISION_CHECKPOINT" && "$ALLOW_SYNTHETIC_BASELINE" != 1 ]]; then
    echo "Vision requires --vision-checkpoint with a trained bundle; use --allow-synthetic-baseline only for integration checks." >&2
    exit 2
  fi
  if [[ -n "$VISION_CHECKPOINT" && ( ! -f "$VISION_CHECKPOINT/manifest.json" || ! -f "$VISION_CHECKPOINT/weights.pt" ) ]]; then
    echo "Vision checkpoint must be a complete bundle containing manifest.json and weights.pt." >&2
    exit 2
  fi
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Run ID may contain only letters, numbers, dots, underscores and hyphens." >&2
  exit 2
fi
RUN_ROOT="$OUTPUT_DIR/$RUN_ID"
CURRENT_COMMIT=$(git rev-parse HEAD)
CLI=(python -m tn_compression)
ACTIVE_CONFIG_HASH=

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

hash_text() {
  sha256sum | awk '{print $1}'
}

stage_signature() {
  local command_text
  printf -v command_text '%q ' "$@"
  printf '%s\0%s' "$ACTIVE_CONFIG_HASH" "$command_text" | hash_text
}

run_stage() {
  local suite_dir=$1
  local name=$2
  shift 2
  local marker="$suite_dir/.stages/$name.done"
  local log="$suite_dir/logs/$name.log"
  local signature command_text recorded_signature recorded_commit
  signature=$(stage_signature "$@")
  printf -v command_text '%q ' "$@"
  if [[ -f "$marker" ]]; then
    recorded_signature=$(sed -n 's/^signature=//p' "$marker")
    recorded_commit=$(sed -n 's/^git_commit=//p' "$marker")
    if [[ -z "$recorded_signature" || "$recorded_signature" != "$signature" ]]; then
      echo "Refusing to reuse stage $name: completion marker does not match its configuration and command." >&2
      exit 2
    fi
    echo "Skipping completed stage $name (commit ${recorded_commit:-unknown})"
    return
  fi
  if [[ "$DRY_RUN" == 1 ]]; then
    print_command "$@"
    return
  fi
  mkdir -p "$suite_dir/.stages" "$suite_dir/logs"
  print_command "$@" | tee "$log"
  "$@" 2>&1 | tee -a "$log"
  {
    printf 'signature=%s\n' "$signature"
    printf 'git_commit=%s\n' "$CURRENT_COMMIT"
    printf 'config_sha256=%s\n' "$ACTIVE_CONFIG_HASH"
    printf 'command=%s\n' "$command_text"
  } > "$marker.tmp"
  mv "$marker.tmp" "$marker"
}

record_environment() {
  local recorded_file="$RUN_ROOT/environment/git-commit.txt"
  local recorded_commit=
  if [[ -f "$recorded_file" ]]; then
    recorded_commit=$(cat "$recorded_file")
    if [[ "$recorded_commit" != "$CURRENT_COMMIT" && "$ALLOW_CODE_CHANGE" != 1 ]]; then
      echo "Refusing to resume run $RUN_ID: recorded commit $recorded_commit differs from $CURRENT_COMMIT." >&2
      echo "Use --allow-code-change to retain per-stage revision provenance." >&2
      exit 2
    fi
  fi
  if [[ "$DRY_RUN" == 1 ]]; then
    print_command git rev-parse HEAD
    print_command python -m pip freeze
    print_command nvidia-smi
    if [[ -n "$recorded_commit" && "$recorded_commit" != "$CURRENT_COMMIT" ]]; then
      echo "+ record mixed-revision resume: $recorded_commit -> $CURRENT_COMMIT"
    fi
    return
  fi
  mkdir -p "$RUN_ROOT/environment"
  if [[ ! -f "$recorded_file" ]]; then
    printf '%s\n' "$CURRENT_COMMIT" > "$recorded_file"
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
  if [[ ! -f "$RUN_ROOT/environment/commits-used.txt" ]]; then
    printf '%s\n' "${recorded_commit:-$CURRENT_COMMIT}" > "$RUN_ROOT/environment/commits-used.txt"
  fi
  grep -Fxq "$CURRENT_COMMIT" "$RUN_ROOT/environment/commits-used.txt" || \
    printf '%s\n' "$CURRENT_COMMIT" >> "$RUN_ROOT/environment/commits-used.txt"
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
  ACTIVE_CONFIG_HASH=$(sha256sum "$config" | awk '{print $1}')
  local source_args=()
  local report_args=()
  if [[ "$name" == vision && -n "$VISION_CHECKPOINT" ]]; then
    source_args=(--checkpoint "$VISION_CHECKPOINT")
    # A changed checkpoint invalidates stage reuse, even at the same path.
    ACTIVE_CONFIG_HASH=$(printf '%s\n%s\n' "$ACTIVE_CONFIG_HASH" "$(sha256sum "$VISION_CHECKPOINT/manifest.json" "$VISION_CHECKPOINT/weights.pt")" | hash_text)
  elif [[ "$name" == vision || "$name" == smoke ]]; then
    report_args=(--synthetic-baseline)
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

  # Freeze the exact starting weights once; all analysis and measurements reload them.
  run_stage "$suite_dir" original-snapshot "${CLI[@]}" snapshot \
    --config "$resolved" "${source_args[@]}" --output-dir "$suite_dir/original" --device "$DEVICE"
  local original="$suite_dir/original/bundle"
  run_stage "$suite_dir" original-evaluation "${CLI[@]}" evaluate \
    --config "$resolved" --checkpoint "$original" \
    --output-dir "$suite_dir/original-evaluation" --role test --device "$DEVICE"
  run_stage "$suite_dir" original-benchmark "${CLI[@]}" benchmark \
    --config "$resolved" --checkpoint "$original" \
    --output-dir "$suite_dir/original-benchmark" --device "$DEVICE"

  # Bounded activation collection and local reconstruction screening.
  run_stage "$suite_dir" calibrated-screening "${CLI[@]}" analyze \
    --config "$resolved" --output-dir "$suite_dir/calibrated" \
    --level calibrated --checkpoint "$original" --device "$DEVICE"

  # Full task interventions, greedy selection and cumulative rollback.
  run_stage "$suite_dir" validated-analysis "${CLI[@]}" analyze \
    --config "$resolved" --output-dir "$suite_dir/analysis" \
    --level validated --checkpoint "$original" --device "$DEVICE"

  # An unsuccessful search is a reportable outcome, not a compression success.
  if [[ "$DRY_RUN" == 0 ]]; then
    local plan_exit=0
    python -m tn_compression.comparison --run-dir "$suite_dir" --check-plan "${report_args[@]}" || plan_exit=$?
    if [[ "$plan_exit" == 3 ]]; then
      return
    elif [[ "$plan_exit" != 0 ]]; then
      return "$plan_exit"
    fi
  else
    print_command python -m tn_compression.comparison --run-dir "$suite_dir" --check-plan "${report_args[@]}"
  fi

  run_stage "$suite_dir" compression "${CLI[@]}" compress \
    --config "$resolved" --plan "$suite_dir/analysis/compression_plan.json" \
    --output-dir "$suite_dir/compression" --checkpoint "$original" --device "$DEVICE"

  local checkpoint="$suite_dir/compression/bundle"
  if [[ "$WITH_RECOVERY" == 1 ]]; then
    run_stage "$suite_dir" recovery "${CLI[@]}" finetune \
      --config "$resolved" --checkpoint "$checkpoint" \
      --output-dir "$suite_dir/recovery" --device "$DEVICE"
    checkpoint="$suite_dir/recovery/bundle"
  elif [[ "$DRY_RUN" == 0 ]]; then
    mkdir -p "$suite_dir/.stages"
    {
      echo "Recovery intentionally disabled; pass --with-recovery to enable it."
      printf 'git_commit=%s\nconfig_sha256=%s\n' "$CURRENT_COMMIT" "$ACTIVE_CONFIG_HASH"
    } > "$suite_dir/.stages/recovery.skipped"
  fi

  run_stage "$suite_dir" evaluation "${CLI[@]}" evaluate \
    --config "$resolved" --checkpoint "$checkpoint" \
    --output-dir "$suite_dir/evaluation" --role test --device "$DEVICE"

  run_stage "$suite_dir" benchmark "${CLI[@]}" benchmark \
    --config "$resolved" --checkpoint "$checkpoint" \
    --output-dir "$suite_dir/benchmark" --device "$DEVICE"

  run_stage "$suite_dir" comparison python -m tn_compression.comparison \
    --run-dir "$suite_dir" --compressed-bundle "$checkpoint" "${report_args[@]}"
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
  echo "Run complete: $RUN_ROOT. Inspect each comparison.json for feasibility and final quality; completion is not compression success."
fi
