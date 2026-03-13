#!/usr/bin/env bash
# Run full benchmark for all models across all scenarios.
# Usage: bash scripts/run_all.sh [--runs N] [--seed S]
#
# Runs each model sequentially to avoid Bedrock throttling.
# Results are saved to results/ and leaderboard is generated at the end.

set -euo pipefail

SEED="${SEED:-42}"
RUNS="${RUNS:-1}"

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --seed) SEED="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

MODELS=(
  claude-opus-4.5
  claude-opus-4.6
  claude-sonnet-4.6
  nova-premier
  llama3-3-70b
  deepseek-r1
  mistral-large
)

echo "=== Ad Arena Full Benchmark ==="
echo "Seed: $SEED | Runs per model: $RUNS"
echo "Models: ${MODELS[*]}"
echo "================================"
echo ""

# Clean old results
rm -rf results/*
echo "Cleared previous results."
echo ""

FAILED=()

for model in "${MODELS[@]}"; do
  echo "──────────────────────────────────────"
  echo "Starting: $model ($(date +%H:%M:%S))"
  echo "──────────────────────────────────────"
  START_TIME=$SECONDS
  if python examples/run_bedrock.py \
      --model "$model" \
      --benchmark \
      --seed "$SEED" \
      --runs "$RUNS"; then
    ELAPSED=$(( SECONDS - START_TIME ))
    echo "✓ $model completed in $(( ELAPSED / 60 ))m$(( ELAPSED % 60 ))s"
  else
    echo "✗ $model FAILED"
    FAILED+=("$model")
  fi
  echo ""
done

# Regenerate leaderboard from ALL saved results
echo "Rebuilding leaderboard from all results..."
python -c "
from ad_arena.benchmark.results_store import ResultsStore
store = ResultsStore()
results = store.load_all()
if results:
    store.update_leaderboard(results)
    print(f'Leaderboard updated with {len(results)} results.')
else:
    print('No results found.')
"

# Generate HTML leaderboard
echo "Generating HTML leaderboard..."
python -c "from ad_arena.cli import leaderboard_main; leaderboard_main()"

echo ""
echo "=== DONE ==="
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Failed models: ${FAILED[*]}"
else
  echo "All models completed successfully."
fi
echo "Results: results/"
echo "Leaderboard: docs/leaderboard/index.html"
