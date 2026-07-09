#!/usr/bin/env bash
# parallel_train.sh — Run Phase 2 (training) across N parallel workers,
# then merge and run Phases 3-5 sequentially.
#
# Usage:
#   bash parallel_train.sh \
#       --database classification_database.parquet \
#       --mapping structured_mapping.json \
#       --output validated_smirks.json \
#       --api-key "$GEMINI_API_KEY" \
#       --classes-file tier_recommendations_classes.json \
#       --workers 5 \
#       [-- extra args for generalize_smirks_validated.py]
#
# Everything after "--" is forwarded to each worker invocation.
# If --classes-file is not given, you can pass --classes directly in
# the extra args section.

set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────────────

DATABASE=""
MAPPING=""
OUTPUT=""
API_KEY=""
CLASSES_FILE=""
CLASSES_INLINE=""
N_WORKERS=5
MODEL="gemini-3-flash-preview"
EXTRA_ARGS=()
PYTHON="${PYTHON:-python}"
FP_N_JOBS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --database)     DATABASE="$2";      shift 2 ;;
        --mapping)      MAPPING="$2";       shift 2 ;;
        --output)       OUTPUT="$2";        shift 2 ;;
        --api-key)      API_KEY="$2";       shift 2 ;;
        --classes-file) CLASSES_FILE="$2";  shift 2 ;;
        --classes)      shift; CLASSES_INLINE=(); while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do CLASSES_INLINE+=("$1"); shift; done ;;
        --workers)      N_WORKERS="$2";     shift 2 ;;
        --model)        MODEL="$2";         shift 2 ;;
        --fp-n-jobs)    FP_N_JOBS="$2";     shift 2 ;;
        --python)       PYTHON="$2";        shift 2 ;;
        --)             shift; EXTRA_ARGS=("$@"); break ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$DATABASE" || -z "$MAPPING" || -z "$OUTPUT" ]]; then
    echo "ERROR: --database, --mapping, and --output are required."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATED_PY="$SCRIPT_DIR/generalize_smirks_validated.py"

OUTPUT_DIR="$(dirname "$OUTPUT")"
OUTPUT_STEM="$(basename "$OUTPUT" .json)"

# ── Step 1: Collect classes ──────────────────────────────────────────────────

echo "=== Step 1: Collecting classes ==="

if [[ -n "$CLASSES_FILE" ]]; then
    CLASS_SOURCE="--classes-file $CLASSES_FILE"
    # Extract class list from JSON
    ALL_CLASSES=$("$PYTHON" -c "
import json, sys
with open('$CLASSES_FILE') as f:
    data = json.load(f)
classes = data.get('all_classes', [])
print('\n'.join(classes))
")
elif [[ ${#CLASSES_INLINE[@]} -gt 0 ]]; then
    CLASS_SOURCE="--classes ${CLASSES_INLINE[*]}"
    ALL_CLASSES=$(printf '%s\n' "${CLASSES_INLINE[@]}")
else
    echo "ERROR: Either --classes-file or --classes must be provided."
    exit 1
fi

N_CLASSES=$(echo "$ALL_CLASSES" | wc -l | tr -d ' ')
echo "  Total classes: $N_CLASSES"
echo "  Workers: $N_WORKERS"

# ── Step 2: Run Phase 1 (split) once ────────────────────────────────────────

echo ""
echo "=== Step 2: Phase 1 (split) ==="

"$PYTHON" "$VALIDATED_PY" \
    --database "$DATABASE" \
    --mapping "$MAPPING" \
    --output "$OUTPUT" \
    $CLASS_SOURCE \
    --model "$MODEL" \
    --phase split \
    "${EXTRA_ARGS[@]}"

echo "  Split done."

# ── Step 3: Split classes into N chunks and launch workers ───────────────────

echo ""
echo "=== Step 3: Phase 2 (training) — $N_WORKERS parallel workers ==="

# Generate chunk files and seed worker checkpoints from existing progress
CHUNK_DIR="$OUTPUT_DIR/${OUTPUT_STEM}_chunks"
mkdir -p "$CHUNK_DIR"

MAIN_CKPT="$OUTPUT_DIR/${OUTPUT_STEM}_checkpoint.json"

"$PYTHON" -c "
import json, sys
from pathlib import Path

classes = '''$ALL_CLASSES'''.strip().split('\n')
n_workers = $N_WORKERS
main_ckpt_path = Path(r'$MAIN_CKPT')
output_dir = Path(r'$OUTPUT_DIR')
stem = '$OUTPUT_STEM'
chunk_dir = Path(r'$CHUNK_DIR')

# Load existing checkpoint (has split + already-done train_results)
main_ckpt = {}
if main_ckpt_path.exists():
    with open(main_ckpt_path) as f:
        main_ckpt = json.load(f)

existing_train = main_ckpt.get('train_results', {})
split = main_ckpt.get('split', {})
n_done = len(existing_train)

# Filter to only remaining classes
remaining = [c for c in classes if c not in existing_train]
print(f'  Already done: {n_done}, remaining: {len(remaining)}')

# Round-robin distribution of REMAINING classes only
chunks = [[] for _ in range(n_workers)]
for i, cls in enumerate(remaining):
    chunks[i % n_workers].append(cls)

for i, chunk in enumerate(chunks):
    # Write chunk class list
    chunk_path = chunk_dir / f'chunk_{i}.json'
    with open(chunk_path, 'w') as f:
        json.dump({'all_classes': chunk}, f)

    # Seed worker checkpoint with split + existing train_results
    # so that workers skip already-done classes automatically
    worker_ckpt_path = output_dir / f'{stem}_worker{i}_checkpoint.json'
    worker_ckpt = {
        'class_column': main_ckpt.get('class_column', 'mixed'),
        'split_column': main_ckpt.get('split_column'),
        'split': split,
        'train_results': {},  # only this worker's results go here
    }
    with open(worker_ckpt_path, 'w') as f:
        json.dump(worker_ckpt, f)

    print(f'  Chunk {i}: {len(chunk)} classes -> {chunk_path}')
"

# Launch workers in parallel
PIDS=()
for i in $(seq 0 $((N_WORKERS - 1))); do
    CHUNK_FILE="$CHUNK_DIR/chunk_${i}.json"
    WORKER_OUTPUT="$OUTPUT_DIR/${OUTPUT_STEM}_worker${i}.json"

    echo "  Starting worker $i (output: $WORKER_OUTPUT)..."

    "$PYTHON" "$VALIDATED_PY" \
        --database "$DATABASE" \
        --mapping "$MAPPING" \
        --output "$WORKER_OUTPUT" \
        --classes-file "$CHUNK_FILE" \
        --model "$MODEL" \
        --phase train \
        --api-key "$API_KEY" \
        "${EXTRA_ARGS[@]}" \
        > "$CHUNK_DIR/worker_${i}.log" 2>&1 &

    PIDS+=($!)
done

echo "  Waiting for ${#PIDS[@]} workers..."

# Wait for all workers and track failures
FAILURES=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "  WARNING: Worker $i (PID ${PIDS[$i]}) failed. Check $CHUNK_DIR/worker_${i}.log"
        FAILURES=$((FAILURES + 1))
    else
        echo "  Worker $i done."
    fi
done

if [[ $FAILURES -gt 0 ]]; then
    echo "  $FAILURES worker(s) failed. Continuing with available results..."
fi

# ── Step 4: Merge worker results into main checkpoint ────────────────────────

echo ""
echo "=== Step 4: Merging worker results ==="

"$PYTHON" -c "
import json, sys
from pathlib import Path

output = '$OUTPUT'
output_dir = Path('$OUTPUT_DIR')
stem = '$OUTPUT_STEM'
n_workers = $N_WORKERS

# Load main checkpoint (has the split from Phase 1 + existing train_results)
ckpt_path = Path(output).with_name(Path(output).stem + '_checkpoint.json')
with open(ckpt_path) as f:
    main_ckpt = json.load(f)

# Start from existing train_results (the 329 already-done classes)
merged_train = main_ckpt.get('train_results', {})
n_before = len(merged_train)
print(f'  Existing train_results: {n_before} classes')

for i in range(n_workers):
    worker_ckpt_path = output_dir / f'{stem}_worker{i}_checkpoint.json'
    if not worker_ckpt_path.exists():
        print(f'  Worker {i}: no checkpoint found, skipping')
        continue
    with open(worker_ckpt_path) as f:
        worker_ckpt = json.load(f)
    worker_train = worker_ckpt.get('train_results', {})
    merged_train.update(worker_train)
    print(f'  Worker {i}: {len(worker_train)} classes')

main_ckpt['train_results'] = merged_train
n_after = len(merged_train)

# Also merge token usage
total_calls = 0
total_prompt = 0
total_completion = 0
for i in range(n_workers):
    worker_ckpt_path = output_dir / f'{stem}_worker{i}_checkpoint.json'
    if not worker_ckpt_path.exists():
        continue
    with open(worker_ckpt_path) as f:
        wc = json.load(f)
    usage = wc.get('token_usage_phase2', {})
    total_calls += usage.get('total_calls', 0)
    total_prompt += usage.get('prompt_tokens', 0)
    total_completion += usage.get('completion_tokens', 0)
main_ckpt['token_usage_phase2'] = {
    'total_calls': total_calls,
    'prompt_tokens': total_prompt,
    'completion_tokens': total_completion,
    'total_tokens': total_prompt + total_completion,
}

# Atomic write
import tempfile, os
tmp_fd, tmp_path = tempfile.mkstemp(dir=ckpt_path.parent, suffix='.tmp')
with os.fdopen(tmp_fd, 'w') as f:
    json.dump(main_ckpt, f, indent=2, ensure_ascii=False)
os.replace(tmp_path, ckpt_path)

print(f'  Merged: {n_before} -> {n_after} classes in {ckpt_path}')
print(f'  Token usage: {total_calls} calls, {total_prompt + total_completion:,} total tokens')
"

# ── Step 5: Run Phases 3-5 sequentially ──────────────────────────────────────

echo ""
echo "=== Step 5: Phase 3 (FP testing) ==="

"$PYTHON" "$VALIDATED_PY" \
    --database "$DATABASE" \
    --mapping "$MAPPING" \
    --output "$OUTPUT" \
    $CLASS_SOURCE \
    --model "$MODEL" \
    --phase fp_test \
    --fp-n-jobs "$FP_N_JOBS" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Step 6: Phase 4 (fine-tuning) ==="

# Fine-tuning also calls Gemini — could split this too in a future version,
# but it's typically much faster than Phase 2 (fewer classes have FPs).
"$PYTHON" "$VALIDATED_PY" \
    --database "$DATABASE" \
    --mapping "$MAPPING" \
    --output "$OUTPUT" \
    $CLASS_SOURCE \
    --model "$MODEL" \
    --phase finetune \
    --api-key "$API_KEY" \
    --fp-n-jobs "$FP_N_JOBS" \
    --fp-retest \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Step 7: Phase 5 (evaluation) ==="

"$PYTHON" "$VALIDATED_PY" \
    --database "$DATABASE" \
    --mapping "$MAPPING" \
    --output "$OUTPUT" \
    $CLASS_SOURCE \
    --model "$MODEL" \
    --phase eval \
    --fp-n-jobs "$FP_N_JOBS" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Done! ==="
echo "  Output: $OUTPUT"
echo "  Metrics: ${OUTPUT_DIR}/${OUTPUT_STEM}_metrics.csv"
echo "  SMIRKS DB: ${OUTPUT_DIR}/${OUTPUT_STEM}_smirks_db.json"
echo "  Worker logs: $CHUNK_DIR/worker_*.log"
