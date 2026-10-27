#!/usr/bin/env bash
# Push a Kaggle kernel from one of our kaggle/<name>/ dirs.
# Usage: ./scripts/push_kernel.sh zero-shot-baseline
#
# Prereqs: kaggle CLI authed; kernel-metadata.json present in the target dir.

set -euo pipefail

KERNEL_NAME="${1:-}"
if [[ -z "$KERNEL_NAME" ]]; then
    echo "Usage: $0 <kernel-dir-name>" >&2
    echo "  e.g. $0 zero-shot-baseline" >&2
    echo "  available: $(ls kaggle/ 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
fi

KERNEL_DIR="kaggle/$KERNEL_NAME"
if [[ ! -d "$KERNEL_DIR" ]]; then
    echo "Error: $KERNEL_DIR does not exist" >&2
    exit 1
fi

if [[ ! -f "$KERNEL_DIR/kernel-metadata.json" ]]; then
    echo "Error: $KERNEL_DIR/kernel-metadata.json missing" >&2
    exit 1
fi

# Sync the latest notebook from notebooks/ into the push dir.
# Convention: kaggle/<name>/01-<name>.ipynb mirrors notebooks/baseline/01-<name>.ipynb
#             or notebooks/experiments/<name>.ipynb depending on stage.
SOURCE_IPYNB=$(grep -E '"code_file"' "$KERNEL_DIR/kernel-metadata.json" | sed -E 's/.*"code_file" *: *"([^"]+)".*/\1/')
SOURCE_NAME=$(basename "$SOURCE_IPYNB")

# Try to find the canonical source notebook
for candidate in \
    "notebooks/baseline/$SOURCE_NAME" \
    "notebooks/eval/$SOURCE_NAME" \
    "notebooks/experiments/$SOURCE_NAME"; do
    if [[ -f "$candidate" ]]; then
        echo "Syncing: $candidate -> $KERNEL_DIR/$SOURCE_NAME"
        cp "$candidate" "$KERNEL_DIR/$SOURCE_NAME"
        break
    fi
done

echo "Pushing $KERNEL_DIR..."
uv run kaggle kernels push -p "$KERNEL_DIR"
