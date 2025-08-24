#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

LLM_DIR="../src/LLM"
cd "$LLM_DIR"

exec python3 miniBERT_accuracy.py
