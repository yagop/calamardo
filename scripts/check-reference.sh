#!/bin/sh
# Milestone 1 check: greedy decoding with a bf16 cache must produce exactly the
# reference's tokens. The reference is mlx-lm from PR #2086 (Inkling 2 support,
# pinned commit) on MLX 0.32.4, installed in third_party/refvenv.
#
# Usage: scripts/check-reference.sh PROMPT_FILE [MAX_TOKENS]
# PROMPT_FILE holds the full prompt text, chat format included, or token ids
# separated by spaces if its name ends in .ids.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROMPT=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
MAX=${2:-256}
VENV=$ROOT/third_party/refvenv
REPO=${CALAMARDO_REPO:-thinkingmachines/Inkling-2-Mini}
CACHE=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
SNAPSHOT=$CACHE/models--$(echo "$REPO" | sed 's|/|--|')/snapshots/$(cat "$CACHE/models--$(echo "$REPO" | sed 's|/|--|')/refs/main")
MLX_LM=git+https://github.com/jundot/mlx-lm@0c4c99192230f1943dd468d297f0b8a64ae7b7c2

if [ ! -x "$VENV/bin/python" ]; then
    PY=${PYTHON:-python3.12}
    "$PY" -m venv "$VENV"
    "$VENV/bin/python" -I -m pip install -q --disable-pip-version-check "mlx==0.32.4" "$MLX_LM"
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "reference..."
(cd "$TMP" && "$VENV/bin/python" -I "$ROOT/scripts/reference/ref_greedy.py" "$SNAPSHOT" "$PROMPT" "$MAX" > ref.json)
echo "calamardo..."
cargo build -q --release --manifest-path "$ROOT/Cargo.toml" -p calamardo
case "$PROMPT" in
    *.ids) INPUT="--tokens=$(tr -s ' \n' ',' < "$PROMPT" | sed 's/^,//; s/,$//')" ;;
    *) INPUT="--raw=$(cat "$PROMPT")" ;;
esac
"$ROOT/target/release/calamardo" generate "$INPUT" --ids --max-tokens "$MAX" > "$TMP/ours.txt"
python3 -I "$ROOT/scripts/reference/compare.py" "$TMP/ref.json" "$TMP/ours.txt"
