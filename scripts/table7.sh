#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Work in miniBERT directory
MINIBERT_DIR="$REPO_ROOT/src/LLM/miniBERT"
LINEAR_DIR="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_circuit/circuit/linear"
TIMES_FILE="$LINEAR_DIR/proof_times.txt" 
NON_LINEAR_EXAMPLE_DIR="$REPO_ROOT/src/proof_generation/ZIP_proof_generation/ZIP_lookup/examples"

rm -f "$TIMES_FILE"
pushd "$MINIBERT_DIR" >/dev/null

echo
echo "==============================================================="
echo " Performance of ZIP on mini-BERT using the SST-2 Dataset"
echo "==============================================================="
echo
echo "⚠️  WARNING:"
echo "  This script first proves the non-linear functions in mini-BERT"
echo "  and then proves the linear operations."
echo
echo "  We constructed the non-linear circuits in this paper."
echo "  For proving linear operations, we directly use the zk-Location"
echo "  paper implementation — proving linear ops is NOT our contribution."
echo
echo "  Hardware used: 48 CPU cores (Intel® Xeon® Platinum 8360Y, 2.40 GHz)"
echo "  and 512 GB RAM. The underlying PLONK prover uses all available"
echo "  CPU cores by default, so runtimes vary with core count."
echo "==============================================================="
echo

# Interactive choice
echo "Proving both linear and non-linear operations takes ~40 hours"
echo "and requires ~200 GB of RAM."
echo
echo "Only proving non-linear operations (our contribution) takes ~3 hours."
echo
read -p "Enter 0 to reproduce only non-linear circuits (~3h), or 1 to run both (~40h): " choice
echo

python minibert.py
python infer_minibert.py

python embedding_layer.py
python encoder_block_0.py
python encoder_block_1.py
python encoder_block_2.py
python encoder_block_3.py
python classifier_output.py

popd >/dev/null

cd "$LINEAR_DIR/mini_bert_output_add_mult/non-linear"

# Concatenate logs
cat gelu_*.txt > all_gelu.txt
cat exp_*.txt > all_exp.txt
cat inv_sqrt_*.txt > all_inv_sqrt.txt

rm -f gelu_*.txt exp_*.txt inv_sqrt_*.txt

split -l 4096 -d -a 2 --additional-suffix=.txt all_gelu.txt gelu_part_

for i in $(seq -w 0 11); do
  n=$((10#$i + 1))
  dst="$NON_LINEAR_EXAMPLE_DIR/y_yprime_examples_minibert_act$n"
  mkdir -p "$dst"
  mv "gelu_part_${i}.txt" "$dst/gelu_y_yprime.txt"
done

mkdir -p "$NON_LINEAR_EXAMPLE_DIR/y_yprime_examples_minibert_act13"
mv "all_exp.txt" "$NON_LINEAR_EXAMPLE_DIR/y_yprime_examples_minibert_act13/softmax_y_yprime.txt"

mkdir -p "$NON_LINEAR_EXAMPLE_DIR/y_yprime_examples_minibert_act14"
mv "all_inv_sqrt.txt" "$NON_LINEAR_EXAMPLE_DIR/y_yprime_examples_minibert_act14/layernorm_y_yprime.txt"

cd "$LINEAR_DIR/mini_bert_output_add_mult"
rm -rf non-linear

echo "====================================================================="
echo "= Proving non-linear layers (activations): GeLU, Softmax, LayerNorm ="
echo "====================================================================="

# Prove GELU (12 parts)
for i in $(seq 1 12); do
  echo "Proving $i/12 gelu..."
  "$SCRIPT_DIR/table4.sh" 1 "y_yprime_examples_minibert_act${i}" gelu
done

echo "Proving softmax..."
"$SCRIPT_DIR/table4.sh" 1 y_yprime_examples_minibert_act13 softmax

echo "Proving layernorm..."
"$SCRIPT_DIR/table4.sh" 1 y_yprime_examples_minibert_act14 layernorm


# Proving linear layers only if user chose 1
if [[ "$choice" == "1" ]]; then
  echo "==========================="
  echo "== Proving linear layers =="
  echo "==========================="
  echo

  cd "$LINEAR_DIR"

  for i in {1..777}; do
    python generate_config.py --num-add 60000 --num-mul 60000 --size 18
    go run main.go config.go
  done

  python generate_config.py --num-add 60000 --num-mul 0 --size 18
  go run main.go config.go

  python generate_config.py --num-add 60000 --num-mul 0 --size 18
  go run main.go config.go
  
  python generate_config.py --num-add 60000 --num-mul 0 --size 18
  go run main.go config.go

  python generate_config.py --num-add 20206 --num-mul 31352 --size 18
  go run main.go config.go

  echo
  echo "*********************************************"
  # ---- Totals from proof_times.txt ----
  if [[ -s "$TIMES_FILE" ]]; then
    awk -F',' '
      { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); p += $1 + 0; v += $2 + 0 }
      END {
        printf("total proving time for all linear layers : %.6f sec\n", p + 0);
        printf("total verification time for all linear layers: %.6f sec\n", v + 0);
      }' "$TIMES_FILE"
  else
    echo "No proof times found at $TIMES_FILE"
  fi
  echo "*********************************************"
  echo
else
  echo "Skipping linear layers (only reproducing non-linear circuits)."
  echo
fi
