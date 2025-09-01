import os
import json
import numpy as np
import csv
import struct

print("Running classifier...")

def f64_hex(x: float) -> str:
    return "0x" + format(struct.unpack(">Q", struct.pack(">d", float(x)))[0], "016x")

BASE_DIR   =  "../../proof_generation/ZIP_proof_generation/ZIP_circuit/circuit/linear/mini_bert_output_add_mult"
mul_dir  = os.path.join(BASE_DIR, "multiplication")
add_dir  = os.path.join(BASE_DIR, "addition")
for d in (mul_dir, add_dir):
    os.makedirs(d, exist_ok=True)

with open("bert_mini_inference_layer_inputs.json", "r") as f:
    acts = json.load(f)
with open("bert_mini_params.json", "r") as f:
    params = json.load(f)

X = np.array(acts['classifier'], dtype=np.float64)
W = np.array(params['classifier.weight'], dtype=np.float64)  # [2, 256]
b = np.array(params['classifier.bias'],   dtype=np.float64)  # [2]

batch, in_f = X.shape
out_f, _    = W.shape

mul_count = add_count = 0
with open(os.path.join(mul_dir, "classifier_mul.txt"), "w", newline="") as mf, \
     open(os.path.join(add_dir, "classifier_add.txt"), "w", newline="") as af:

    mw = csv.writer(mf)
    aw = csv.writer(af)

    for n in range(batch):
        for of in range(out_f):
            acc = 0.0
            for inf in range(in_f):
                wv   = W[of, inf]
                xv   = X[n, inf]
                prod = wv * xv
                mw.writerow([f"{f64_hex(wv)}", f" {f64_hex(xv)}", f" {f64_hex(prod)}"])
                mul_count += 1

                prev = acc
                acc += prod
                aw.writerow([f"{f64_hex(prev)}", f" {f64_hex(prod)}", f" {f64_hex(acc)}"])
                add_count += 1

            # bias add
            prev = acc
            acc += b[of]
            aw.writerow([f"{f64_hex(prev)}", f" {f64_hex(float(b[of]))}", f" {f64_hex(acc)}"])
            add_count += 1

logits = X @ W.T + b

print("Classifier logits shape:", logits.shape)
print("Classifier logits:", logits)

print("\n===== TOTAL OPS (Classifier) =====")
print(f"Additions:        {add_count}")
print(f"Multiplications:  {mul_count}")