import os
import csv
import json
import math
import numpy as np
from numbers import Number
import struct

PARAM_JSON = "bert_mini_params.json"
ACT_JSON   = "bert_mini_inference_layer_inputs.json"
LN_EPS     = 1e-12

BASE_DIR   =  "../../proof_generation/ZIP_proof_generation/ZIP_circuit/circuit/linear/mini_bert_output_add_mult"
DIR_ADD    = os.path.join(BASE_DIR, "addition")
DIR_MUL    = os.path.join(BASE_DIR, "multiplication")
DIR_DIV    = os.path.join(BASE_DIR, "division")
DIR_NONLIN = os.path.join(BASE_DIR, "non-linear")

print("Running embedding_layer...")

for d in [DIR_ADD, DIR_MUL, DIR_DIV, DIR_NONLIN]:
    os.makedirs(d, exist_ok=True)

with open(PARAM_JSON, "r") as f:
    params = json.load(f)
with open(ACT_JSON, "r") as f:
    acts = json.load(f)

# Ids
word_ids = np.array(acts["bert.embeddings.word_embeddings"],       dtype=int)  # (B,S)
pos_ids  = np.array(acts["bert.embeddings.position_embeddings"],   dtype=int)  # (B,S)
tok_ids  = np.array(acts["bert.embeddings.token_type_embeddings"], dtype=int)  # (B,S)
B, S = word_ids.shape

# Weights
W_word = np.array(params["bert.embeddings.word_embeddings.weight"],     dtype=float)  # (Vw,d)
W_pos  = np.array(params["bert.embeddings.position_embeddings.weight"], dtype=float)  # (Vp,d)
W_tok  = np.array(params["bert.embeddings.token_type_embeddings.weight"],dtype=float) # (Vs,d)
ln_w   = np.array(params["bert.embeddings.LayerNorm.weight"],           dtype=float)
ln_b   = np.array(params["bert.embeddings.LayerNorm.bias"],             dtype=float)

Vw, d  = W_word.shape
Vp, _  = W_pos.shape
Vs, _  = W_tok.shape

mul_idx = add_idx = sub_idx = div_idx = 0
total_add = total_mul = total_sub = total_div = 0
nonlin_files = []
bit_decomp_calls = 0

def f64_hex(x: float) -> str:
    return "0x" + format(struct.unpack(">Q", struct.pack(">d", float(x)))[0], "016x")

def next_mul():
    global mul_idx
    p = os.path.join(DIR_MUL, f"embedding_layer_mul_{mul_idx}.txt")
    mul_idx += 1
    return p

def next_add():
    global add_idx
    p = os.path.join(DIR_ADD, f"embedding_layer_add_{add_idx}.txt")
    add_idx += 1
    return p

def next_div():
    global div_idx
    p = os.path.join(DIR_DIV, f"embedding_layer_div_{div_idx}.txt")
    div_idx += 1
    return p


def int_to_bits(n, K):
    global bit_decomp_calls
    bit_decomp_calls += 1
    return [(n >> i) & 1 for i in range(K)]

def pad_to_power_of_two(W):
    V, d_local = W.shape
    K = math.ceil(math.log2(V)) if V > 1 else 1
    P = 1 << K
    if P == V:
        return W, K
    pad = np.zeros((P - V, d_local), dtype=W.dtype)
    return np.vstack([W, pad]), K

def binary_mux_select_logged(id_scalar, W_const,
                             level_mul_w, level_mul_b,
                             level_add,
                             b_idx, s_idx, tag):
    """
    Logged mux for one (b,s). Returns selected row (d,)
    nodes = (1 - b) * left + b * right
    """
    row_count, d_local = W_const.shape
    K = int(math.log2(row_count))
    bits = int_to_bits(int(id_scalar), K)

    nodes = W_const.copy()  # (2^K, d)

    for lvl, bit in enumerate(bits):
        left  = nodes[0::2]
        right = nodes[1::2]
        n_pairs = left.shape[0]
        b_val = float(bit)
        one_minus_b = 1.0 - b_val

        # log operations
        for p_idx in range(n_pairs):
            for f_idx in range(d_local):
                # (1 - b) logged as addition triplet: 1 + (-b) = 1-b
                sub_res = one_minus_b
                neg_b   = -b_val
                level_add[lvl].writerow([f"{f64_hex(1.0)}", f" {f64_hex(neg_b)}", f" {f64_hex(sub_res)}"])

                # multiplications
                prod_left  = one_minus_b * left[p_idx, f_idx]
                prod_right = b_val * right[p_idx, f_idx]

                # addition of the two products
                newv = prod_left + prod_right

                level_mul_w[lvl].writerow([f"{f64_hex(one_minus_b)}", f" {f64_hex(left[p_idx, f_idx])}",  f" {f64_hex(prod_left)}"])
                level_mul_b[lvl].writerow([f"{f64_hex(b_val)}",       f" {f64_hex(right[p_idx, f_idx])}", f" {f64_hex(prod_right)}"])
                level_add[lvl].writerow(  [f"{f64_hex(prod_left)}",   f" {f64_hex(prod_right)}",         f" {f64_hex(newv)}"])

        nodes = one_minus_b * left + b_val * right

    return nodes[0]

def select_rows_logged(ids, W, tag):
    global total_add, total_mul, total_sub

    W_pad, K = pad_to_power_of_two(W)
    out = np.zeros((ids.shape[0], ids.shape[1], W.shape[1]), dtype=W.dtype)
    d_local = W.shape[1]

    mul_w_path = next_mul()   # (1-b)*left
    mul_b_path = next_mul()   # b*right
    add_path  = next_add()

    with open(mul_w_path, "w", newline="") as mwf, \
         open(mul_b_path, "w", newline="") as mbf, \
         open(add_path,  "w", newline="") as af:

        mul_w = csv.writer(mwf)
        mul_b = csv.writer(mbf)
        add_w = csv.writer(af)

        level_mul_w = {lvl: mul_w for lvl in range(K)}
        level_mul_b = {lvl: mul_b for lvl in range(K)}
        level_add   = {lvl: add_w for lvl in range(K)}

        for b_idx in range(ids.shape[0]):
            for s_idx in range(ids.shape[1]):
                row = binary_mux_select_logged(
                    ids[b_idx, s_idx], W_pad,
                    level_mul_w, level_mul_b,
                    level_add,
                    b_idx, s_idx, tag
                )
                out[b_idx, s_idx] = row

    P = W_pad.shape[0]               # 2^K
    pairs_total = P - 1              # internal nodes
    per_feat_ops = pairs_total * d_local
    total_mul += per_feat_ops     # muls
    total_add += per_feat_ops     # for (1 + -b)

    return out

out_word = select_rows_logged(word_ids, W_word, "word")
out_pos  = select_rows_logged(pos_ids,  W_pos,  "pos")
out_tok  = select_rows_logged(tok_ids,  W_tok,  "tok")

# ───────── emb_sum = out_word + out_pos + out_tok ─────────
sum1 = np.zeros_like(out_word, dtype=np.float64)
emb_sum = np.zeros_like(out_word, dtype=np.float64)

add_path = next_add()
with open(add_path, "w", newline="") as af:
    aw = csv.writer(af)
    add_c = 0
    for b in range(B):
        for s in range(S):
            for f in range(d):
                a = out_word[b, s, f]; b_ = out_pos[b, s, f]
                sm = a + b_
                sum1[b, s, f] = sm
                aw.writerow([f"{f64_hex(a)}", f" {f64_hex(b_)}", f" {f64_hex(sm)}"])
                add_c += 1
total_add += add_c

add_path = next_add()
with open(add_path, "w", newline="") as af:
    aw = csv.writer(af)
    add_c2 = 0
    for b in range(B):
        for s in range(S):
            for f in range(d):
                a = sum1[b, s, f]; b_ = out_tok[b, s, f]
                sm = a + b_
                emb_sum[b, s, f] = sm
                aw.writerow([f"{f64_hex(a)}", f" {f64_hex(b_)}", f" {f64_hex(sm)}"])
                add_c2 += 1
total_add += add_c2

# ───────── LayerNorm ─────────
# mean
add_path = next_add()
mul_path = next_mul()
m1 = np.zeros((B, S, 1), dtype=np.float64)

add_c = mul_c = 0
with open(add_path, "w", newline="") as af, open(mul_path, "w", newline="") as mf:
    aw = csv.writer(af); mw = csv.writer(mf)
    inv_d = 1.0 / d
    for b in range(B):
        for s in range(S):
            acc = 0.0
            for f in range(d):
                prev = acc
                addend = emb_sum[b, s, f]
                acc += addend
                aw.writerow([f"{f64_hex(prev)}", f" {f64_hex(addend)}",  f" {f64_hex(acc)}"])
                add_c += 1
            mean_val = acc * inv_d
            mw.writerow([f"{f64_hex(acc)}",  f" {f64_hex(inv_d)}",   f" {f64_hex(mean_val)}"])
            mul_c += 1
            m1[b, s, 0] = mean_val
total_add += add_c; total_mul += mul_c

# center as addition with negated mean
add_path = next_add()
y1 = np.zeros_like(emb_sum, dtype=np.float64)
add_c_center = 0
with open(add_path, "w", newline="") as af:
    aw = csv.writer(af)
    for b in range(B):
        for s in range(S):
            m_val = m1[b, s, 0]
            neg_m = -m_val
            for f in range(d):
                v = emb_sum[b, s, f]
                diff = v + neg_m
                y1[b, s, f] = diff
                aw.writerow([f"{f64_hex(v)}", f" {f64_hex(neg_m)}", f" {f64_hex(diff)}"])
                add_c_center += 1
total_add += add_c_center

# square
mul_path = next_mul()
y2 = np.zeros_like(y1, dtype=np.float64)
mul_c = 0
with open(mul_path, "w", newline="") as mf:
    mw = csv.writer(mf)
    for b in range(B):
        for s in range(S):
            for f in range(d):
                v = y1[b, s, f]
                prod = v * v
                y2[b, s, f] = prod
                mw.writerow([f"{f64_hex(v)}", f" {f64_hex(v)}", f" {f64_hex(prod)}"])
                mul_c += 1
total_mul += mul_c

# variance
add_path = next_add()
mul_path = next_mul()
y3 = np.zeros((B, S, 1), dtype=np.float64)

add_c = mul_c = 0
with open(add_path, "w", newline="") as af, open(mul_path, "w", newline="") as mf:
    aw = csv.writer(af); mw = csv.writer(mf)
    inv_d = 1.0 / d
    for b in range(B):
        for s in range(S):
            acc = 0.0
            for f in range(d):
                prev = acc
                addend = y2[b, s, f]
                acc += addend
                aw.writerow([f"{f64_hex(prev)}", f" {f64_hex(addend)}", f" {f64_hex(acc)}"])
                add_c += 1
            var_val = acc * inv_d
            mw.writerow([f"{f64_hex(acc)}",  f" {f64_hex(inv_d)}",  f" {f64_hex(var_val)}"])
            mul_c += 1
            y3[b, s, 0] = var_val
total_add += add_c; total_mul += mul_c

# + eps
add_path = next_add()
y4 = np.zeros_like(y3, dtype=np.float64)
add_c = 0
with open(add_path, "w", newline="") as af:
    aw = csv.writer(af)
    for b in range(B):
        for s in range(S):
            prev = y3[b, s, 0]
            newv = prev + LN_EPS
            y4[b, s, 0] = newv
            aw.writerow([f"{f64_hex(prev)}", f" {f64_hex(LN_EPS)}", f" {f64_hex(newv)}"])
            add_c += 1
total_add += add_c

# Save y4 (variance+eps)
inv1_path = os.path.join(DIR_NONLIN, "inv_sqrt_1_embed.txt")
bound_lo, bound_hi = 0.001, 210.0
y4_clamped = np.clip(y4, bound_lo, bound_hi)
with open(inv1_path, "w") as nf:
    for b in range(B):
        for s in range(S):
            inp = float(y4_clamped[b, s, 0])
            out = float(1.0 / np.sqrt(inp))
            nf.write(f"{f64_hex(out)}, {f64_hex(inp)}\n")
nonlin_files.append(inv1_path)

# inv_sqrt2 = 1 / sqrt(y4)
inv_sqrt2 = 1.0 / np.sqrt(y4_clamped)

# normalize
mul_path = next_mul()
y5 = np.zeros_like(y1, dtype=np.float64)
mul_c = 0
with open(mul_path, "w", newline="") as mf:
    mw = csv.writer(mf)
    for b in range(B):
        for s in range(S):
            scale = inv_sqrt2[b, s, 0]
            for f in range(d):
                v = y1[b, s, f]
                prod = v * scale
                y5[b, s, f] = prod
                mw.writerow([f"{f64_hex(v)}", f" {f64_hex(scale)}", f" {f64_hex(prod)}"])
                mul_c += 1
total_mul += mul_c

# affine
mul_path = next_mul()
add_path = next_add()
emb_ln = np.zeros_like(y5, dtype=np.float64)

mul_c = add_c = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    for b in range(B):
        for s in range(S):
            for f in range(d):
                prod = y5[b, s, f] * float(ln_w[f])
                mw.writerow([f"{f64_hex(y5[b, s, f])}", f" {f64_hex(float(ln_w[f]))}", f" {f64_hex(prod)}"])
                mul_c += 1

                prev = prod
                newv = prev + float(ln_b[f])
                aw.writerow([f"{f64_hex(prev)}",        f" {f64_hex(float(ln_b[f]))}", f" {f64_hex(newv)}"])
                add_c += 1

                emb_ln[b, s, f] = newv

# Final output
print("\nEmbedding output (post-LN) shape:", emb_ln.shape)
print(emb_ln)

total_mul += mul_c; total_add += add_c

# ───────── Totals ─────────
def sum_scalar_counts(prefix):
    return sum(int(v) for k, v in globals().items()
               if k.startswith(prefix) and isinstance(v, Number))

add_total = total_add + sum_scalar_counts("add_count")
mul_total = total_mul + sum_scalar_counts("mul_count")

inv_sqrt_calls = y4.size  # 1/sqrt(var+eps) per (B,S)

print("\n===== TOTAL OPS (Embedding Layer) =====")
print(f"Additions:        {add_total}")
print(f"Multiplications:  {mul_total}")
print("Non-linear:")
print(f"  inv_sqrt calls:       {inv_sqrt_calls}")
print(f"  bit_decomp calls:     {bit_decomp_calls}")
