import os
import csv
import json
import numpy as np
from transformers import AutoTokenizer
import struct

print("Running encoder_block_0...")

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def f64_hex(x: float) -> str:
    return "0x" + format(struct.unpack(">Q", struct.pack(">d", float(x)))[0], "016x")

def log_mul(writer, a, b):
    p = a * b
    writer.writerow([f"{f64_hex(a)}", f" {f64_hex(b)}", f" {f64_hex(p)}"])
    return p

def log_add(writer, a, b):
    s = a + b
    writer.writerow([f"{f64_hex(a)}", f" {f64_hex(b)}", f" {f64_hex(s)}"])
    return s

def log_sub_as_add(writer, a, b):
    nb = -b                      
    s  = a + nb                 
    writer.writerow([f"{f64_hex(a)}", f" {f64_hex(nb)}", f" {f64_hex(s)}"])
    return s

def log_div(writer, numerator, denominator):
    q = numerator / denominator
    writer.writerow([f"{f64_hex(numerator)}", f" {f64_hex(denominator)}", f" {f64_hex(q)}"])
    return q

PARAM_JSON = "bert_mini_params.json"
ACT_JSON   = "bert_mini_inference_layer_inputs.json"
MODEL_DIR  = "bert_mini_sst2_finetuned"

BASE_DIR   =  "../../proof_generation/ZIP_proof_generation/ZIP_circuit/circuit/linear/mini_bert_output_add_mult"
DIR_ADD    = os.path.join(BASE_DIR, "addition")
DIR_MUL    = os.path.join(BASE_DIR, "multiplication")
DIR_DIV    = os.path.join(BASE_DIR, "division")
DIR_NONLIN = os.path.join(BASE_DIR, "non-linear")

for d in [DIR_ADD, DIR_MUL, DIR_DIV, DIR_NONLIN]:
    os.makedirs(d, exist_ok=True)

with open(PARAM_JSON, "r") as f:
    params = json.load(f)
with open(ACT_JSON, "r") as f:
    acts = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
enc = tokenizer(
    "The movie was delightful and I enjoyed every moment.",
    truncation=True,
    padding="max_length",
    max_length=12,
    return_tensors="np"
)
attn_mask = enc["attention_mask"]         # shape: (1, 128)
B, S      = attn_mask.shape
mask_4d   = attn_mask.reshape(B, 1, 1, S) # (B, 1, 1, S)
M         = (1 - mask_4d) * (-1e2)        # broadcast mask


# 1) Embedding LayerNorm (reconstruct post-LN embeddings)
inp_pre = np.array(acts["bert.embeddings.LayerNorm"], dtype=float)  # [1, S, H]
ln_w    = np.array(params["bert.embeddings.LayerNorm.weight"], dtype=float)
ln_b    = np.array(params["bert.embeddings.LayerNorm.bias"],   dtype=float)

mean = inp_pre.mean(axis=-1, keepdims=True)
var  = inp_pre.var(axis=-1, keepdims=True)
x    = (inp_pre - mean) / np.sqrt(var + 1e-12) * ln_w + ln_b  # post-LN embeddings

# 2) Q/K/V projections (matmul + bias add + logging)
Wq = np.array(params["bert.encoder.layer.0.attention.self.query.weight"], dtype=float)
Bq = np.array(params["bert.encoder.layer.0.attention.self.query.bias"],   dtype=float)
Wk = np.array(params["bert.encoder.layer.0.attention.self.key.weight"],   dtype=float)
Bk = np.array(params["bert.encoder.layer.0.attention.self.key.bias"],     dtype=float)
Wv = np.array(params["bert.encoder.layer.0.attention.self.value.weight"], dtype=float)
Bv = np.array(params["bert.encoder.layer.0.attention.self.value.bias"],   dtype=float)

Bsz, seq_len, hidden = x.shape

mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_0.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_0.txt")

def project_with_logging(X, W, b, tag, mw, aw):
    Y = np.zeros_like(X)
    mul_c = add_c = 0
    for s in range(seq_len):
        for out_f in range(hidden):
            acc = 0.0
            for in_f in range(hidden):
                wv = W[out_f, in_f]
                xv = X[0, s, in_f]
                prod = log_mul(mw, wv, xv)
                mul_c += 1

                prev = acc
                acc  = log_add(aw, prev, prod)
                add_c += 1

            # bias add
            prev = acc
            acc  = log_add(aw, prev, b[out_f])
            add_c += 1

            Y[0, s, out_f] = acc
    return Y, mul_c, add_c

with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    q, m_q, a_q = project_with_logging(x, Wq, Bq, "q", mw, aw)
    k, m_k, a_k = project_with_logging(x, Wk, Bk, "k", mw, aw)
    v, m_v, a_v = project_with_logging(x, Wv, Bv, "v", mw, aw)

total_mul = m_q + m_k + m_v
total_add = a_q + a_k + a_v

# 3) Reshape & transpose for multi-head attention
B, S, H = q.shape
Nh      = 4
Hd      = H // Nh

qh = q.reshape(B, S, Nh, Hd).transpose(0,2,1,3)  # (B, Nh, S, Hd)
kh = k.reshape(B, S, Nh, Hd).transpose(0,2,1,3)
vh = v.reshape(B, S, Nh, Hd).transpose(0,2,1,3)

# 4) Scaled Dot-Product Attention (matmuls, mask, softmax, ctx)
# 4.1) qh @ kh^T
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_1.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_1.txt")

mul_count1 = add_count1 = 0
manual_temp = np.zeros((B, Nh, S, S), dtype=np.float64)

with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf)
    aw = csv.writer(af)
    for b in range(B):
        for h in range(Nh):
            for i in range(S):
                for j in range(S):
                    acc = 0.0
                    for d in range(Hd):
                        qv = qh[b,h,i,d]
                        kv = kh[b,h,j,d]
                        prod = log_mul(mw, qv, kv)
                        mul_count1 += 1

                        prev = acc
                        acc  = log_add(aw, prev, prod)
                        add_count1 += 1
                    manual_temp[b,h,i,j] = acc

# 4.2) scale by 1/sqrt(Hd)
with open(os.path.join(DIR_NONLIN, "inv_sqrt_0_block0.txt"), "w") as nf:
    bound_lo, bound_hi = 0.001, 210.0
    Hd_clamped = float(np.clip(Hd, bound_lo, bound_hi))
    out0 = float(1.0 / np.sqrt(Hd_clamped))
    nf.write(f"{f64_hex(out0)}, {f64_hex(Hd_clamped)}\n")
inv_sqrt = out0

mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_2.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_2.txt")
mul_count2 = add_count2 = 0
scores = np.zeros_like(manual_temp, dtype=np.float64)

with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    for b in range(B):
        for h in range(Nh):
            for i in range(S):
                for j in range(S):
                    val = manual_temp[b, h, i, j]
                    product = log_mul(mw, val, inv_sqrt)
                    mul_count2 += 1
                    scores[b,h,i,j] = product

# 4.3) Apply mask: scores += M
mask_add_path = os.path.join(DIR_ADD, "encoder_block_0_add_3.txt")
mask_mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_3.txt")

mul_count3 = add_count3 = 0
with open(mask_mul_path, "w", newline="") as mf, open(mask_add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    for b in range(B):
        for h in range(Nh):
            for i in range(S):
                for j in range(S):
                    prev   = scores[b, h, i, j]
                    addend = M[b, 0, 0, j]
                    newv   = log_add(aw, prev, addend)
                    add_count3 += 1
                    scores[b, h, i, j] = newv

# 4.4) Softmax (exp and sums)
exp_path = os.path.join(DIR_NONLIN, "exp_0.txt")
exp_vals = np.zeros_like(scores, dtype=np.float64)
count_exp = 0
bound_lo, bound_hi = -8.0, 20.0
yprime_scores = np.clip(scores, bound_lo, bound_hi)
with open(exp_path, "w") as ef:
    for b in range(B):
        for h in range(Nh):
            for i in range(S):
                for j in range(S):
                    yp = yprime_scores[b, h, i, j]
                    y  = np.exp(yp)
                    exp_vals[b, h, i, j] = y
                    ef.write(f"{f64_hex(y)}, {f64_hex(yp)}\n")
                    count_exp += 1

# Sum along last axis
sum_add_path = os.path.join(DIR_ADD, "encoder_block_0_add_4.txt")
sum_mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_4.txt")
sum_exp = np.zeros((B, Nh, S, 1), dtype=np.float64)

mul_count4 = add_count4 = 0
with open(sum_mul_path, "w", newline="") as mf, open(sum_add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    for b in range(B):
        for h in range(Nh):
            for i in range(S):
                acc = 0.0
                for j in range(S):
                    prev   = acc
                    addend = exp_vals[b, h, i, j]
                    acc    = log_add(aw, prev, addend)
                    add_count4 += 1
                sum_exp[b, h, i, 0] = acc

# Reciprocal (1 / sum_exp)
div_path = os.path.join(DIR_DIV, "encoder_block_0_div_0.txt")
inv = np.zeros_like(sum_exp, dtype=np.float64)
div_count0 = 0
with open(div_path, "w", newline="") as df:
    dw = csv.writer(df)
    for b in range(B):
        for h in range(Nh):
            for i in range(S):
                denom = sum_exp[b, h, i, 0]
                res = log_div(dw, 1.0, denom)
                inv[b, h, i, 0] = res
                div_count0 += 1

# Elementwise: attn = exp_vals * inv
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_5.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_5.txt")
attn = np.zeros_like(exp_vals, dtype=np.float64)

mul_count5 = add_count5 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    for b in range(B):
        for h in range(Nh):
            for i in range(S):
                scale = inv[b, h, i, 0]
                for j in range(S):
                    e     = exp_vals[b, h, i, j]
                    prod  = log_mul(mw, e, scale)
                    mul_count5 += 1
                    attn[b, h, i, j] = prod

# Context: ctx = attn @ vh
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_6.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_6.txt")
ctx = np.zeros((B, Nh, S, Hd), dtype=np.float64)

mul_count6 = add_count6 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    for b in range(B):
        for h in range(Nh):
            for i in range(S):
                for d in range(Hd):
                    acc = 0.0
                    for j in range(S):
                        a = attn[b,h,i,j]
                        v_ = vh[b,h,j,d]
                        prod = log_mul(mw, a, v_)
                        mul_count6 += 1

                        prev = acc
                        acc  = log_add(aw, prev, prod)
                        add_count6 += 1
                    ctx[b,h,i,d] = acc

# Merge heads
ctx = ctx.transpose(0,2,1,3).reshape(B, S, H)

# 5) Attention output projection + bias
Wout = np.array(params["bert.encoder.layer.0.attention.output.dense.weight"], dtype=float)
Bout = np.array(params["bert.encoder.layer.0.attention.output.dense.bias"],   dtype=float)

mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_7.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_7.txt")
attn_out = np.zeros((B, S, H), dtype=np.float64)

mul_count7 = add_count7 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    for b in range(B):
        for s_idx in range(S):
            for out_f in range(H):
                acc = 0.0
                for in_f in range(H):
                    x_val = ctx[b, s_idx, in_f]
                    w_val = Wout[out_f, in_f]
                    prod  = log_mul(mw, x_val, w_val)
                    mul_count7 += 1

                    prev = acc
                    acc  = log_add(aw, prev, prod)
                    add_count7 += 1

                # bias
                prev = acc
                acc  = log_add(aw, prev, float(Bout[out_f]))
                add_count7 += 1

                attn_out[b, s_idx, out_f] = acc

# 6) Residual + LayerNorm1
# Residual: res1 = attn_out + x
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_8.txt")
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_8.txt")  # empty
res1 = np.zeros_like(attn_out, dtype=np.float64)

add_count8 = mul_count8 = 0
with open(add_path, "w", newline="") as af, open(mul_path, "w", newline="") as mf:
    aw = csv.writer(af); mw = csv.writer(mf)
    mw.writerow(["(no mults in res1 = attn_out + x)"])
    for b in range(B):
        for s_idx in range(S):
            for f in range(H):
                a  = attn_out[b, s_idx, f]
                xb = x[b, s_idx, f]
                s  = log_add(aw, a, xb)
                add_count8 += 1
                res1[b, s_idx, f] = s

# ---------- LN1 ----------
ln1_w = np.array(params["bert.encoder.layer.0.attention.output.LayerNorm.weight"], dtype=float)
ln1_b = np.array(params["bert.encoder.layer.0.attention.output.LayerNorm.bias"],   dtype=float)

# m1 = mean(res1)
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_9.txt")
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_9.txt")
m1 = np.zeros((B, S, 1), dtype=np.float64)

add_count9 = mul_count9 = 0
with open(add_path, "w", newline="") as af, open(mul_path, "w", newline="") as mf:
    aw = csv.writer(af); mw = csv.writer(mf)
    inv_H = 1.0 / H
    for b in range(B):
        for s_idx in range(S):
            acc = 0.0
            for f in range(H):
                prev   = acc
                addend = res1[b, s_idx, f]
                acc    = log_add(aw, prev, addend)
                add_count9 += 1
            mean_val = log_mul(mw, acc, inv_H)
            mul_count9 += 1
            m1[b, s_idx, 0] = mean_val

# y1 = res1 - m1 (center)  -> log & count as addition with negated second operand
add_path_sub0 = os.path.join(DIR_ADD, "encoder_block_0_add_9a.txt")
y1 = np.zeros_like(res1, dtype=np.float64)
add_count_sub0 = 0
with open(add_path_sub0, "w", newline="") as af:
    aw = csv.writer(af)
    for b in range(B):
        for s_idx in range(S):
            m_val = m1[b, s_idx, 0]
            for f in range(H):
                r_val = res1[b, s_idx, f]
                diff  = log_sub_as_add(aw, r_val, m_val)  # a + (-b)
                y1[b, s_idx, f] = diff
                add_count_sub0 += 1

# y2 = y1 * y1
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_10.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_10.txt")
y2 = np.zeros_like(y1, dtype=np.float64)

mul_count10 = add_count10 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    aw.writerow(["(no additions for y2 = y1 * y1)"])
    for b in range(B):
        for s_idx in range(S):
            for f in range(H):
                v = y1[b, s_idx, f]
                prod = log_mul(mw, v, v)
                mul_count10 += 1
                y2[b, s_idx, f] = prod

# y3 = sum(y2)
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_11.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_11.txt")
y3 = np.zeros((B, S, 1), dtype=np.float64)

mul_count11 = add_count11 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    mw.writerow(["(no mults in y3 = sum(y2))"])
    for b in range(B):
        for s_idx in range(S):
            acc = 0.0
            for f in range(H):
                prev   = acc
                addend = y2[b, s_idx, f]
                acc    = log_add(aw, prev, addend)
                add_count11 += 1
            y3[b, s_idx, 0] = acc

# v1 = y3 * inv_H
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_12.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_12.txt")
v1 = np.zeros_like(y3, dtype=np.float64)

mul_count12 = add_count12 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    aw.writerow(["(no additions in v1 = y3 * inv_H)"])
    for b in range(B):
        for s_idx in range(S):
            y_val = y3[b, s_idx, 0]
            prod  = log_mul(mw, y_val, inv_H)
            mul_count12 += 1
            v1[b, s_idx, 0] = prod

# y4 = v1 + eps
eps = 1e-12
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_13.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_13.txt")
y4 = np.zeros_like(v1, dtype=np.float64)
mul_count13 = add_count13 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    mw.writerow(["(no multiplications in y4 = v1 + eps)"])
    for b in range(B):
        for s_idx in range(S):
            prev = v1[b, s_idx, 0]
            newv = log_add(aw, prev, eps)
            add_count13 += 1
            y4[b, s_idx, 0] = newv

# Save y4 as inv_sqrt_2.txt (variance + eps)
inv1_path = os.path.join(DIR_NONLIN, "inv_sqrt_1_block0.txt")
bound_lo, bound_hi = 0.001, 210.0
y4_clamped = np.clip(y4, bound_lo, bound_hi)
with open(inv1_path, "w") as nf:
    for b in range(B):
        for s_idx in range(S):
            inp = float(y4_clamped[b, s_idx, 0])
            out = float(1.0 / np.sqrt(inp))
            nf.write(f"{f64_hex(out)}, {f64_hex(inp)}\n")

# inv_sqrt2 = 1 / sqrt(y4)
inv_sqrt2 = 1.0 / np.sqrt(y4_clamped)

# y5 = y1 * inv_sqrt2
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_14.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_14.txt")
y5 = np.zeros_like(y1, dtype=np.float64)

mul_count14 = add_count14 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    aw.writerow(["(no additions in y5 = y1 * inv_sqrt2)"])
    for b in range(B):
        for s_idx in range(S):
            scale = inv_sqrt2[b, s_idx, 0]
            for f in range(H):
                v = y1[b, s_idx, f]
                prod = log_mul(mw, v, scale)
                mul_count14 += 1
                y5[b, s_idx, f] = prod

# ln1_out = y5 * ln1_w + ln1_b
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_15.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_15.txt")
ln1_out = np.zeros_like(y5, dtype=np.float64)

mul_count15 = add_count15 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    for b in range(B):
        for s_idx in range(S):
            for f in range(H):
                yv   = y5[b, s_idx, f]
                wv   = float(ln1_w[f])
                prod = log_mul(mw, yv, wv)
                mul_count15 += 1

                bias = float(ln1_b[f])
                newv = log_add(aw, prod, bias)
                add_count15 += 1

                ln1_out[b, s_idx, f] = newv

# 7) Feed-Forward Network (dense in/out + GELU input dump)
Wi = np.array(params["bert.encoder.layer.0.intermediate.dense.weight"], dtype=float)
Bi = np.array(params["bert.encoder.layer.0.intermediate.dense.bias"],   dtype=float)
Wo = np.array(params["bert.encoder.layer.0.output.dense.weight"],       dtype=float)
Bo = np.array(params["bert.encoder.layer.0.output.dense.bias"],         dtype=float)

# hidden = ln1_out @ Wi.T + Bi
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_16.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_16.txt")

Bsz, S, H_in = ln1_out.shape       # 256
H_out, H_chk = Wi.shape            # 1024 x 256
assert H_chk == H_in, "Dim mismatch between ln1_out and Wi"
hidden = np.zeros((Bsz, S, H_out), dtype=np.float64)

mul_count16 = add_count16 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    for b in range(Bsz):
        for s_idx in range(S):
            for of in range(H_out):
                acc = 0.0
                for inf in range(H_in):
                    x_val = ln1_out[b, s_idx, inf]
                    w_val = Wi[of, inf]
                    prod  = log_mul(mw, x_val, w_val)
                    mul_count16 += 1

                    prev = acc
                    acc  = log_add(aw, prev, prod)
                    add_count16 += 1

                # bias
                prev = acc
                acc  = log_add(aw, prev, float(Bi[of]))
                add_count16 += 1

                hidden[b, s_idx, of] = acc

gelu_in_path = os.path.join(DIR_NONLIN, "gelu_0.txt")
count_gelu_in = 0
bound = 5.0
y_prime = np.clip(hidden, -bound, bound)
with open(gelu_in_path, "w") as gf:
    for b in range(Bsz):
        for s_idx in range(S):
            for f in range(H_out):
                y = gelu(y_prime[b, s_idx, f])
                gf.write(f"{f64_hex(y)}, {f64_hex(y_prime[b, s_idx, f])}\n")
                count_gelu_in += 1

# Apply GELU
act = gelu(y_prime)

# ffn = act @ Wo.T + Bo
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_17.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_17.txt")

Bsz, S, H_in2  = act.shape        # 1024
H_out2, H_chk2 = Wo.shape         # 256 x 1024
assert H_chk2 == H_in2, "Dim mismatch between act and Wo"
ffn = np.zeros((Bsz, S, H_out2), dtype=np.float64)

mul_count17 = add_count17 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    for b in range(Bsz):
        for s_idx in range(S):
            for of in range(H_out2):
                acc = 0.0
                for inf in range(H_in2):
                    a_val = act[b, s_idx, inf]
                    w_val = Wo[of, inf]
                    prod  = log_mul(mw, a_val, w_val)
                    mul_count17 += 1

                    prev = acc
                    acc  = log_add(aw, prev, prod)
                    add_count17 += 1

                # bias
                prev = acc
                acc  = log_add(aw, prev, float(Bo[of]))
                add_count17 += 1

                ffn[b, s_idx, of] = acc

# 8) Residual + LayerNorm2
# Residual: res2 = ffn + ln1_out
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_18.txt")
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_18.txt")  # no mults
res2 = np.zeros_like(ffn, dtype=np.float64)

add_count18 = mul_count18 = 0
with open(add_path, "w", newline="") as af, open(mul_path, "w", newline="") as mf:
    aw = csv.writer(af); mw = csv.writer(mf)
    mw.writerow(["(no mults in res2 = ffn + ln1_out)"])
    for b in range(Bsz):
        for s_idx in range(S):
            for f in range(H):
                v1 = ffn[b, s_idx, f]
                v2 = ln1_out[b, s_idx, f]
                s  = log_add(aw, v1, v2)
                add_count18 += 1
                res2[b, s_idx, f] = s

# ---------- LN2 (mirrors LN1, new file indices) ----------
ln2_w  = np.array(params["bert.encoder.layer.0.output.LayerNorm.weight"], dtype=float)
ln2_b  = np.array(params["bert.encoder.layer.0.output.LayerNorm.bias"],   dtype=float)

# m2
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_19.txt")
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_19.txt")
m2 = np.zeros((B, S, 1), dtype=np.float64)

add_count19 = mul_count19 = 0
with open(add_path, "w", newline="") as af, open(mul_path, "w", newline="") as mf:
    aw = csv.writer(af); mw = csv.writer(mf)
    inv_H = 1.0 / H
    for b in range(B):
        for s_idx in range(S):
            acc = 0.0
            for f in range(H):
                prev   = acc
                addend = res2[b, s_idx, f]
                acc    = log_add(aw, prev, addend)
                add_count19 += 1
            mean_val = log_mul(mw, acc, inv_H)
            mul_count19 += 1
            m2[b, s_idx, 0] = mean_val

# t1 = res2 - m2  -> log & count as addition with negated second operand
add_path_sub1 = os.path.join(DIR_ADD, "encoder_block_0_add_19a.txt")
t1 = np.zeros_like(res2, dtype=np.float64)
add_count_sub1 = 0
with open(add_path_sub1, "w", newline="") as af:
    aw = csv.writer(af)
    for b in range(B):
        for s_idx in range(S):
            m_val = m2[b, s_idx, 0]
            for f in range(H):
                r = res2[b, s_idx, f]
                d = log_sub_as_add(aw, r, m_val)  # a + (-b)
                t1[b, s_idx, f] = d
                add_count_sub1 += 1

# t2 = t1 * t1
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_20.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_20.txt")
t2 = np.zeros_like(t1, dtype=np.float64)
mul_count20 = add_count20 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    aw.writerow(["(no additions for t2 = t1 * t1)"])
    for b in range(B):
        for s_idx in range(S):
            for f in range(H):
                v = t1[b, s_idx, f]
                prod = log_mul(mw, v, v)
                mul_count20 += 1
                t2[b, s_idx, f] = prod

# t3 = sum(t2)
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_21.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_21.txt")
t3 = np.zeros((B, S, 1), dtype=np.float64)
mul_count21 = add_count21 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    mw.writerow(["(no mults in t3 = sum(t2))"])
    for b in range(B):
        for s_idx in range(S):
            acc = 0.0
            for f in range(H):
                prev   = acc
                addend = t2[b, s_idx, f]
                acc    = log_add(aw, prev, addend)
                add_count21 += 1
            t3[b, s_idx, 0] = acc

# v2 = t3 * inv_H
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_22.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_22.txt")
v2 = np.zeros_like(t3, dtype=np.float64)
mul_count22 = add_count22 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    aw.writerow(["(no additions in v2 = t3 * inv_H)"])
    for b in range(B):
        for s_idx in range(S):
            tv = t3[b, s_idx, 0]
            prod = log_mul(mw, tv, inv_H)
            mul_count22 += 1
            v2[b, s_idx, 0] = prod

# t4 = v2 + eps
eps = 1e-12
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_23.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_23.txt")
t4 = np.zeros_like(v2, dtype=np.float64)
mul_count23 = add_count23 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    mw.writerow(["(no multiplications in t4 = v2 + eps)"])
    for b in range(B):
        for s_idx in range(S):
            prev = v2[b, s_idx, 0]
            newv = log_add(aw, prev, eps)
            add_count23 += 1
            t4[b, s_idx, 0] = newv

# Save t4 as inv_sqrt_2.txt (variance + eps for LN2)
t4_path = os.path.join(DIR_NONLIN, "inv_sqrt_2_block0.txt")
bound_lo, bound_hi = 0.001, 210.0
t4_clamped = np.clip(t4, bound_lo, bound_hi)
with open(t4_path, "w") as nf:
    for b in range(B):
        for s_idx in range(S):
            inp = float(t4_clamped[b, s_idx, 0])
            out = float(1.0 / np.sqrt(inp))
            nf.write(f"{f64_hex(out)}, {f64_hex(inp)}\n")

# inv_sqrt3 = 1 / sqrt(t4)
inv_sqrt3 = 1.0 / np.sqrt(t4_clamped)

# t5 = t1 * inv_sqrt3
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_24.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_24.txt")
t5 = np.zeros_like(t1, dtype=np.float64)
mul_count24 = add_count24 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    aw.writerow(["(no additions in t5 = t1 * inv_sqrt3)"])
    for b in range(B):
        for s_idx in range(S):
            scale = inv_sqrt3[b, s_idx, 0]
            for f in range(H):
                v  = t1[b, s_idx, f]
                prod = log_mul(mw, v, scale)
                mul_count24 += 1
                t5[b, s_idx, f] = prod

# ln2_out = t5 * ln2_w + ln2_b
mul_path = os.path.join(DIR_MUL, "encoder_block_0_mul_25.txt")
add_path = os.path.join(DIR_ADD, "encoder_block_0_add_25.txt")
ln2_out = np.zeros_like(t5, dtype=np.float64)
mul_count25 = add_count25 = 0
with open(mul_path, "w", newline="") as mf, open(add_path, "w", newline="") as af:
    mw = csv.writer(mf); aw = csv.writer(af)
    for b in range(B):
        for s_idx in range(S):
            for f in range(H):
                tv   = t5[b, s_idx, f]
                wv   = float(ln2_w[f])
                prod = log_mul(mw, tv, wv)
                mul_count25 += 1

                bias = float(ln2_b[f])
                newv = log_add(aw, prod, bias)
                add_count25 += 1

                ln2_out[b, s_idx, f] = newv

# Final output
print("Encoder Block 0 output (post-LN) shape:", ln2_out.shape)
print(ln2_out)

# 9) Totals for operations
# Start with Q/K/V tallies
add_total = total_add
mul_total = total_mul
div_total = 0
sub_total = 0

# Sweep all *_count* vars
for name, val in list(globals().items()):
    if name.startswith("add_count"):
        add_total += val
    elif name.startswith("mul_count"):
        mul_total += val
    elif name.startswith("div_count"):
        div_total += val
    elif name.startswith("sub_count"):
        sub_total += val  # remains 0 since we converted subs to adds

# Non-linear ops
gelu_calls = count_gelu_in          # GELU applied to each hidden element
exp_calls  = count_exp              # number of exp() evaluations in softmax
inv_sqrt_calls = 1 + y4.size + t4.size  # 1 for Hd, plus LN1 and LN2 vectors

print("\n===== TOTAL OPS (Block 0) =====")
print(f"Additions:        {add_total}")
print(f"Multiplications:  {mul_total}")
print(f"Divisions:        {div_total}")
print("Non-linear:")
print(f"  GELU (inputs logged): {gelu_calls}")
print(f"  exp() calls:           {exp_calls}")
print(f"  inv_sqrt calls:        {inv_sqrt_calls}")