import os, json, numpy as np, struct
def f64_hex(x: float) -> str:
    return "0x" + format(struct.unpack(">Q", struct.pack(">d", float(x)))[0], "016x")

base = os.path.join(os.path.dirname(__file__), "lenet_output")
os.makedirs(base, exist_ok=True)
mul_path = os.path.join(base, "multiplication.txt")
add_path = os.path.join(base, "addition.txt")

with open("lenet_model_params.json") as f: params = json.load(f)
with open("lenet_inference_layer_inputs.json") as f: acts = json.load(f)

X = np.array(acts["conv2_in"], dtype=np.float64)       # [1,6,14,14]
W = np.array(params["conv2.weight"], dtype=np.float64) # [16,6,5,5]
b = np.array(params["conv2.bias"],   dtype=np.float64) # [16]

batch, in_ch, H, W_img = X.shape
out_ch, _, kH, kW = W.shape
H_out = H - kH + 1; W_out = W_img - kW + 1

mul_count = add_count = 0
with open(mul_path, "a") as mf, open(add_path, "a") as af:
    for n in range(batch):
        for oc in range(out_ch):
            for i in range(H_out):
                for j in range(W_out):
                    acc = 0.0
                    for ic in range(in_ch):
                        for u in range(kH):
                            for v in range(kW):
                                wv = W[oc, ic, u, v]; xv = X[n, ic, i + u, j + v]
                                prod = wv * xv
                                mf.write(f"{f64_hex(wv)}, {f64_hex(xv)}, {f64_hex(prod)}\n"); mul_count += 1
                                prev = acc; acc = prev + prod
                                af.write(f"{f64_hex(prev)}, {f64_hex(prod)}, {f64_hex(acc)}\n"); add_count += 1
                    prev = acc; addend = float(b[oc]); acc = prev + addend
                    af.write(f"{f64_hex(prev)}, {f64_hex(addend)}, {f64_hex(acc)}\n"); add_count += 1

print(f"[conv2] multiplications logged: {mul_count}")
print(f"[conv2] additions       logged: {add_count}")
