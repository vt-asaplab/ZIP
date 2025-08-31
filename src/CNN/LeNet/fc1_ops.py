import os, json, numpy as np, struct
def f64_hex(x: float) -> str:
    return "0x" + format(struct.unpack(">Q", struct.pack(">d", float(x)))[0], "016x")

base = os.path.join(os.path.dirname(__file__), "lenet_output")
os.makedirs(base, exist_ok=True)
mul_path = os.path.join(base, "multiplication.txt")
add_path = os.path.join(base, "addition.txt")

with open("lenet_model_params.json") as f: params = json.load(f)
with open("lenet_inference_layer_inputs.json") as f: acts = json.load(f)

X = np.array(acts["fc1_in"], dtype=np.float64)          # [1,400]
W = np.array(params["fc1.weight"], dtype=np.float64)    # [120,400]
b = np.array(params["fc1.bias"],   dtype=np.float64)    # [120]

batch, in_f = X.shape
out_f, _ = W.shape

mul_count = add_count = 0
with open(mul_path, "a") as mf, open(add_path, "a") as af:
    for n in range(batch):
        for of in range(out_f):
            acc = 0.0
            for inf in range(in_f):
                wv = W[of, inf]; xv = X[n, inf]; prod = wv * xv
                mf.write(f"{f64_hex(wv)}, {f64_hex(xv)}, {f64_hex(prod)}\n"); mul_count += 1
                prev = acc; acc = prev + prod
                af.write(f"{f64_hex(prev)}, {f64_hex(prod)}, {f64_hex(acc)}\n"); add_count += 1
            prev = acc; addend = float(b[of]); acc = prev + addend
            af.write(f"{f64_hex(prev)}, {f64_hex(addend)}, {f64_hex(acc)}\n"); add_count += 1

print(f"[fc1] multiplications   logged: {mul_count}")
print(f"[fc1] additions         logged: {add_count}")
