import os, json, numpy as np, struct
def f64_hex(x: float) -> str:
    return "0x" + format(struct.unpack(">Q", struct.pack(">d", float(x)))[0], "016x")

base = os.path.join(os.path.dirname(__file__), "lenet_output")
os.makedirs(base, exist_ok=True)
mul_path = os.path.join(base, "multiplication.txt")
add_path = os.path.join(base, "addition.txt")

with open("lenet_inference_layer_inputs.json") as f: acts = json.load(f)
X = np.array(acts["pool1_in"], dtype=np.float64)  # [1,6,28,28]

batch, ch, H, W = X.shape
kernel = stride = 2
H_out, W_out = H // stride, W // stride

mul_count = add_count = 0
with open(mul_path, "a") as mf, open(add_path, "a") as af:
    factor = 1.0 / (kernel * kernel)  # 0.25
    for n in range(batch):
        for c in range(ch):
            for i in range(H_out):
                for j in range(W_out):
                    acc = 0.0
                    for dx, dy in ((0,0),(0,1),(1,0),(1,1)):
                        val = X[n, c, stride*i + dx, stride*j + dy]
                        prev = acc; acc = prev + val
                        af.write(f"{f64_hex(prev)}, {f64_hex(val)}, {f64_hex(acc)}\n"); add_count += 1
                    prod = acc * factor
                    mf.write(f"{f64_hex(factor)}, {f64_hex(acc)}, {f64_hex(prod)}\n"); mul_count += 1

print(f"[pool1] multiplications logged: {mul_count}")
print(f"[pool1] additions       logged: {add_count}")
