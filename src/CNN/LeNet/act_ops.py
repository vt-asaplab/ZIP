# act1_y_yprime_dump.py
import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

def float64_to_hex(x: float) -> str:
    u = np.float64(x).view(np.uint64)
    return f"0x{int(u):016x}"

def main():
    parser = argparse.ArgumentParser(description="activation for act*_in, dump y,y' in hex")
    parser.add_argument("--act", choices=["gelu", "elu", "selu"], required=True,
                        help="activation to use on the input (y')")
    parser.add_argument("--key", choices=["act1_in", "act2_in", "act3_in", "act4_in"],
                        default="act1_in",
                        help="which layer input to use from lenet_inference_layer_inputs.json (default: act1_in)")
    args = parser.parse_args()
    act = args.act.lower()
    layer_key = args.key  # e.g., "act1_in"

    script_dir = os.path.abspath(os.path.dirname(__file__))

    TARGET_ROOT = os.path.abspath(
        os.path.join(
            script_dir,
            "..", "..",
            "proof_generation", "ZIP_proof_generation", "ZIP_lookup", "examples"
        )
    )

    acts_path = os.path.join(script_dir, "lenet_inference_layer_inputs.json")

    # derive layer number from key: "act1_in" -> "1", etc.
    layer_num = layer_key.replace("_in", "").replace("act", "")
    out_dir   = os.path.join(TARGET_ROOT, f"y_yprime_examples_lenet_act{layer_num}")
    os.makedirs(out_dir, exist_ok=True)

    out_path  = os.path.join(out_dir, f"{act}_y_yprime.txt")

    with open(acts_path) as f:
        acts = json.load(f)

    if layer_key not in acts:
        raise KeyError(f"Expected key '{layer_key}' in lenet_inference_layer_inputs.json")

    x_np = np.array(acts[layer_key], dtype=np.float64)
    x    = torch.from_numpy(x_np).to(dtype=torch.float64)

    bound = 30.0 if act in ("elu", "selu") else 5.0
    y_prime = torch.clamp(x, min=-bound, max=bound)

    # ---- Activation(y') -> y ----
    if act == "gelu":
        y = F.gelu(y_prime, approximate="tanh")
    elif act == "elu":
        y = F.elu(y_prime)
    else:  # selu
        y = F.selu(y_prime)

    def to_hex_pairs(y_np, yp_np):
        for yi, ypi in zip(y_np.reshape(-1), yp_np.reshape(-1)):
            u_y  = np.float64(yi).view(np.uint64)
            u_yp = np.float64(ypi).view(np.uint64)
            yield f"0x{int(u_y):016x}, 0x{int(u_yp):016x}\n"

    y_np  = y.cpu().numpy()
    yp_np = y_prime.cpu().numpy()

    with open(out_path, "w") as txt:
        txt.writelines(to_hex_pairs(y_np, yp_np))

    print(f"[{layer_key}] act={act}  shape={tuple(x_np.shape)}  elements={x_np.size}")

if __name__ == "__main__":
    main()
