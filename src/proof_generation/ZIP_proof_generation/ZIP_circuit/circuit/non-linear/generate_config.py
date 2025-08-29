import argparse
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
output_go_file = os.path.join(script_dir, "config.go")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1'):
        return True
    if v.lower() in ('false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected (true/false).')

parser = argparse.ArgumentParser(description="Generate Go config from preset and flags.")
parser.add_argument("--preset", choices=["gelu", "selu", "elu", "softmax", "layernorm"], required=True, help="Activation preset")
parser.add_argument("--instances", type=int, required=True, help="Number of instances")
parser.add_argument("--mode", choices=["fig1", "table1/2", "table"], required=True, help="Evaluation mode flag")
parser.add_argument("--proving", type=str2bool, required=True, help="Set proving flag (true or false)")
parser.add_argument("--table_size", type=int, help="Override TABLE_SIZE")
parser.add_argument("--table_prime_size", type=int, help="Override TABLE_PRIME_SIZE")
parser.add_argument("--values_dir", type=str, default="y_yprime_examples",
                    help="Directory containing Y/Y' pair files (absolute or relative)")

args = parser.parse_args()

presets = {
    "gelu": {
        "TABLE_SIZE": 70,
        "PRIVATE_VECTOR_SIZE": 10,
        "TABLE_PRIME_SIZE": 8,
        "PRIVATE_VECTOR_PRIME_SIZE": 2,
        "E_VALUE": 11,
        "M_VALUE": 52,
        "DELTA_VALUE": 0x3fb70a3d70a3d70a, # 0.09
    },
    "selu": {
        "TABLE_SIZE": 35,
        "PRIVATE_VECTOR_SIZE": 7,
        "TABLE_PRIME_SIZE": 6,
        "PRIVATE_VECTOR_PRIME_SIZE": 2,
        "E_VALUE": 11,
        "M_VALUE": 52,
        "DELTA_VALUE": 0x3F4D7DBF487FCB92, # 0.0009
    },
    "elu": {
        "TABLE_SIZE": 36,
        "PRIVATE_VECTOR_SIZE": 6,
        "TABLE_PRIME_SIZE": 7,
        "PRIVATE_VECTOR_PRIME_SIZE": 2,
        "E_VALUE": 11,
        "M_VALUE": 52,
        "DELTA_VALUE": 0x3F4D7DBF487FCB92, # 0.0009
    },
    "softmax": {
        "TABLE_SIZE": 70,
        "PRIVATE_VECTOR_SIZE": 7,
        "TABLE_PRIME_SIZE": 11,
        "PRIVATE_VECTOR_PRIME_SIZE": 2,
        "E_VALUE": 11,
        "M_VALUE": 52,
        "DELTA_VALUE": 0x3F826E978D4FDF3B, # 0.009
    },
    "layernorm": {
        "TABLE_SIZE": 88,
        "PRIVATE_VECTOR_SIZE": 4,
        "TABLE_PRIME_SIZE": 23,
        "PRIVATE_VECTOR_PRIME_SIZE": 2,
        "E_VALUE": 11,
        "M_VALUE": 52,
        "DELTA_VALUE": 0x3F826E978D4FDF3B, # 0.009
    }    
}

params = presets[args.preset].copy()
params["NUM_INSTANCES"] = args.instances
params["EVAL_MODE"] = f'"{args.mode}"'
params["ACTIVATION"] = f'"{args.preset}"'
params["PROVING"] = args.proving
params["VALUES_DIR"] = f'"{args.values_dir}"' 

if args.table_size is not None:
    if args.table_size <= 0:
        raise argparse.ArgumentTypeError("--table_size must be positive")
    params["TABLE_SIZE"] = args.table_size

if args.table_prime_size is not None:
    if args.table_prime_size <= 0:
        raise argparse.ArgumentTypeError("--table_prime_size must be positive")
    params["TABLE_PRIME_SIZE"] = args.table_prime_size

if args.table_size is not None and args.table_prime_size is not None:
    denom = args.table_prime_size - 1
    if denom <= 0:
        raise argparse.ArgumentTypeError("--table_prime_size must be >= 2 (denominator > 0)")
    if args.table_size % denom != 0:
        raise argparse.ArgumentTypeError(
            f"--table_size ({args.table_size}) must be divisible by (--table_prime_size - 1) "
            f"= {denom} to get an integer PRIVATE_VECTOR_SIZE"
        )
    params["PRIVATE_VECTOR_SIZE"] = args.table_size // denom

n = args.instances
preset = args.preset
if preset == "gelu":
    size_value = 8 if n <= 7 else 12 if n <= 255 else 16
elif preset in ["elu", "selu", "softmax", "layernorm"]:
    size_value = 8 if n <= 7 else 12 if n <= 511 else 16
else:
    raise ValueError("Unsupported preset.")
params["SIZE_VALUE"] = size_value

# Generate Go file
const_lines = ["package main\n\nconst (\n"]
for key, value in params.items():
    if isinstance(value, bool):
        const_lines.append(f"\t{key} = {str(value).lower()}\n")
    elif isinstance(value, int) and "DELTA" in key:
        const_lines.append(f"\t{key} = 0x{value:X}\n")
    else:
        const_lines.append(f"\t{key} = {value}\n")
const_lines.append(")\n")

with open("config.go", "w") as f:
    f.writelines(const_lines)
