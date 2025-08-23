import argparse

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
output_go_file = os.path.join(script_dir, "config.go")


# Helper function to parse string as boolean
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1'):
        return True
    elif v.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (true/false).')

# Argument parsing
parser = argparse.ArgumentParser(description="Generate Go config from preset and flags.")
parser.add_argument("--preset", choices=["gelu", "selu", "elu"], required=True, help="Activation preset")
parser.add_argument("--instances", type=int, required=True, help="Number of instances")
parser.add_argument("--mode", choices=["fig1", "table1/2"], required=True, help="Evaluation mode flag")
parser.add_argument("--proving", type=str2bool, required=True, help="Set proving flag (true or false)")

args = parser.parse_args()

# Define configurations for each preset (excluding SIZE_VALUE and PROVING)
presets = {
    "gelu": {
        "TABLE_SIZE": 70,
        "PRIVATE_VECTOR_SIZE": 10,
        "TABLE_PRIME_SIZE": 8,
        "PRIVATE_VECTOR_PRIME_SIZE": 2,
        "E_VALUE": 11,
        "M_VALUE": 52,
    },
    "selu": {
        "TABLE_SIZE": 35,
        "PRIVATE_VECTOR_SIZE": 7,
        "TABLE_PRIME_SIZE": 6,
        "PRIVATE_VECTOR_PRIME_SIZE": 2,
        "E_VALUE": 11,
        "M_VALUE": 52,
    },
    "elu": {
        "TABLE_SIZE": 36,
        "PRIVATE_VECTOR_SIZE": 6,
        "TABLE_PRIME_SIZE": 7,
        "PRIVATE_VECTOR_PRIME_SIZE": 2,
        "E_VALUE": 11,
        "M_VALUE": 52,
    }
}

# Load preset
params = presets[args.preset]
params["NUM_INSTANCES"] = args.instances
params["EVAL_MODE"] = f'"{args.mode}"'
params["ACTIVATION"] = f'"{args.preset}"'
params["PROVING"] = args.proving 

# Logic to determine SIZE_VALUE
n = args.instances
preset = args.preset

if preset == "gelu":
    if n <= 7:
        size_value = 8
    elif n <= 255:
        size_value = 12
    else:
        size_value = 16
elif preset in ["elu", "selu"]:
    if n <= 7:
        size_value = 8
    elif n <= 511:
        size_value = 12
    else:
        size_value = 16
else:
    raise ValueError("Unsupported preset.")

params["SIZE_VALUE"] = size_value

# Generate Go file
output_go_file = "config.go"
const_lines = ["package main\n\nconst (\n"]

for key, value in params.items():
    if isinstance(value, bool):
        const_lines.append(f"\t{key} = {str(value).lower()}\n")
    else:
        const_lines.append(f"\t{key} = {value}\n")

const_lines.append(")\n")

with open(output_go_file, "w") as f:
    f.writelines(const_lines)
