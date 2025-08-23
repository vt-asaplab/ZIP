import re
import ast
import sys
import os
import struct
from typing import List, Union
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def float_to_ieee754_hex(f: float) -> str:
    packed = struct.pack('>d', f)
    bits, = struct.unpack('>Q', packed)
    return f"0x{bits:016x}"

def format_hex_line(lst: List[float]) -> str:
    hex_vals = [float_to_ieee754_hex(x) for x in lst]
    return ", ".join(hex_vals)

def hex_str(data: Union[List[float], List[List[float]]]) -> str:

    if all(isinstance(x, float) for x in data):
        return format_hex_line(data) + "\n"
    else:
        return "\n".join(format_hex_line(row) for row in data) + "\n"

def save_hex(path: str, data: Union[List[float], List[List[float]]]) -> None:
    with open(path, "w") as f:
        f.write(hex_str(data))

def convert_to_IEEE754_double(activation, intervals, coefficients):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "precomputed_lookup_tables_ieee754_hex")
    os.makedirs(out_dir, exist_ok=True)

    save_hex(os.path.join(out_dir, f"{activation}_intervals_ieee754.txt"), intervals)
    save_hex(os.path.join(out_dir, f"{activation}_coefficients_ieee754.txt"), coefficients)

def main():
    activation = sys.argv[1]

    input_filename = activation + '_approx.py'

    out_py_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "precomputed_lookup_tables_ieee754")
    os.makedirs(out_py_dir, exist_ok=True)
    output_filename = os.path.join(out_py_dir, activation + '_approx.py')

    patterns = {
        'breaks': r"^\s*breaks\s*=",
        'coeffA': r"^\s*coeffA\s*=",
        'scaler': r"^\s*scaler\s*="
    }

    extracted = {}
    with open(input_filename, 'r') as infile:
        for line in infile:
            for key, pattern in patterns.items():
                if re.match(pattern, line):
                    extracted[key] = line.strip()
                    break

    # Process breaks
    if 'breaks' in extracted:
        try:
            parts = extracted['breaks'].split("=", 1)
            breaks_list = ast.literal_eval(parts[1].strip())
            if activation == "elu":
                breaks_list.append(30.0)
            elif activation == "selu":
                breaks_list.append(30.0)
            elif activation == "gelu":
                breaks_list.append(5.0)
            elif activation == "softmax":
                breaks_list.append(20.0)
            elif activation == "layernorm":
                breaks_list.append(210.0)
            intervals = [float(x) for x in breaks_list]
        except Exception as e:
            print(f"Error parsing breaks: {e}")
            intervals = None
    else:
        intervals = None

    # Process coeffA and scaler with NumPy float64 multiply
    if 'coeffA' in extracted and 'scaler' in extracted:
        try:
            coeffA_list = ast.literal_eval(extracted['coeffA'].split("=", 1)[1].strip())
            scaler_list = ast.literal_eval(extracted['scaler'].split("=", 1)[1].strip())

            coeffA_arr = np.array(coeffA_list, dtype=np.float64)
            scaler_arr = np.array(scaler_list, dtype=np.float64)

            # element-wise multiply in float64
            coefficients_arr = coeffA_arr * scaler_arr

            coefficients = coefficients_arr.tolist()
        except Exception as e:
            print(f"Error processing coeffA and scaler: {e}")
            coefficients = None
    else:
        coefficients = None

    if intervals is None or coefficients is None:
        print("Error: Could not extract intervals or coefficients.")
        sys.exit(1)

    # Generate the piecewise polynomial approximation module
    FILE_TEMPLATE = f'''import torch
import torch.nn as nn

intervals = {intervals}
coefficients = {coefficients}

def approx_{activation}(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)

    def poly_eval(z: torch.Tensor, coeffs):
        # Evaluate the polynomial sum of coeffs[k] * z^k
        result = torch.zeros_like(z)
        for power, c in enumerate(coeffs):
            result += c * (z ** power)
        return result

    num_intervals = len(intervals)
    for i in range(num_intervals - 1):
        lower = intervals[i]
        upper = intervals[i + 1]
        mask = (x >= lower) & (x < upper)
        out[mask] = poly_eval(x[mask], coefficients[i])

    # Values >= the last boundary
    mask_last = (x >= intervals[-1])
    out[mask_last] = poly_eval(x[mask_last], coefficients[-1])

    # Values < the first boundary
    mask_first = (x < intervals[0])
    out[mask_first] = poly_eval(x[mask_first], coefficients[0])

    return out
'''

    with open(output_filename, 'w') as outfile:
        outfile.write(FILE_TEMPLATE)

    convert_to_IEEE754_double(activation, intervals, coefficients)

    print(f"Piecewise polynomial function for {activation} activation has been generated in {output_filename}")

if __name__ == "__main__":
    main()
