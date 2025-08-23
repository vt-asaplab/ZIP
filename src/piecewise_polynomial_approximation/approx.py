import sympy as sp
from NFGen.main import generate_nonlinear_config
import NFGen.CodeTemplet.templet as temp
import NFGen.PerformanceModel.time_ops as to
import math
import numpy as np
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#constant factors
PAI = 3.141592653589793
TAU_2 = 0.959502
ALPHA1 = 1.0
ALPHA2 = 1.6732632423543772848170429916717
LAMBDA = 1.0507009873554804934193349852946
E = 2.7182818
C1 = 0.044715
TAU_half = 1.7725
G = 0.5

platform = "Rep3"
f = 48 
n = 96
profiler_file = 'NFGen/src/NFGen/PerformanceModel/' + platform + "_kmProfiler.pkl"

def func_inv_sqrt(x, lib=sp):
    return 1 / lib.sqrt(x)

def func_reciprocal(x):
    return 1 / x

def func_exp(x, lib=sp):
    return lib.exp(x)

def elu(x):
    pos_flag = x > 0
    res = x * pos_flag + (1 - pos_flag) * ALPHA1 * (func_exp(x, lib=np) - 1)
    return res
    
def selu(x):
    pos_flag = x > 0
    res = LAMBDA * x * pos_flag + (1 - pos_flag) * LAMBDA * (
        ALPHA2 * func_exp(x, lib=np) - ALPHA2)
    return res

def gelu(x):
    constant = math.sqrt(2 / PAI)
    x1 = constant * (x + C1 * x * x * x)
    ep = func_exp(x1, lib=np)
    en = func_exp(-x1, lib=np)
    return 0.5 * x * (1 + ((ep - en) * func_reciprocal(ep + en)))


elu_config = {
    "function": elu, # function config.
    "range": (-30, 30),
    "k_max": 10, # set the maximum order.
    "tol": 1e-4, # percision config.
    "ms": 1000, # maximum samples.
    "zero_mask": 1e-6,
    "n": n, # <n, f> fixed-point config.
    "f": f,
    "profiler": profiler_file, # profiler model source file.
    "code_templet": temp.templet_spdz, # spdz templet.
    "code_language": "python", # indicating the templet language.
    "config_file": "./elu_approx.py", # generated code file.
    "time_dict": to.basic_time[platform], # basic operation time cost.
    "derivative_flag": False,
}

selu_config = {
    "function": selu, # function config.
    "range": (-30, 30),
    "k_max": 12, # set the maximum order.
    "tol": 1e-4, # percision config.
    "ms": 1000, # maximum samples.
    #"range": (-5, 5),
    #"k_max": 10, # set the maximum order.
    #"tol": 1e-10, # percision config.
    #"ms": 1000, # maximum samples.
    "zero_mask": 1e-6,
    "n": n, # <n, f> fixed-point config.
    "f": f,
    "profiler": profiler_file, # profiler model source file.
    "code_templet": temp.templet_spdz, # spdz templet.
    "code_language": "python", # indicating the templet language.
    "config_file": "./selu_approx.py", # generated code file.
    "time_dict": to.basic_time[platform], # basic operation time cost.
    "derivative_flag": False,
}

gelu_config = {
    "function": gelu, # function config.
    "range": (-5, 5),
    "k_max": 10, # set the maximum order.
    "tol": 1e-6, # percision config.
    "ms": 1000, # maximum samples.
    "zero_mask": 1e-6,
    "n": n, # <n, f> fixed-point config.
    "f": f,
    "profiler": profiler_file, # profiler model source file.
    "code_templet": temp.templet_spdz, # spdz templet.
    "code_language": "python", # indicating the templet language.
    "config_file": "./gelu_approx.py", # generated code file.
    "time_dict": to.basic_time[platform], # basic operation time cost.
    "derivative_flag": False,
}

#softmax
exp_config = {
    "function": func_exp, # function config.
    "range": (-8, 20),
    "k_max": 13, # set the maximum order.
    "tol": 1e-3, # percision config.
    "ms": 1000, # maximum samples.
    "zero_mask": 1e-6,
    "n": n, # <n, f> fixed-point config.
    "f": f,
    "profiler": profiler_file, # profiler model source file.
    "code_templet": temp.templet_spdz, # spdz templet.
    "code_language": "python", # indicating the templet language.
    "config_file": "./softmax_approx.py", # generated code file.
    "time_dict": to.basic_time[platform], # basic operation time cost.
}

#layernorm
func_inv_sqrt_config = {
    "function": func_inv_sqrt, # function config.
    "range": (0.001, 210),
    "k_max": 10, # set the maximum order.
    "tol": 1e-3, # percision config.
    "ms": 1000, # maximum samples.
    "zero_mask": 1e-6,
    "n": n, # <n, f> fixed-point config.
    "f": f,
    "profiler": profiler_file, # profiler model source file.
    "code_templet": temp.templet_spdz, # spdz templet.
    "code_language": "python", # indicating the templet language.
    "config_file": "./layernorm_approx.py", # generated code file.
    "time_dict": to.basic_time[platform], # basic operation time cost.
}

def main():
    #using NFGen library to generate the piecewise polynomial approximation
    non_linear_operation = sys.argv[1]
    if non_linear_operation == "elu":
        generate_nonlinear_config(elu_config)
    elif non_linear_operation == "selu":
        generate_nonlinear_config(selu_config)
    elif non_linear_operation == "gelu":
        generate_nonlinear_config(gelu_config)
    elif non_linear_operation == "softmax":
        generate_nonlinear_config(exp_config)
    elif non_linear_operation == "layernorm":
        generate_nonlinear_config(func_inv_sqrt_config)            
    else:
        print(f"Unsupported non-linear function: {non_linear_operation}")
        sys.exit(1)    

if __name__ == "__main__":
    main()
