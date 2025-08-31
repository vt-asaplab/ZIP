#!/usr/bin/env python3
import argparse, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_GO  = os.path.join(SCRIPT_DIR, "config.go")

def main():
    p = argparse.ArgumentParser(description="Generate Go config.go for linear ops (fixed E/M).")
    p.add_argument("--num-add", type=int, required=True, help="NUM_INSTANCES_ADD")
    p.add_argument("--num-mul", type=int, required=True, help="NUM_INSTANCES_MULT")
    p.add_argument("--size",    type=int, required=True, help="SIZE_VALUE (any integer)")
    args = p.parse_args()

    go_src = f"""package main

const (
\tNUM_INSTANCES_ADD  = {args.num_add}
\tNUM_INSTANCES_MULT = {args.num_mul}
\tE_VALUE            = 11
\tM_VALUE            = 52
\tSIZE_VALUE         = {args.size}
)
"""
    with open(OUTPUT_GO, "w") as f:
        f.write(go_src)

if __name__ == "__main__":
    main()
