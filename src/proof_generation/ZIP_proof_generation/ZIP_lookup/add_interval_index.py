import argparse, re, struct, sys, os

NUM_RE = re.compile(r'0[xX][0-9a-fA-F]+|[0-9]+')

def tok_to_u64(tok: str) -> int:
    tok = tok.strip()
    if tok.lower().startswith("0x"):
        return int(tok, 16)
    return int(tok, 10)

def u64_to_float(u: int) -> float:
    return struct.unpack(">d", u.to_bytes(8, "big"))[0]

def read_intervals(path: str):
    with open(path, "r") as f:
        txt = f.read()
    toks = NUM_RE.findall(txt)
    if len(toks) < 2:
        raise ValueError(f"Intervals file {path} must contain at least 2 numbers")
    vals_f = [u64_to_float(tok_to_u64(t)) for t in toks]
    for i in range(len(vals_f)-1):
        if not (vals_f[i] <= vals_f[i+1]):
            raise ValueError(f"Intervals not nondecreasing near index {i} in {path}")
    return vals_f, toks

def find_interval_index(intervals_f, yprime_f):
    n = len(intervals_f)
    if yprime_f <= intervals_f[0]:
        return 0
    if yprime_f >= intervals_f[-1]:
        return n - 2
    for i in range(n - 1):
        if intervals_f[i] <= yprime_f <= intervals_f[i+1]:
            return i
    raise ValueError(f"y'={yprime_f!r} not found in any interval")

def process_pairs(pairs_path: str, intervals_f):
    out_lines = []
    with open(pairs_path, "r") as f:
        for line in f:
            raw = line.rstrip("\n")
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                out_lines.append(raw)
                continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 2:
                raise ValueError(f"Bad line (need at least 2 values): {raw}")
            try:
                yp_u = tok_to_u64(parts[1])
                yp_f = u64_to_float(yp_u)
            except Exception as e:
                raise ValueError(f"Failed parsing y' in line: {raw}\n{e}") from e

            idx = find_interval_index(intervals_f, yp_f)

            new_line = f"{parts[0]}, {parts[1]}, {idx}"
            out_lines.append(new_line)
    return out_lines

def main():
    ap = argparse.ArgumentParser(description="Annotate y_yprime pairs with interval index.")
    ap.add_argument("--pairs", required=True, help="Path to {act}_y_yprime.txt")
    ap.add_argument("--intervals", required=True, help="Path to {act}_intervals_ieee754.txt")
    ap.add_argument("--output", help="Output path (default: overwrite --pairs)")
    args = ap.parse_args()

    intervals_f, _interval_tokens = read_intervals(args.intervals)
    out_lines = process_pairs(args.pairs, intervals_f)

    out_path = args.output or args.pairs
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
        f.write("\n")

if __name__ == "__main__":
    main()
