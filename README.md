# Zero-Knowledge AI Inference with High Precision

This repository contains the full implementation of **ZIP** paper (accepted to [**ACM CCS 2025**](https://www.sigsac.org/ccs/CCS2025/)).       
The manuscript will be available online shortly.

> **Warning**: This code is a research prototype intended for proof-of-concept purposes only and is not ready for production use.

## Code Structure

```text
ZIP/   
├── scripts/  
└── src/  
    ├── CNN/
    ├── LLM/
    ├── piecewise_polynomial_approximation
    │   ├── NFGen/
    │   └── precomputed_lookup_tables_ieee754/              # Precomputed lookup tables (decimal)
    └── proof_generation  
        ├── caulk/
        ├── ZIP_proof_generation
        │   ├── precomputed_lookup_tables_ieee754_hex/      # Precomputed lookup tables (hex)
        │   ├── ZIP_circuit 
        │   │   └── circuit      
        │   │       ├── linear/                             # 🔶 Our linear PLONK circuits (proved via zk-Location)
        │   │       └── non-linear/                         # 🔶 Our non-linear PLONK circuits
        │   └── ZIP_lookup                                  # 🔶 Our multi-lookup argument
        │       └── examples/                               # y, y′ inputs (see Eq. (2) in the paper)
        └── zk-Location/
```

## Prerequisites
Before running the scripts, ensure that you have the following installed:

- **Python**: 3.9+
- **Rust**: 1.72.0-nightly
- **Cargo**: 1.72.0-nightly (Rust package management)
- **Go**: go1.24.4

> Note: The package commands below use Debian/Ubuntu (```apt-get```). For Fedora/RHEL, replace with ```dnf/yum```.

## ⚠️ Git LFS Required (before cloning)

This repo uses **Git Large File Storage (LFS)** for the SRS files in  `src/proof_generation/caulk/srs` (~800 MB total).   

```bash
# Install Git LFS **before** you clone: 
sudo apt-get install git-lfs

# Initialize LFS hooks
git lfs install

# Normal clone
git clone https://github.com/vt-asaplab/ZIP.git
cd ZIP
```

## Installation

### Installing Python:

   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-pip
   ```

### Installing Rust and Cargo

   ```bash
   # Install rustup (Rust toolchain manager)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   . "$HOME/.cargo/env"     # load cargo into your shell
   
   # Install and set stable Rust globally
   rustup toolchain install 1.80.1
   rustup default 1.80.1
   
   # Verify
   rustc --version           # should print: rustc 1.80.1 (or newer)
   cargo --version
   ```

### Installing Go

   ```bash
   GO_VERSION=1.24.4
   wget https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz
   sudo rm -rf /usr/local/go
   sudo tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz

   # Add to PATH (if not already present)
   echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
   # (Optional) user GOPATH for binaries you install with `go install`
   echo 'export GOPATH=$HOME/go' >> ~/.bashrc
   echo 'export PATH=$GOPATH/bin:$PATH' >> ~/.bashrc
   source ~/.bashrc

   # Verify
   go version # # go version go1.24.4 linux/amd64
   ```

## How to Run

1. **Create the environment and install Python dependencies:**
   ```bash
   conda create -n zip python=3.9 -y
   conda activate zip
   ```
   ```bash
   python -m pip install -U torch numpy scipy pandas scikit-learn sympy torchvision
   chmod +x scripts/*.sh
   ```

2. **Precompute piecewise polynomial approximations for the target non-linear functions**
   ```bash
   ./scripts/precompute.sh
   ```
### Reproduce the Paper Results

- **Reproducibility note:**  
   - To match the paper’s numbers, use hardware comparable to our reference machine:
      - **48 CPU cores** (Intel® Xeon® Platinum 8360Y, 2.40 GHz) and **1 TB RAM**.  
   - The underlying **PLONK** prover uses **all available CPU cores** by default, so runtimes vary with core count.  
   - When running, ensure the machine is otherwise idle to obtain consistent timings.

1. **Generate Figure 1:** ✅
   > **Memory requirement:** ≈ **51 GB** RAM (peak usage)   
   > **End-to-end runtime:** ≈ **45 min** (wall-clock)   
   ```bash
   ./scripts/fig1.sh
   ```
2. **Generate Table 1 & 2:** ✅
   > **Memory requirement:** ≈ **250 GB** RAM (peak usage)   
   > **End-to-end runtime:** ≈ **90 min** (wall-clock)   

   ```bash
   ./scripts/table1_2.sh
   ```
3. **Generate Table 3:** ✅
   > **Memory requirement:** ≈ **7.4 GB** RAM (peak usage)   
   > **End-to-end runtime:** ≈ **80 min** (wall-clock)   

   ```bash
   ./scripts/table3.sh
   ```
4. **Generate Table 4:** ✅
   > **Memory requirement:** < **1 GB** RAM (peak usage)   
   > **End-to-end runtime:** < **1 sec** (wall-clock)   

   ```bash
   ./scripts/table4.sh <mode> [values_dir] ["activations"]

   # Quiet mode (recommended): runs everything silently and prints only the totals.
   ./scripts/table4.sh 1 y_yprime_examples gelu

   # Verbose mode: shows full logs from Python/Go/Rust plus the totals at the end.
   ./scripts/table4.sh 0 y_yprime_examples gelu
   ```   
5. **Generate Table 5:** ⏳ In progress
   > **Memory requirement:** ≈ **ZZ GB** RAM (peak usage)   
   > **End-to-end runtime:** ≈ **ZZ min** (wall-clock)   

   ```bash
   ./scripts/table5.sh
   ```
6. **Generate Table 6:** ✅
   > **Memory requirement:** ≈ **5 GB** RAM (peak usage)   
   > **End-to-end runtime:** ≈ **12 min** (wall-clock)   
   > **Reported metric:** Table 6 reports the **average over 10 runs** under identical settings.

   ```bash
   ./scripts/table6.sh
   ```
7. **Generate Table 7:** ⏳ In progress
   > **Memory requirement:** ≈ **ZZ GB** RAM (peak usage)   
   > **End-to-end runtime:** ≈ **ZZ min** (wall-clock)   

   ```bash
   ./scripts/table7.sh
   ```
   
## Artifact Documentation

## Acknowledgments
This project builds upon and integrates several existing open-source implementations:    
- **[gnark](https://github.com/Consensys/gnark)**: Used for PLONK proving and verification. 
- **[NFGen](https://github.com/Fannxy/NFGen)**: Incorporated in `src/piecewise_polynomial_approximation/NFGen/` to precompute piecewise-polynomial approximations of non-linear functions.  
- **[zk-Location](https://github.com/tumberger/zk-Location)**: Incorporated in `src/proof_generation/zk-Location` to prove IEEE-754–compliant **linear** operations.   
- **[caulk](https://github.com/caulk-crypto/caulk)**: Integrated into `src/proof_generation/caulk`, and we modified specific files to enable compatibility of its multi-lookup arguments with the ZIP framework.   

We gratefully acknowledge the authors and maintainers of these projects for making their work available to the community.

## Citing

If you use this repository or build upon our work, we would appreciate it if you cite our paper using the following BibTeX entry:

⏳ In progress

