# Zero-Knowledge AI Inference with High Precision

This repository contains the full implementation of **ZIP** paper (accepted to [**ACM CCS 2025**](https://www.sigsac.org/ccs/CCS2025/)).       
The manuscript will be available online shortly.

> **Warning**: This code is a research prototype intended for proof-of-concept purposes only and is not ready for production use.

## Code Structure

⏳ In progress

## Prerequisites
Before running the scripts, ensure that you have the following installed:

- **Python**: 3.9+
- **Rust**: 1.72.0-nightly
- **Cargo**: 1.72.0-nightly (Rust package management)
- **Go**: go1.24.4

> Note: The package commands below use Debian/Ubuntu (```apt-get```). For Fedora/RHEL, replace with ```dnf/yum```.

## Installation

### Installing Python:

   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-pip
   ```

### Installing Rust and Cargo

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   . "$HOME/.cargo/env"
   rustup toolchain install nightly-2023-06-26
   rustup default nightly-2023-06-26  # Set this version globally

   # Verify
   rustc --version  # Verify the Rust compiler version
   cargo --version  # Verify the Cargo version
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
   python -m pip install -U torch numpy scipy pandas scikit-learn sympy
   chmod +x precompute.sh fig1.sh table1_2.sh table3.sh table4.sh table5.sh table6.sh table7.sh
   ```

2. **Precompute piecewise polynomial approximations for the target non-linear functions**
   ```bash
   ./precompute.sh
   ```
### Reproduce the Paper Results

- **Reproducibility note:**  
   - To match the paper’s numbers, use hardware comparable to our reference machine:
      - **48 CPU cores** (Intel® Xeon® Platinum 8360Y, 2.40 GHz) and **1 TB RAM**.  
   - The underlying **PLONK** prover uses **all available CPU cores** by default, so runtimes vary with core count.  
   - When running, ensure the machine is otherwise idle to obtain consistent timings.

1. **Generate Figure 1:** ✅
   > **Memory requirement:** **51 GB** RAM (peak usage)   
   > **End-to-end runtime:** **35 min** (wall-clock)   
   ```bash
   ./fig1.sh
   ```
2. **Generate Table 1 & 2:** ✅
   > **Memory requirement:** **250 GB** RAM (peak usage)   
   > **End-to-end runtime:** **80 min** (wall-clock)   

   ```bash
   ./table1_2.sh
   ```
3. **Generate Table 3:** ⏳ In progress
   > **Memory requirement:** **ZZ GB** RAM (peak usage)   
   > **End-to-end runtime:** **ZZ min** (wall-clock)   

   ```bash
   ./table3.sh
   ```
4. **Generate Table 4:** ⏳ In progress
   > **Memory requirement:** **ZZ GB** RAM (peak usage)   
   > **End-to-end runtime:** **ZZ min** (wall-clock)   

   ```bash
   ./table4.sh
   ```
5. **Generate Table 5:** ⏳ In progress
   > **Memory requirement:** **ZZ GB** RAM (peak usage)   
   > **End-to-end runtime:** **ZZ min** (wall-clock)   

   ```bash
   ./table5.sh
   ```
6. **Generate Table 6:** ⏳ In progress
   > **Memory requirement:** **ZZ GB** RAM (peak usage)   
   > **End-to-end runtime:** **ZZ min** (wall-clock)   

   ```bash
   ./table6.sh
   ```
7. **Generate Table 7:** ⏳ In progress
   > **Memory requirement:** **ZZ GB** RAM (peak usage)   
   > **End-to-end runtime:** **ZZ min** (wall-clock)   

   ```bash
   ./table7.sh
   ```
   
## Artifact Documentation

## Acknowledgments
This project uses [gnark](https://github.com/Consensys/gnark) for PLONK proving/verification and [NFGen](https://github.com/Fannxy/NFGen) (in `src/piecewise_polynomial_approximation/NFGen/`) to precompute piecewise-polynomial approximations of non-linear functions.

## Citing

If you use this repository or build upon our work, we would appreciate it if you cite our paper using the following BibTeX entry:

⏳ In progress

