# Zero-Knowledge AI Inference with High Precision

This repository contains the full implementation of **ZIP** paper (accepted to [**ACM CCS 2025**](https://www.sigsac.org/ccs/CCS2025/)). The manuscript will be available online shortly.

> **Warning**: This code is a research prototype intended for proof-of-concept purposes only and is not ready for production use.

## Code Structure

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

1. **Set up environment and dependencies:**:
   ```bash
   conda create -n zip python=3.9 -y
   conda activate zip
   pip install -r requirements.txt
   ```

## Artifact Documentation

## Acknowledgments

