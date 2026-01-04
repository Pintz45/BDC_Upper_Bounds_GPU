# BDC_Upper_Bounds_GPU

This repository contains GPU-accelerated implementations for computing **upper bounds on the capacity of the Binary Deletion Channel (BDC)**. The approach is based on [this]() work and extends a [previous implementation](https://github.com/ittai-rubinstein/BDC_Upper_Bounds) by Ittai Rubinstein and Roni Con, which itself is based on the **Blahut–Arimoto Algorithm (BAA)**.

The main contributions of this repository are:
- Several algorithmic optimizations of BAA for the BDC.
- Massive GPU parallelization using CUDA.
- The ability to study significantly more challenging parameter regimes than previously possible.

---

## Problem Setting

We study a binary deletion channel denoted by \( W_{n,k} \):

- An input binary string of length \( n \) is transmitted.
- The channel uniformly at random selects a subset of \( k \) positions.
- The selected bits are kept (in order), while the remaining bits are deleted.
- The output is a binary string of length \( k \).

The goal is to compute upper bounds on the channel capacity of \( W_{n,k} \) and, through some theoretical results (see [this]()), upper bounds for the classic binary deletion channel of deletion probability d.

---

## Repository Structure

### `backend/`

This folder contains the full implementation of the optimized Blahut–Arimoto algorithm for the channel \( W_{n,k} \).

#### `BAA.cu`
- Implements the optimized Blahut–Arimoto algorithm for the binary deletion channel.
- Includes GPU-specific optimizations and parallelization.
- For a more in-depth discussion of the implementation, follow the [this]() paper and read the comments in the code.

#### `generate_bit_transition_cache.cpp`
- Generates cache files named `cache_n_k`, stored in the `transition_counts/` folder.
- Each cache file encodes a matrix where the entry corresponding to strings \( (x, y) \) counts how many times the \( k \)-bit string `y` appears as a subsequence of the \( n \)-bit string `x`.
- These cache files are used to efficiently compute transition probabilities between input strings `x` and output strings `y`, as described in the underlying paper.

#### `utils.cc` and `cache_io.h`
- Provide utilities for:
  - Loading cache files from disk.
  - Managing cache data in memory during execution.

#### `test_bit_baa.cu`
- The main entry point for running the Blahut–Arimoto algorithm.
- User-configurable parameters:
  - `n`: input string length.
  - `k`: output string length.
  - `a`: tolerance threshold for convergence of the BAA estimate.
- The algorithm runs until the convergence criterion is met.
- Supports:
  - Starting from a uniform input distribution, or
  - Loading a previously saved input distribution from disk.
- The current input distribution is saved to disk every 100 iterations by default (this interval can be changed).

---

### `distributions/`

- Stores saved input distributions.
- These distributions can be reused to initialize future runs of the algorithm.

---

## Running the Blahut–Arimoto Algorithm

### 1. Configure Parameters
Edit `test_bit_baa.cu` and set:
- Desired values of `n`, `k`, and `a`.
- Whether to start from a uniform distribution or load one from file (`read_from_file` flag).

### 2. Install CUDA
Make sure the NVIDIA CUDA toolkit is installed:
```bash
sudo apt install nvidia-cuda-toolkit
```

### 3. Compile
Compile the CUDA code using `nvcc`:
```bash
nvcc -arch=? --extended-lambda -std=c++17 backend/*.cc backend/*.cu -o test_bit_baa
```
Replace ? with the appropriate architecture of your GPU. For example, RTX 5090: use sm_89.

### 3. Run
Run the executable:
```bash
./test_bit_baa
```
[Here](https://drive.google.com/drive/folders/17jwDnhdhlL6CTP9mE2dC7aoKNJ81NfeF) you can find the (almost) optimal input distributions for C_{29,k} and C_{31,k}. As explained before, If you want to run a simulation starting from one of this distributions, add the distribution file to the folder "distributions" and turn the "read_from_file" flag to value "true" when running the file "test_bit_baa.cu".



  


