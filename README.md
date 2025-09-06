# BDC_Upper_Bounds_GPU
This repository contains implementations for computing upper bounds on the capacity of the Binary Deletion Channel (BDC). The work extends a [previous approach]{https://github.com/ittai-rubinstein/BDC_Upper_Bounds} based on the Blahut–Arimoto algorithm by adding several optimizations that allow the study of more challenging parameter regimes.

Methods

To make the computation feasible for larger values of n and k, we employ new algorithmic and computational techniques:

- GPU acceleration for parallelized evaluations.

- Efficient subsequence and supersequence enumeration to compute transition probabilities efficiently.

- Codeword symmetry reduction to minimize the number of unique cases considered.
