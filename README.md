# BDC_Upper_Bounds_GPU
This repository contains implementations for computing upper bounds on the capacity of the Binary Deletion Channel (BDC) for a fixed input length n and output length k, based on [this]() work. The work extends a [previous approach](https://github.com/ittai-rubinstein/BDC_Upper_Bounds), by Ittai Rubinstein and Roni Con, based on the Blahut–Arimoto algorithm by adding several optimizations that allow the study of more challenging parameter regimes.

To make the computation feasible for larger values of n and k, we employ new algorithmic and computational techniques:

- GPU acceleration

- Efficient subsequence and supersequence enumeration to compute BAA updates more efficiently.

- Using codeword symmetry to reduce the number of cases considered.



[Here](https://drive.google.com/drive/folders/17jwDnhdhlL6CTP9mE2dC7aoKNJ81NfeF) you can find the (almost) optimal input distributions for C_{29,k} obtained for a tolerance of a = 0.005. If you want to run a simulation starting from one of this distributions, add the distribution file to the folder "distributions" and turn the "read_from_file" flag to value "true" when running the file "test_bit_baa.cu"


  


