# cisc662
Contains code for reproducing experiments done on Bridges-2 for simulating QAOA using CUDA-Q, along with the results.

## Usage
1. `make install`: Build the Singularity image (locally) - it contains the CUDA-Q binaries as the base image, with Nsight Compute installed on top of it for profiling.
2. `make run`: Invoke the code by exec'ing in the Singularity image. Currently assumes that you are inside an interactive mode on a job node.
3. `make analyze`: Generate the plots. Currently assumes that the results are in the results/ folder. 

Acknowledgement: This experiment builds upon the QAOA examples mentioned in the Nvidia CUDA-Quantum documentation: https://nvidia.github.io/cuda-quantum/0.8.0/using/examples/qaoa.html