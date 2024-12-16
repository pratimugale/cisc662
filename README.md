# cisc662
Contains code for reproducing experiments done on Bridges-2 and Delta for simulating QAOA using CUDA-Q, along with the results.

## Blog Link
A blog for the project is written [here](blog.md).

## Usage
1. `make install`: Build the Singularity image (locally) - it contains the CUDA-Q binaries as the base image, with Nsight Compute installed on top of it for profiling.
2. `make run`: Invoke the code by exec'ing in the Singularity image. Currently assumes that you are inside an interactive mode on a job node.
3. `make analyze`: Generate the plots. Currently assumes that the results are in the results/ folder.

## Acknowledgement
This work used Bridges-2 at Pittsburgh Supercomputing Center through allocation from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

This research used the Delta advanced computing and data resource which is supported by the National Science Foundation (award OAC 2005572) and the State of Illinois. Delta is a joint effort of the University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications.

This experiment builds upon the QAOA examples mentioned in the Nvidia CUDA-Quantum documentation: https://nvidia.github.io/cuda-quantum/0.8.0/using/examples/qaoa.html

Acknowledgement: This experiment builds upon the QAOA examples mentioned in the Nvidia CUDA-Quantum documentation: https://nvidia.github.io/cuda-quantum/0.8.0/using/examples/qaoa.html
