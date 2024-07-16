# Acceleration of FPTT-based training of LSTM for S-MNIST dataset on RISC-V SOC

## Introduction
This workload defines accelerated training of LSTM for S-MNIST by FPTT algorithm on the RISC-V platform.

## Directory structure
**data**:       contains input 4 samples to be used (data already normalized) and the corresponding labels; network parameters (weights/biases) for different cases <br>
**lib**:    includes RISC-V libraries and Gemmini (a ROCC, RISC-V Custom Co-processor for MatMul) libraries <br>
**src**:        C source files etc <br>
**build**:  contains directories to build the binaries for x86/RISC-V, moreover the target RISC-V flow has two versions (baremetal or with OS). <br>


## Access to FPGA
|  IP    | 131.155.126.188 |
| ------ | ------ |
|username| vcu118-XXX         |
|password| vcu118-XXX         |


## Prerequisites 
 - All scripts are Linux-based <br>
 - To compile and run the workload on RISC-V platform, use the FireSim and FPGA board, [Chipyard repository](https://github.com/ucb-bar/chipyard) should be correctly installed and set in your directory to activate RISC-V binary toolset, functional models, etc.
