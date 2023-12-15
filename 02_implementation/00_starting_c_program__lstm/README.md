# Training script of LSTM for MNIST dataset

## Introduction
LSTM training of S-MNIST dataset, for x86/RISC-V platform. RISC-V platform has an optional MatMul Accelerator, namely Gemmini. 

## Directory structure
**data**:       contains input 5 samples to be used (data already normalized), network parameters (weights/biases) for different cases <br>
**lib**:    Gemmini Libraries having handy functions to manipulate Gemmini Accelerator, e.g. matmul\_tiled\_auto <br>
**src**:        C source files etc <br>
**build**:  build the application on x86/RISC-V, moreover RISC-V can be baremetal or with OS. <br>


## Cases included
 - 1. MNIST streaming by rows, regular (offline) 
 - 2. MNIST streaming by pixels, regular (offline)
 - 3. MNIST streaming by pixels, online formulation, with optional regularization (in other words, **FPTT**)

***offline*** means that backward path is not performed until the entire input sample is forwarded into the network 
***online*** means interleaved forward and backward path of one input sample, e.g., a sample has 100 steps, we can do forward pass for 20 steps, then do the backward pass and weight update. Repeating this process 4 more times means the network performs the online formulation of training, by breaking the original input into 5 even pieces.

