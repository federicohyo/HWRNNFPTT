# Training script of LSTM for MNIST dataset

## Introduction
This application targets on some tiny devices like bare-metal machines, i.e. without the Operating System. So things like ***malloc*** are not used.

## Directory structure
**data**:       contains input 5 samples to be used (data already normalized), network parameters (weights/biases) for different cases <br>
**include**:    header files, having declarations and brief descriptions of functions and data structures <br>
**src**:        source files, having function definitions etc <br>

## The dependency of files
    matrix_ops.c <- lstm.c <- caselist.c <- main.c

## Cases included
 - 1. MNIST streaming by rows, offline, with optional regularization 
 - 2. MNIST streaming by pixels, offline, with optional regularization 
 - 3. MNIST streaming by pixels, online, with optional regularization (in other words, **FPTT**)

***offline*** means that backward path is not performed until the entire input sample is processed the network
***online*** means interleaved forward and backward path.


## Get started
- 1. View `./include/headers.h` to know macros.
- 2. in Makefile, change **-D** to choose cases
- 3. if case 3 is to be run, specify **K** value in `./include/headers.h`
- 4. Linux Command Line: `make; ./run `

## Future Improvements
This application is just acceptable now, as it is manually compared Python model and the difference is smaller than 1e-6. A scoreboard will be added later.
Also only one **K**  value is tested so far.