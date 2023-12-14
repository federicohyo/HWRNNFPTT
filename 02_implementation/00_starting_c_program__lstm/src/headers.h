#ifndef _HEADERS_H_
#define _HEADERS_H_

// basic library
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// libraries from RISC-V & Gemmini
#ifdef PLATFORM_RV
#include "include/gemmini.h"
#include "encoding.h"
#endif
#ifdef PLATFORM_X86
#include <time.h>
#endif

// ----------------------------------------------------------------
// Choice: use macros to choose adaptions for different platform
// ----------------------------------------------------------------
// #define PLATFORM_X86 // x86 platform
// #define PLATFORM_RV  // RSIC-V platform, with a optional GEMMINI accelerator
// Has been defined in the Makefiles

#ifdef PLATFORM_RV
    #define FULL_BIAS_WIDTH 1 // required for Gemmini Mat-Mul 
    #define PRINT_PERF   // performance profiling of latency on RISC-V using internal cycle counter (on x86 we can simply use gprof)
    // #define USE_GEMMINI_LIB //define this option in Makefile
    #ifdef USE_GEMMINI_LIB
        // define the way of Matrix Multiplication
        #define WMM WS // on Gemmini, Weight Stationary mode
        // #define WMM CPU // on RISC-V CPU 
        // #define WMM OS // on Gemmini, Output Stationary mode
    #endif
#endif

// ----------------------------------------------------------------
// Choice: data type fp32/fp64/bf16
// ----------------------------------------------------------------
typedef float FP;
// typedef double FP;
// typedef unint16_t FP // Brain Float 16

// ----------------------------------------------------------------
// Choice: the debug information to be printed
// ----------------------------------------------------------------
#define PRINT_CONFIG // basic config, memory usage 
#define PRINT_COUNTER// counters of operations
#define PRINT_DEBUG  // print part of parameters

// ----------------------------------------------------------------
// define the network size
// ----------------------------------------------------------------
#define L2S (10)
#define L1S (128)
#define NUM_OF_P (784) // number of pixels in an input image

// ----------------------------------------------------------------
// define the way of training 
// ----------------------------------------------------------------
//#define MNIST_OFFLINE_BY_ROWS
//#define MNIST_OFFLINE_BY_PIXELS
#define MNIST_ONLINE_BY_PIXELS

#define BS 4    // batch size

#ifdef MNIST_OFFLINE_BY_ROWS 
    #define L0S (28)
    #define TS 28   // time steps, i.e. the length of sequences
    #define REGULARIZATION 0
#endif

#ifdef MNIST_OFFLINE_BY_PIXELS
    #define L0S (1)
    #define TS 784   // time steps, i.e. the length of sequences
    #define REGULARIZATION 0
#endif

#ifdef MNIST_ONLINE_BY_PIXELS // define K and TS in Makefile
    #define L0S (1)
    #define REGULARIZATION 1
#endif


// ----------------------------------------------------------------
// define the hyperparameters 
// ----------------------------------------------------------------
#define LR 0.01
#define ALPHA 0.1


#endif//_HEADERS_H_ 