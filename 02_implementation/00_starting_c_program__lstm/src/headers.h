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
    #ifndef RVMULTICORE    
        #include "encoding.h"
    #else// multi-core riscv platform
        // #include <riscv-pk/encoding.h>
        #include "encoding.h"
        #include "util_multicore.h"
    #endif
#endif
#ifdef PLATFORM_X86
    #include <time.h>
    #include <stdint.h>
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
// Choice: data type fp32/bf16
// ----------------------------------------------------------------
#ifdef FP32
    typedef float FP;

    #define f2hex(dat) (*(uint32_t*)&dat) // print float(fp32) in hexdecimal format 
#endif

#ifdef BF16
    typedef uint16_t FP; // Brain Float 16
    #define ELEM_T_IS_LOWPREC_FLOAT
#endif

    #define u16_to_u32(dat) ((uint32_t)dat << 16)
    #define u32_to_fp32(dat) (*(float*)&dat)

    #define fp32_to_u32(dat) (*(uint32_t*)&dat)
    #define fp32_to_u16(dat) ( (fp32_to_u32(dat)>>16) + ((fp32_to_u32(dat)&0x8000)==0x8000) ) // rounding

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

#ifndef ELEM_T_IS_LOWPREC_FLOAT // if data type is float32
    #define LR 0.01
    #define ALPHA 0.1
#else // if data type is bf16
    #define LR      0x3c24  // 0.01 (approx.)
    #define ALPHA   0x3dcd  // 0.1  (approx.)
    #define P1      0x3f00  // 0.5
    #define P2      0x40a0  // 0.5/ALPHA (0.5/0.1)
    #define AVG     0x3e80  // 1/BS
#endif

#define M2O(array) ((FP*)array)

#endif//_HEADERS_H_ 
