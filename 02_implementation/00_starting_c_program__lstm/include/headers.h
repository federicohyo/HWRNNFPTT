#ifndef _HEADERS_H_
#define _HEADERS_H_

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// --------------------------------
// choose the data type fp32/fp64
// --------------------------------
typedef float FP;
// typedef double FP;


// --------------------------------
// define the network size
// --------------------------------
#define L2S (10)
#define L1S (128)

#define K 0 // default FPTT is disabled
#ifdef BY_ROW_BPTT
    #define L0S (28)
    #define TS 28   // time steps, i.e. the length of sequences
#endif

#ifdef BY_PIXEL_BPTT
    #define L0S (1)
    #define TS 784   // time steps, i.e. the length of sequences
#endif

#ifdef BY_PIXEL_FPTT_K_28
    #define L0S (1)
    #undef K
    #define K 28
    #define TS 28   // time steps, i.e. the length of subsequences, T/K
#endif

#ifdef BY_PIXEL_FPTT_K_56
    #define L0S (1)
    #undef K
    #define K 56
    #define TS 14   // time steps, i.e. the length of subsequences, T/K
#endif



#define BS 5    // batch size


// --------------------------------
// define the hyperparameters 
// --------------------------------
#define LR 0.01
#define ALPHA 0.1

#endif//_HEADERS_H_ 