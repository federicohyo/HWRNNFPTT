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
#ifdef MNIST_OFFLINE_BY_ROWS 
    #define L0S (28)
    #define TS 28   // time steps, i.e. the length of sequences
#endif

#ifdef MNIST_OFFLINE_BY_PIXELS
    #define L0S (1)
    #define TS 784   // time steps, i.e. the length of sequences
#endif

#ifdef MNIST_ONLINE_BY_PIXELS
    #define L0S (1)
    #undef K
    #define K 28
    #define TS 28   // time steps, i.e. the length of subsequences, NUM_OF_P/K
#endif



#define BS 5    // batch size
#define NUM_OF_P 784 // number of pixels in an input image

// --------------------------------
// define the hyperparameters 
// --------------------------------
#define LR 0.01
#define ALPHA 0.1

#endif//_HEADERS_H_ 