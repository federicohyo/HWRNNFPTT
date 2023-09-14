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
#define L0S (28)
#define L1S (128)
#define L2S (10)

#define L1_W_SIZE (L1S * (L1S + L2S))
#define L1_B_SIZE L1S

#define L2_W_SIZE (L2S * (L2S + L1S))
#define L2_B_SIZE L2S


#define TS 28   // time steps, i.e. the length of subsequences
#define BS 5    // batch size




#endif//_HEADERS_H_ 