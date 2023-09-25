#ifndef _MATRIX_OPS_H_
#define _MATRIX_OPS_H_

#include "headers.h"


/* --------------------------------------
            LINEAR FUNCTIONS
            matrix operations
-------------------------------------- */
// matrix multiplication (a,c) = (a,b) . (b,c)
void mat_mul(FP* dst, FP* src_a, FP* src_b, int a_row, int a_col, int b_row, int b_col);

// matrix multiplication with the second source matrix transposed
// then add bias
void mat_mul_b_T_add_bias(FP* dst, FP* src_a, FP* src_b, int a_row, int a_col, int b_row, int b_col, FP* bias);

// matrix multiplication with the first source matrix to be transposed
// the resulting matrix will be averaged over Batch Size
void mat_mul_a_T_average(FP* dst, FP* src_a, FP* src_b, int a_row, int a_col, int b_row, int b_col, int n_samples);

// convert a matrix to an array by sequeezing the first dimension
// i.e. (200,10) -> (1,10)
void mat2vec_avr_sequeeze(FP* dst, FP* src, int src_row, int src_col);


/* --------------------------------------
            LINEAR FUNCTIONS
            element-wise operations
-------------------------------------- */
// element-wise MUL/MAC (Multiply and Acummulate) on arrays/matrices of the same size 
void element_wise_mul(FP* mat_out, FP* mat_in_a, FP* mat_in_b, int row, int col);
void element_wise_mac(FP* mat_out, FP* mat_in_a, FP* mat_in_b, int row, int col);
void element_wise_sub(FP* dst, FP* src1, FP* src2, int row, int col);


/* --------------------------------------
            NON-LINEAR FUNCTIONS
-------------------------------------- */
// float tanh function [tanhf] for the scalar is provided in  library <math.h> 
void tanhf_on_matrix(FP* mat_out, FP* mat_in, int row, int col);

// sigmoid function on scalar
FP sigmoid(FP x);
// sigmoid function on matrix
void sigmoid_on_matrix(FP* mat_out, FP* mat_in, int row, int col);

// row should be the batch size
// col should be the number of classifications
void softmax(FP* dst, FP* src, int row, int col);


#endif//_MATRIX_OPS_H_