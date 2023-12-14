#ifndef _MATRIX_OPS_H_
#define _MATRIX_OPS_H_

#include "counters_timers.h"
/* --------------------------------------
            LINEAR FUNCTIONS
            matrix operations
-------------------------------------- */
// matrix multiplication (a,c) = (a,b) . (b,c)
void mat_mul(FP* dst, FP* src_a, FP* src_b, int a_row, int a_col, int b_row, int b_col)
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

#ifndef USE_GEMMINI_LIB 
    FP tmp=0;
    if (a_col != b_row)
    {
        printf("[mat_mul]: Size not matched!\n"); exit(1);
    }
    for(int i=0; i<a_row; i++)
        for(int j=0; j<b_col; j++)
        {
            for(int k=0; k<a_col; k++)
                tmp += src_a[i*a_col + k] * src_b[j + k*b_col];

            dst[i*b_col + j] = tmp;
            tmp=0;
        }
#endif

#ifdef USE_GEMMINI_LIB 
  tiled_matmul_auto(a_row, b_col, a_col,
                    src_a, src_b, NULL, dst,
                    a_col, b_col, b_col, b_col,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                    false, false,
                    false, !FULL_BIAS_WIDTH,
                    0,
                    WMM);

#endif

#ifdef PRINT_PERF
size_t end = rdcycle();
acc_mat_mul += (end - start);
#endif

#ifdef PRINT_COUNTER
// non-functional: operation counter below
bp_mul += a_row * a_col * b_col;
bp_add += a_row * (a_col-1) * b_col;
// workload for index calculation
index_mul += a_row * b_col * (a_col*2 + 1);
index_add += a_row * b_col * (a_col*2 + 1);
#endif

}

// matrix multiplication with the second source matrix transposed
// then add bias
void mat_mul_b_T_add_bias(FP* dst, FP* src_a, FP* src_b, int a_row, int a_col, int b_row, int b_col, FP* bias)
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

#ifndef USE_GEMMINI_LIB 
    FP tmp=0;
    if (a_col != b_col) // source matrix B is to be transposed
    {
        printf("[mat_mul_b_T]: Size not matched!\n"); exit(1);
    }

    for(int i=0; i<a_row; i++)
        for(int j=0; j<b_row; j++)
        {
            for(int k=0; k<a_col; k++)
                tmp += src_a[i*a_col + k] * src_b[j*a_col + k];

            dst[i*b_row + j] = tmp;
            tmp = 0;
        }
    for(int i=0; i<a_row; i++)
        for(int j=0; j<b_row; j++)
            dst[i*b_row + j] += bias[j];
#endif

#ifdef USE_GEMMINI_LIB 
    tiled_matmul_auto(a_row, b_row, a_col,
                      src_a, src_b, bias, dst,
                      a_col, b_col, b_row, b_row,
                      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                      NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, true,
                      false, true,
                      false, !FULL_BIAS_WIDTH,
                      0,
                      WMM);
#endif



#ifdef PRINT_PERF
size_t end = rdcycle();
acc_mat_mul_b_T = acc_mat_mul_b_T + (end - start);
#endif

#ifdef PRINT_COUNTER
// non-functional: operation counter below
fp_mul += a_row * a_col * b_row;
fp_add += (a_row * (a_col-1) * b_row) + (a_row * b_row);
// workload for index calculation
index_mul += a_row * b_row * (a_col*2 + 2);
index_add += a_row * b_row * (a_col*2 + 2);
#endif

}

// matrix multiplication with the first source matrix to be transposed
// the resulting matrix will be averaged over Batch Size
void mat_mul_a_T_average(FP* dst, FP* src_a, FP* src_b, int a_row, int a_col, int b_row, int b_col, int n_samples)
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

#ifndef USE_GEMMINI_LIB 
    FP tmp = 0;
    if (a_row != b_row) // source matrix A is to be transposed
    {
        printf("[mat_mul_a_T_average]: Size not matched!\n"); exit(1);
    }
    for(int i=0; i<a_col; i++)
        for(int j=0; j<b_col; j++)
        {
            for(int k=0; k<a_row; k++)
                tmp += src_a[i + k*a_col] * src_b[j + k*b_col];

            dst[i*b_col + j] += tmp/n_samples;
            tmp = 0;
        }
#endif 

#ifdef USE_GEMMINI_LIB 
  tiled_matmul_auto(a_col, b_col, a_row,
                    src_a, src_b, NULL, dst,
                    a_col, b_col, b_col, b_col,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, 1/((FP)n_samples), 0, false,
                    true, false,
                    false, !FULL_BIAS_WIDTH,
                    0,
                    WMM);
#endif

#ifdef PRINT_PERF
size_t end = rdcycle();
acc_mat_mul_a_T = acc_mat_mul_a_T + (end - start);
#endif

#ifdef PRINT_COUNTER
// non-functional: operation counter below
bp_mul += (a_col * a_row * b_col);
bp_add += (a_col * (a_row-1) * b_col) + (a_col*b_col);
bp_div += a_col*b_col;
// workload for index calculation
index_mul += a_col * b_col * (a_row*2 + 1);
index_add += a_col * b_col * (a_row*2 + 1);
#endif

}

// convert a matrix to an array by sequeezing the first dimension
// i.e. (200,10) -> (1,10)
void mat2vec_avr_sequeeze(FP* dst, FP* src, int src_row, int src_col)
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

    FP tmp=0;
    for(int j=0; j<src_col; j++)
    {
        for(int i=0; i<src_row; i++)
            tmp += src[i*src_col + j];

        dst[j] += tmp/src_row;
        tmp=0;
    }

#ifdef PRINT_PERF
size_t end = rdcycle();
acc_mat2vec_avr_sequeeze += (end - start);
#endif

#ifdef PRINT_COUNTER
// non-functional: operation counter below
bp_add += src_col * (src_row - 1);
bp_div += src_col;
// workload for index calculation
index_mul += src_col * src_row;
index_add += src_col * src_row;
#endif

}

/* --------------------------------------
            LINEAR FUNCTIONS
            element-wise operations
-------------------------------------- */
// element-wise MUL/MAC (Multiply and Acummulate) on arrays/matrices of the same size 
void element_wise_mul(FP* mat_out, FP* mat_in_a, FP* mat_in_b, int row, int col, char which_pass[])
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

    int idx = 0;
    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
        {
            idx = i*col + j;
            mat_out[idx] = mat_in_a[idx] * mat_in_b[idx];
        }

#ifdef PRINT_PERF
size_t end = rdcycle();
acc_element_wise_mul += (end - start);
#endif


#ifdef PRINT_COUNTER
// non-functional: operation counter below
if(strcmp(which_pass, "forward_pass")==0)
    fp_mul += row*col;
else if(strcmp(which_pass, "backward_pass")==0)
    bp_mul += row*col;
else  {
    printf("[element_wise_mul] called by Unknown Pass, exit\n"); exit(1); }
index_mul += row*col;
index_add += row*col;
#endif


}

void element_wise_mac(FP* mat_out, FP* mat_in_a, FP* mat_in_b, int row, int col)
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

    int idx = 0;
    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
        {
            idx = i*col + j;
            mat_out[idx] += mat_in_a[idx] * mat_in_b[idx];
        }


#ifdef PRINT_PERF
size_t end = rdcycle();
acc_element_wise_mac += (end - start);
#endif

#ifdef PRINT_COUNTER
// non-functional: operation counter below
fp_mul += row*col;
fp_add += row*col;
// workload for index calculation
index_mul += row*col;
index_add += row*col;
#endif

}

void element_wise_sub(FP* dst, FP* src1, FP* src2, int row, int col)
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

    int idx = 0;
    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
        {
            idx = i*col + j;
            dst[idx] = src1[idx] - src2[idx];
        }


#ifdef PRINT_PERF
size_t end = rdcycle();
acc_element_wise_sub += (end - start);
#endif

#ifdef PRINT_COUNTER
// non-functional: operation counter below
bp_sub += row*col;
// workload for index calculation
index_mul += row*col;
index_add += row*col;
#endif

}


/* --------------------------------------
            NON-LINEAR FUNCTIONS
-------------------------------------- */
// float tanh function [tanhf] for the scalar is provided in  library <math.h> 
void tanhf_on_matrix(FP* mat_out, FP* mat_in, int row, int col)
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
            mat_out[i*col + j] = tanhf(mat_in[i*col + j]);

#ifdef PRINT_PERF
size_t end = rdcycle();
acc_tanhf_on_matrix += (end - start);
#endif

}

// sigmoid function on scalar
FP sigmoid(FP x) 
{
     FP result;
     result = 1 / (1 + exp(-x));
     return result;
}

// sigmoid function on matrix
void sigmoid_on_matrix(FP* mat_out, FP* mat_in, int row, int col)
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
            mat_out[i*col + j] = sigmoid(mat_in[i*col + j]);

#ifdef PRINT_PERF
size_t end = rdcycle();
acc_sigmoid_on_matrix += (end - start);
#endif

}

// row should be the batch size
// col should be the number of classifications
void softmax(FP* dst, FP* src, int row, int col)
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif
    /* Python code:
    def softmax(x):  
        e_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
        return e_x / e_x.sum(axis=1,keepdims= True) 
    */
    FP tmp[L2S];
    FP max;
    FP sum;

    for(int i=0; i<row; i++)
    {
        max = src[i*col + 0];
        for(int j=1; j<col; j++)
            if(src[i*col + j]>max)
                max = src[i*col + j];
        
        sum = 0;
        for(int j=0; j<col; j++)
        {
            tmp[j] = exp(src[i*col + j] - max);
            sum += tmp[j];
        }

        for(int j=0; j<col; j++)
            dst[i*col + j] = tmp[j]/sum;
    }

#ifdef PRINT_PERF
size_t end = rdcycle();
acc_softmax += (end - start);
#endif

}

void print_function_acc_time()
{
    //printf("[mat_mul]: \t%ld\n", acc_mat_mul);
    //printf("[mat_mul_b_T_add_bias]: \t%ld\n", acc_mat_mul_b_T);
    //printf("[mat_mul_a_T_average]: \t%ld\n", acc_mat_mul_a_T);

    //printf("[acc_mat2vec_avr_sequeeze]: \t%ld\n", acc_mat2vec_avr_sequeeze);
    //printf("[acc_element_wise_mul]: \t%ld\n", acc_element_wise_mul);
    //printf("[acc_element_wise_mac]: \t%ld\n", acc_element_wise_mac);
    //printf("[acc_element_wise_sub]: \t%ld\n", acc_element_wise_sub);
    //printf("[acc_tanhf_on_matrix]: \t%ld\n", acc_tanhf_on_matrix);
    //printf("[acc_sigmoid_on_matrix]: \t%ld\n", acc_sigmoid_on_matrix);
    //printf("[acc_softmax]: \t%ld\n", acc_softmax);

    //printf("[acc_load_sub_seq_to_xc]: \t%ld\n", acc_load_sub_seq_to_xc);
    //printf("[acc_relay_network_states]: \t%ld\n", acc_relay_network_states);
    //printf("[acc_fill_l1_h_into_xc]: \t%ld\n", acc_fill_l1_h_into_xc);
    //printf("[acc_find_ds]: \t%ld\n", acc_find_ds);
    //printf("[acc_find_d_l1_ifgo_input]: \t%ld\n", acc_find_d_l1_ifgo_input);
    //printf("[acc_update_d_l1_h]: \t%ld\n", acc_update_d_l1_h);
    //printf("[acc_SGD]: \t%ld\n", acc_SGD);
    //printf("[acc_FPTT_SGD]: \t%ld\n", acc_FPTT_SGD);

    acc_total = acc_mat_mul + acc_mat_mul_b_T + acc_mat_mul_a_T + acc_mat2vec_avr_sequeeze + acc_element_wise_mac + acc_element_wise_mul + acc_element_wise_sub + acc_tanhf_on_matrix + acc_sigmoid_on_matrix + acc_softmax + acc_load_sub_seq_to_xc + acc_relay_network_states + acc_fill_l1_h_into_xc + acc_find_ds + acc_find_d_l1_ifgo_input + acc_update_d_l1_h + acc_SGD + acc_FPTT_SGD ;
    printf("accumulated execution time of functions: \n");
    printf("MatMul (Type1-basic): \t%ld\n", (acc_mat_mul+acc_update_d_l1_h));
    printf("MatMul (Type2-bT): \t%ld\n", (acc_mat_mul_b_T));
    printf("MatMul (Type3-aT): \t%ld\n", (acc_mat_mul_a_T));
    printf("conversion Mat2Vec: \t%ld\n", (acc_mat2vec_avr_sequeeze));
    printf("Element-wise op between matrices (basic): \t%ld\n", (acc_element_wise_mac+acc_element_wise_mul+acc_element_wise_sub) );
    printf("Element-wise op between matrices (combined): \t%ld\n", (acc_find_d_l1_ifgo_input+acc_find_ds));
    #ifndef K_VAL
        printf("Element-wise op between matrices (SGD): \t%ld\n", (acc_SGD));
    #else
        printf("Element-wise op between matrices (SGD-FPTT): \t%ld\n", (acc_FPTT_SGD));
    #endif

    printf("Sigmoid: \t%ld\n", acc_sigmoid_on_matrix);
    printf("tanh: \t%ld\n", acc_tanhf_on_matrix);
    printf("softmax: \t%ld\n", acc_softmax);
    printf("data movement: \t%ld\n", (acc_load_sub_seq_to_xc+acc_relay_network_states+acc_fill_l1_h_into_xc));
    printf("[acc_total]: \t%ld\n", acc_total);
}

void print_operation_count()
{
    printf("-------- Operations done over forward pass  --------\n");
    printf("ADD/SUB:\t %.1f Mega (%d)\n", (float)(fp_add+fp_sub)/(1000000), (fp_add+fp_sub));
    printf("MUL/DIV:\t %.1f Mega (%d)\n", (float)(fp_mul+fp_div)/(1000000), (fp_mul+fp_div));

    printf("-------- Operations done over backward pass  --------\n");
    printf("ADD/SUB:\t %.1f Mega (%d)\n", (float)(bp_add+bp_sub)/(1000000), (bp_add+bp_sub));
    printf("MUL/DIV:\t %.1f Mega (%d)\n", (float)(bp_mul+bp_div)/(1000000), (bp_mul+bp_div));

    printf("-------- Operations done over optimizer (incl. R(t) calculation if FPTT enabled) --------\n");
    printf("ADD/SUB:\t %.1f Mega (%d)\n", (float)(fptt_add+fptt_sub)/(1000000), (fptt_add+fptt_sub));
    printf("MUL/DIV:\t %.1f Mega (%d)\n", (float)(fptt_mul+fptt_div)/(1000000), (fptt_mul+fptt_div));

    printf("-------- Operations done over index calculation --------\n");
    printf("ADD/SUB:\t %.1f Mega (%d)\n", (float)(index_add+index_sub)/(1000000), (index_add+index_sub));
    printf("MUL/DIV:\t %.1f Mega (%d)\n", (float)(index_mul+index_div)/(1000000), (index_mul+index_div));

}

#endif//_MATRIX_OPS_H_