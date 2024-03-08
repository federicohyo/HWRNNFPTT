#ifndef _MATRIX_OPS_H_
#define _MATRIX_OPS_H_

#include "counters_timers.h"

#ifdef ELEM_T_IS_LOWPREC_FLOAT
#include "bf16_func.h"
#endif


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
                #ifndef ELEM_T_IS_LOWPREC_FLOAT
                    tmp += src_a[i*a_col + k] * src_b[j + k*b_col];
                #else
                    tmp = bf16_add(tmp, bf16_mul(src_a[i*a_col + k], src_b[j + k*b_col]));
                #endif

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
    { printf("[mat_mul_b_T]: Size not matched!\n"); exit(1); }

    for(int i=0; i<a_row; i++)
        for(int j=0; j<b_row; j++)
        {
            for(int k=0; k<a_col; k++)
            {
                #ifndef ELEM_T_IS_LOWPREC_FLOAT
                    tmp += src_a[i*a_col + k] * src_b[j*a_col + k];
                #else
                    tmp = bf16_add(tmp, bf16_mul(src_a[i*a_col + k], src_b[j*a_col + k]) );
                #endif
            }

            // dst[i*b_row + j] = tmp;
            // dst[i*b_row + j] += bias[j];
            #ifndef ELEM_T_IS_LOWPREC_FLOAT
                dst[i*b_row + j] = tmp + bias[j];
            #else
                dst[i*b_row + j] = bf16_add(tmp, bias[j]);
            #endif
            
            tmp = 0;
        }

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
void mat_mul_a_T_average(FP* dst, FP* src_a, FP* src_b, int a_row, int a_col, int b_row, int b_col, float n_samples)
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
            {
                #ifndef ELEM_T_IS_LOWPREC_FLOAT
                    tmp += src_a[i + k*a_col] * src_b[j + k*b_col];
                #else
                    tmp = bf16_add(tmp, bf16_mul(src_a[i + k*a_col], src_b[j + k*b_col]) );
                #endif
            }
            
            #ifndef ELEM_T_IS_LOWPREC_FLOAT
                dst[i*b_col + j] += tmp/n_samples;
            #else
                dst[i*b_col + j] = bf16_add(dst[i*b_col + j], bf16_div(tmp, fp32_to_u16(n_samples)) );
            #endif
            tmp = 0;
        }
#endif 

#ifdef USE_GEMMINI_LIB 


    tiled_matmul_auto(a_col, b_col, a_row,
                        src_a, src_b, NULL, dst,
                        a_col, b_col, b_col, b_col,
                        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                        NO_ACTIVATION, 1/(n_samples), 0, false,
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
#ifndef ELEM_T_IS_LOWPREC_FLOAT // float point 32-bit
    for(int j=0; j<src_col; j++)
    {
        for(int i=0; i<src_row; i++)
            tmp += src[i*src_col + j];

        dst[j] += tmp/src_row;
        tmp=0;
    }
#else // bf16 
    float rows = src_row;

    for(int j=0; j<src_col; j++)
    {
        for(int i=0; i<src_row; i++)
            tmp = bf16_add(tmp, src[i*src_col + j]);

        dst[j] = bf16_add(dst[j], bf16_div(tmp, fp32_to_u16(rows)));
        tmp=0;
    }
#endif



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

/* ------------------------------------
            ELEMENT-WISE MUL
-------------------------------------*/
#ifndef RVMULTICORE
void element_wise_mul(FP* mat_out, FP* mat_in_a, FP* mat_in_b, int row, int col, char which_pass[])
#else
void element_wise_mul(int cid, FP* mat_out, FP* mat_in_a, FP* mat_in_b, int row, int col, char which_pass[])
#endif
{

#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t start = rdcycle();
    #else
        size_t start;
        if(cid==0)
            start = rdcycle();
        // barrier(NCORES);
    #endif
#endif

#ifndef RVMULTICORE // x86 or single-core RV
    for(int i=0; i<row*col; i++)
#else // multi-core RV
    for(int i=cid; i<row*col; i+=NCORES)
#endif
    {
        #ifndef ELEM_T_IS_LOWPREC_FLOAT
            mat_out[i] = mat_in_a[i] * mat_in_b[i];
        #else
            mat_out[i] = bf16_mul(mat_in_a[i] , mat_in_b[i]);
        #endif
    }

#ifdef RVMULTICORE
    barrier(NCORES);
#endif

#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t end = rdcycle();
        acc_element_wise_mul += (end - start);
    #else
        size_t end;
        if(cid==0)
        {
            end = rdcycle();
            acc_element_wise_mul += (end - start);
        }
        // barrier(NCORES);
    #endif
#endif

#ifdef PRINT_COUNTER
    #ifdef RVMULTICORE
    if(cid==0)
    {
    #endif
        // non-functional: operation counter below
        if(strcmp(which_pass, "forward_pass")==0)
            fp_mul += row*col;
        else if(strcmp(which_pass, "backward_pass")==0)
            bp_mul += row*col;
        else  {
            printf("[element_wise_mul] called by Unknown Pass, exit\n"); exit(1); }
        index_mul += row*col;
        index_add += row*col;
    #ifdef RVMULTICORE
    }
    #endif
#endif
}

/* ------------------------------------
            ELEMENT-WISE MAC
-------------------------------------*/
#ifndef RVMULTICORE
void element_wise_mac(FP* mat_out, FP* mat_in_a, FP* mat_in_b, int row, int col)
#else
void element_wise_mac(int cid, FP* mat_out, FP* mat_in_a, FP* mat_in_b, int row, int col)
#endif
{
#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t start = rdcycle();
    #else
        size_t start;
        if(cid==0)
            start = rdcycle();
        // barrier(NCORES);
    #endif
#endif

#ifndef RVMULTICORE // x86 or single-core RV
    for(int i=0; i<row*col; i++)
#else // multi-core RV
    for(int i=cid; i<row*col; i+=NCORES)
#endif
    {
        #ifndef ELEM_T_IS_LOWPREC_FLOAT
        mat_out[i] += mat_in_a[i] * mat_in_b[i];
        #else
        mat_out[i] = bf16_add(mat_out[i], bf16_mul(mat_in_a[i], mat_in_b[i]) );
        #endif
    }

#ifdef RVMULTICORE
    barrier(NCORES);
#endif

#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t end = rdcycle();
        acc_element_wise_mac += (end - start);
    #else
        size_t end;
        if(cid==0)
        {
            end = rdcycle();
            acc_element_wise_mac += (end - start);
        }
        // barrier(NCORES);
    #endif
#endif

#ifdef PRINT_COUNTER
    #ifdef RVMULTICORE
    if(cid==0)
    {
    #endif
        // non-functional: operation counter below
        fp_mul += row*col;
        fp_add += row*col;
        // workload for index calculation
        index_mul += row*col;
        index_add += row*col;
    #ifdef RVMULTICORE
    }
    #endif
#endif
}

/* ------------------------------------
            ELEMENT-WISE SUB 
-------------------------------------*/
#ifndef RVMULTICORE
void element_wise_sub(FP* dst, FP* src1, FP* src2, int row, int col)
#else
void element_wise_sub(int cid, FP* dst, FP* src1, FP* src2, int row, int col)
#endif
{
#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t start = rdcycle();
    #else
        size_t start;
        if(cid==0)
            start = rdcycle();
        // barrier(NCORES);
    #endif
#endif

#ifndef RVMULTICORE // x86 or single-core RV
    for(int i=0; i<row*col; i++)
#else // multi-core RV
    for(int i=cid; i<row*col; i+=NCORES)
#endif
    {
        #ifndef ELEM_T_IS_LOWPREC_FLOAT
            dst[i] = src1[i] - src2[i];
        #else
            dst[i] = bf16_sub(src1[i], src2[i]);
        #endif
    }

#ifdef RVMULTICORE
    barrier(NCORES);
#endif

#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t end = rdcycle();
        acc_element_wise_sub += (end - start);
    #else
        size_t end;
        if(cid==0)
        {
            end = rdcycle();
            acc_element_wise_sub += (end - start);
        }
        // barrier(NCORES);
    #endif
#endif

#ifdef PRINT_COUNTER
    #ifdef RVMULTICORE
    if(cid==0)
    {
    #endif
        // non-functional: operation counter below
        bp_sub += row*col;
        // workload for index calculation
        index_mul += row*col;
        index_add += row*col;
    #ifdef RVMULTICORE
    }
    #endif
#endif
}


/* --------------------------------------
            NON-LINEAR FUNCTIONS
-------------------------------------- */
// float tanh function [tanhf] for the scalar is provided in  library <math.h> 

#ifndef RVMULTICORE
void tanhf_on_matrix(FP* mat_out, FP* mat_in, int row, int col)
#else
void tanhf_on_matrix(int cid, FP* mat_out, FP* mat_in, int row, int col)
#endif
{

#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t start = rdcycle();
    #else
        size_t start;
        if(cid==0)
            start = rdcycle();
        // barrier(NCORES);
    #endif
#endif


#ifndef RVMULTICORE // x86 or single-core RV
    for(int i=0; i<row*col; i++)
#else // multi-core RV
    for(int i=cid; i<row*col; i+=NCORES)
#endif
    {
        #ifndef ELEM_T_IS_LOWPREC_FLOAT
        mat_out[i] = tanhf(mat_in[i]);
        #else
        mat_out[i] = bf16_tanh(mat_in[i]);
        #endif
    }
#ifdef RVMULTICORE
    barrier(NCORES);
#endif


#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t end = rdcycle();
        acc_tanhf_on_matrix += (end - start);
    #else
        size_t end;
        if(cid==0)
        {
            end = rdcycle();
            acc_tanhf_on_matrix += (end - start);
        }
        // barrier(NCORES);
    #endif
#endif

}

// sigmoid function on scalar
FP sigmoid(FP x) 
{
    //  FP result;
    //  result = 1 / (1 + exp(-x));
    //  return result;
    return 1.0f/(1 + exp(-x));
}

// sigmoid function on matrix
#ifndef RVMULTICORE
void sigmoid_on_matrix(FP* mat_out, FP* mat_in, int row, int col)
#else
void sigmoid_on_matrix(int cid, FP* mat_out, FP* mat_in, int row, int col)
#endif
{

#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t start = rdcycle();
    #else
        size_t start;
        if(cid==0)
            start = rdcycle();
        // barrier(NCORES);
    #endif
#endif


#ifndef RVMULTICORE // x86 or single-core RV
    for(int i=0; i<row*col; i++)
#else // multi-core RV
    for(int i=cid; i<row*col; i+=NCORES)
#endif
    {
        #ifndef ELEM_T_IS_LOWPREC_FLOAT
        mat_out[i] = sigmoid(mat_in[i]);
        #else
        mat_out[i] = bf16_sigmoid(mat_in[i]);
        #endif
    }
#ifdef RVMULTICORE
    barrier(NCORES);
#endif


#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t end = rdcycle();
        acc_sigmoid_on_matrix += (end - start);
    #else
        size_t end;
        if(cid==0)
        {
            end = rdcycle();
            acc_sigmoid_on_matrix += (end - start);
        }
        // barrier(NCORES);
    #endif
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

#ifndef ELEM_T_IS_LOWPREC_FLOAT
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
#else
    for(int i=0; i<row; i++)
    {
        max = src[i*col + 0];
        for(int j=1; j<col; j++)
            if(bf16_greater(src[i*col + j], max) )
                max = src[i*col + j];
        
        sum = 0;
        for(int j=0; j<col; j++)
        {
            tmp[j] = bf16_exp(bf16_sub(src[i*col + j], max));
            sum = bf16_add(sum, tmp[j]);
        }

        for(int j=0; j<col; j++)
            dst[i*col + j] = bf16_div(tmp[j], sum);
    }
#endif

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