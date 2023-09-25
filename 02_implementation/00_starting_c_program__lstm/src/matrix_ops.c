/*
    In the directory include/mat_and_arr_ops.h
    see data/function declaration 
*/
#include "matrix_ops.h"

/* --------------------------------------
            LINEAR FUNCTIONS
            matrix operations
-------------------------------------- */
void mat_mul(FP* dst, FP* src_a, FP* src_b, int a_row, int a_col, int b_row, int b_col)
{
    if (a_col != b_row)
    {
        printf("[mat_mul]: Size not matched!\n"); exit(1);
    }

    for(int i=0; i<a_row; i++)
        for(int j=0; j<b_col; j++)
            for(int k=0; k<a_col; k++)
                dst[i*b_col + j] += src_a[i*a_col + k] * src_b[j + k*b_col];
}

void mat_mul_b_T_add_bias(FP* dst, FP* src_a, FP* src_b, int a_row, int a_col, int b_row, int b_col, FP* bias)
{
    if (a_col != b_col) // source matrix B is to be transposed
    {
        printf("[mat_mul_b_T]: Size not matched!\n"); exit(1);
    }

    for(int i=0; i<a_row; i++)
        for(int j=0; j<b_row; j++)
            for(int k=0; k<a_col; k++)
                dst[i*b_row + j] += src_a[i*a_col + k] * src_b[j*a_col + k];


    for(int i=0; i<a_row; i++)
        for(int j=0; j<b_row; j++)
            dst[i*b_row + j] += bias[j];
}

void mat_mul_a_T_average(FP* dst, FP* src_a, FP* src_b, int a_row, int a_col, int b_row, int b_col, int n_samples)
{
    FP tmp = 0;

    if (a_row != b_row) // source matrix A is to be transposed
    {
        printf("[mat_mul]: Size not matched!\n"); exit(1);
    }

    for(int i=0; i<a_col; i++)
        for(int j=0; j<b_col; j++)
        {
            for(int k=0; k<a_row; k++)
                tmp += src_a[i + k*a_col] * src_b[j + k*b_col];

            dst[i*b_col + j] += tmp/n_samples;
            tmp = 0;
        }
}

void mat2vec_avr_sequeeze(FP* dst, FP* src, int src_row, int src_col)
{
    FP tmp=0;

    for(int j=0; j<src_col; j++)
    {
        for(int i=0; i<src_row; i++)
            tmp += src[i*src_col + j];

        dst[j] += tmp/src_row;
        tmp=0;
    }
}

/* --------------------------------------
            LINEAR FUNCTIONS
            element-wise operations
-------------------------------------- */
void element_wise_mul(FP* mat_out, FP* mat_in_a, FP* mat_in_b, int row, int col)
{
    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
            mat_out[i*col + j] = mat_in_a[i*col + j] * mat_in_b[i*col + j];
}

void element_wise_mac(FP* mat_out, FP* mat_in_a, FP* mat_in_b, int row, int col)
{
    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
            mat_out[i*col + j] += mat_in_a[i*col + j] * mat_in_b[i*col + j];
}

void element_wise_sub(FP* dst, FP* src1, FP* src2, int row, int col)
{
    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
            dst[i*col + j] = src1[i*col + j] - src2[i*col + j];
}


/* --------------------------------------
            NON-LINEAR FUNCTIONS
-------------------------------------- */
void tanhf_on_matrix(FP* mat_out, FP* mat_in, int row, int col)
{
    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
            mat_out[i*col + j] = tanhf(mat_in[i*col + j]);
}

FP sigmoid(FP x) 
{
     FP result;
     result = 1 / (1 + exp(-x));
     return result;
}

void sigmoid_on_matrix(FP* mat_out, FP* mat_in, int row, int col)
{
    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
            mat_out[i*col + j] = sigmoid(mat_in[i*col + j]);
}

void softmax(FP* dst, FP* src, int row, int col)
{
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
}

