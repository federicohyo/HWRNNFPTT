/*
    In the directory include/matrix_ops.h
    see data/function declaration 
*/
#include "matrix_ops.h"



int fp_add = 0;
int fp_sub = 0;
int fp_mul = 0;
int fp_div = 0;

int bp_add = 0;
int bp_sub = 0;
int bp_mul = 0;
int bp_div = 0;

int fptt_add = 0;
int fptt_sub = 0;
int fptt_mul = 0;
int fptt_div = 0;

int index_add = 0;
int index_sub = 0;
int index_mul = 0;
int index_div = 0;



/* --------------------------------------
            LINEAR FUNCTIONS
            matrix operations
-------------------------------------- */
void mat_mul(FP* dst, FP* src_a, FP* src_b, int a_row, int a_col, int b_row, int b_col)
{
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


    // non-functional: operation counter below
    bp_mul += a_row * a_col * b_col;
    bp_add += a_row * (a_col-1) * b_col;

    // workload for index calculation
    index_mul += a_row * b_col * (a_col*2 + 1);
    index_add += a_row * b_col * (a_col*2 + 1);
}

void mat_mul_b_T_add_bias(FP* dst, FP* src_a, FP* src_b, int a_row, int a_col, int b_row, int b_col, FP* bias)
{
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


    // non-functional: operation counter below
    fp_mul += a_row * a_col * b_row;
    fp_add += (a_row * (a_col-1) * b_row) + (a_row * b_row);

    // workload for index calculation
    index_mul += a_row * b_row * (a_col*2 + 2);
    index_add += a_row * b_row * (a_col*2 + 2);
}

void mat_mul_a_T_average(FP* dst, FP* src_a, FP* src_b, int a_row, int a_col, int b_row, int b_col, int n_samples)
{
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
    

    // non-functional: operation counter below
    bp_mul += (a_col * a_row * b_col);
    bp_add += (a_col * (a_row-1) * b_col) + (a_col*b_col);
    bp_div += a_col*b_col;

    // workload for index calculation
    index_mul += a_col * b_col * (a_row*2 + 1);
    index_add += a_col * b_col * (a_row*2 + 1);
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


    // non-functional: operation counter below
    bp_add += src_col * (src_row - 1);
    bp_div += src_col;

    // workload for index calculation
    index_mul += src_col * src_row;
    index_add += src_col * src_row;
}

/* --------------------------------------
            LINEAR FUNCTIONS
            element-wise operations
-------------------------------------- */
void element_wise_mul(FP* mat_out, FP* mat_in_a, FP* mat_in_b, int row, int col, char which_pass[])
{
    int idx = 0;
    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
        {
            idx = i*col + j;
            mat_out[idx] = mat_in_a[idx] * mat_in_b[idx];
        }

    // non-functional: operation counter below
    if(strcmp(which_pass, "forward_pass")==0)
    {
        fp_mul += row*col;
        // index calculation
        index_mul += row*col;
        index_add += row*col;
    }
    else if(strcmp(which_pass, "backward_pass")==0)
    {
        bp_mul += row*col;
        // index calculation
        index_mul += row*col;
        index_add += row*col;
    }
    else   
    {
        printf("[element_wise_mul] called by Unknown Pass, exit\n"); exit(1);
    }

}

void element_wise_mac(FP* mat_out, FP* mat_in_a, FP* mat_in_b, int row, int col)
{
    int idx = 0;

    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
        {
            idx = i*col + j;
            mat_out[idx] += mat_in_a[idx] * mat_in_b[idx];
        }


    // non-functional: operation counter below
    fp_mul += row*col;
    fp_add += row*col;

    // workload for index calculation
    index_mul += row*col;
    index_add += row*col;
}

void element_wise_sub(FP* dst, FP* src1, FP* src2, int row, int col)
{
    int idx = 0;

    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
        {
            idx = i*col + j;
            dst[idx] = src1[idx] - src2[idx];
        }


    // non-functional: operation counter below
    bp_sub += row*col;

    // workload for index calculation
    index_mul += row*col;
    index_add += row*col;
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