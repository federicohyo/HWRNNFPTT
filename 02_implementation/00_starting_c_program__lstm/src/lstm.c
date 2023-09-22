/*
    In the directory include/lstm.h
    see explanations of data/function definitions
*/
#include "lstm.h"

FP max_ini_v = sqrt(1/(FP)(L1S));
FP min_ini_v = -sqrt(1/(FP)(L1S)); 


// to be randomized
FP l1_wi[L1S][(L1S + L0S)];
FP l1_wf[L1S][(L1S + L0S)];
FP l1_wg[L1S][(L1S + L0S)];
FP l1_wo[L1S][(L1S + L0S)];
FP l1_bi[L1S];
FP l1_bf[L1S];
FP l1_bg[L1S];
FP l1_bo[L1S];
FP l2_w[L2S][L1S];
FP l2_b[L2S];


// to be randomized
FP l1_wi_rm[L1S][(L1S + L0S)];
FP l1_wf_rm[L1S][(L1S + L0S)];
FP l1_wg_rm[L1S][(L1S + L0S)];
FP l1_wo_rm[L1S][(L1S + L0S)];
FP l1_bi_rm[L1S];
FP l1_bf_rm[L1S];
FP l1_bg_rm[L1S];
FP l1_bo_rm[L1S];
FP l2_w_rm[L2S][L1S];
FP l2_b_rm[L2S];


FP l1_wi_lbd[L1S][(L1S + L0S)] = {0};
FP l1_wf_lbd[L1S][(L1S + L0S)] = {0};
FP l1_wg_lbd[L1S][(L1S + L0S)] = {0};
FP l1_wo_lbd[L1S][(L1S + L0S)] = {0};
FP l1_bi_lbd[L1S] = {0};
FP l1_bf_lbd[L1S] = {0};
FP l1_bg_lbd[L1S] = {0};
FP l1_bo_lbd[L1S] = {0};
FP l2_w_lbd[L2S][L1S] = {0};
FP l2_b_lbd[L2S] = {0};


FP l1_wi_grad[L1S][(L1S + L0S)] = {0};
FP l1_wf_grad[L1S][(L1S + L0S)] = {0};
FP l1_wg_grad[L1S][(L1S + L0S)] = {0};
FP l1_wo_grad[L1S][(L1S + L0S)] = {0};
FP l1_bi_grad[L1S] = {0};
FP l1_bf_grad[L1S] = {0};
FP l1_bg_grad[L1S] = {0};
FP l1_bo_grad[L1S] = {0};
FP l2_w_grad[L2S][L1S] = {0};
FP l2_b_grad[L2S] = {0};


FP xc[TS+1][BS][(L0S+L1S)] = {0};
FP label[BS][L2S] = {0};

FP l1_i_input[TS+1][BS][L1S] = {0};
FP l1_f_input[TS+1][BS][L1S] = {0};
FP l1_g_input[TS+1][BS][L1S] = {0};
FP l1_o_input[TS+1][BS][L1S] = {0};

FP l1_i[TS+1][BS][L1S] = {0};
FP l1_f[TS+1][BS][L1S] = {0};
FP l1_g[TS+1][BS][L1S] = {0};
FP l1_o[TS+1][BS][L1S] = {0};
FP l1_s[TS+1][BS][L1S] = {0};
FP l1_s_tanh[TS+1][BS][L1S] = {0};
FP l1_h[TS+1][BS][L1S] = {0};
FP l2_h[TS+1][BS][L2S] = {0};
FP l2_o[TS+1][BS][L2S] = {0};


FP d_l2_h[BS][L2S] = {0};
FP d_l1_h[BS][L1S] = {0};
FP d_l1_s[BS][L1S] = {0};
FP d_l1_i[BS][L1S] = {0};
FP d_l1_f[BS][L1S] = {0};
FP d_l1_g[BS][L1S] = {0};
FP d_l1_o[BS][L1S] = {0};
FP di_input[BS][L1S] = {0};
FP df_input[BS][L1S] = {0};
FP dg_input[BS][L1S] = {0};
FP do_input[BS][L1S] = {0};


void print_static_memory_usage()
{
    int sum=0;
    int mem_params=0;
    int mem_states=0;
    int mem_grad=0;

    mem_params = 4* (4*(sizeof(l1_wi) + sizeof(l1_bi)) + sizeof(l2_w) + sizeof(l2_b));
    mem_states = sizeof(xc) + sizeof(l1_i_input)*4 + sizeof(l1_i)*7 + sizeof(l2_h)*2;
    mem_grad = sizeof(d_l2_h) + sizeof(d_l1_h)*10;
    sum = mem_params + mem_states +mem_grad;

    printf(" \n---------- Memory Usage (static) ----------\n");
    printf("Network parameters:\t %.2f MB (%d Bytes)\n", ((float)mem_params/1024/1024), mem_params);
    printf("Network states:\t\t %.2f MB (%d Bytes) \n", ((float)mem_states/1024/1024), mem_states);
    printf("Intermediate gradients:\t %.2f MB (%d Bytes) \n", ((float)mem_grad/1024/1024), mem_grad);
    printf("Total:\t\t\t %.2f MB (%d Bytes) \n", ((float)sum/1024/1024), sum);
    printf(" -------------------------------------------\n");
}


void load_param_and_rm(FP* param, FP* rm, int row, int col, char file_dir[], char file_name[])
{
    char file[50];  
    strcpy(file, file_dir);
    strcat(file, file_name);

    FILE* ptr = fopen(file, "r");
    
    if (ptr == NULL) 
    {
        printf("[load_param_and_rm]: No such file: %s.\n", file);    exit(1);
    }

    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
        {
            fscanf(ptr, "%f", &param[i*col +j]);
            rm[i*col +j] = param[i*col +j];
            // fscanf(ptr, "%f", &rm[i]);
        }

    fclose(ptr);
}

void load_all_param_and_rm(char file_dir[])
{

    load_param_and_rm(&l1_wi[0][0], &l1_wi_rm[0][0], L1S, (L1S + L0S), file_dir, "param_l1_wi.txt");
    load_param_and_rm(&l1_wf[0][0], &l1_wf_rm[0][0], L1S, (L1S + L0S), file_dir, "param_l1_wf.txt");
    load_param_and_rm(&l1_wg[0][0], &l1_wg_rm[0][0], L1S, (L1S + L0S), file_dir, "param_l1_wg.txt");
    load_param_and_rm(&l1_wo[0][0], &l1_wo_rm[0][0], L1S, (L1S + L0S), file_dir, "param_l1_wo.txt");

    load_param_and_rm(&l1_bi[0], &l1_bi_rm[0], 1, L1S, file_dir, "param_l1_bi.txt");
    load_param_and_rm(&l1_bf[0], &l1_bf_rm[0], 1, L1S, file_dir, "param_l1_bf.txt");
    load_param_and_rm(&l1_bg[0], &l1_bg_rm[0], 1, L1S, file_dir, "param_l1_bg.txt");
    load_param_and_rm(&l1_bo[0], &l1_bo_rm[0], 1, L1S, file_dir, "param_l1_bo.txt");

    load_param_and_rm(&l2_w[0][0], &l2_w_rm[0][0], L2S, L1S, file_dir, "param_l2_w.txt");
    load_param_and_rm(&l2_b[0], &l2_b_rm[0], 1, L2S, file_dir, "param_l2_b.txt");
}


//  the running mean and the parameter are initialized as the same values
void initialize_param_and_rm(FP* input_param, FP* input_rm, int row, int col, FP min_val, FP max_val)
{
    for (int i = 0; i < row; i++) 
        for (int j = 0; j < col; j++) 
        {
            // Scale the random integer to fit within the desired range
            FP scaledRand = ((FP)rand() / (FP)(RAND_MAX) ) * (max_val - min_val) + min_val;

            // Assign the scaled random value to the array element
            input_param[i*col + j] = scaledRand;
            input_rm[i*col + j] = scaledRand;
        }
} 

void initialize_all_param_and_rm()
{
    // srand(time(NULL));
    srand(0); // specify the seed so that the reuslt can be re-produced

    initialize_param_and_rm( (FP*)l1_wi, (FP*)l1_wi_rm, L1S, (L1S + L0S), max_ini_v, min_ini_v);
    initialize_param_and_rm( (FP*)l1_wf, (FP*)l1_wf_rm, L1S, (L1S + L0S), max_ini_v, min_ini_v);
    initialize_param_and_rm( (FP*)l1_wg, (FP*)l1_wg_rm, L1S, (L1S + L0S), max_ini_v, min_ini_v);
    initialize_param_and_rm( (FP*)l1_wo, (FP*)l1_wo_rm, L1S, (L1S + L0S), max_ini_v, min_ini_v);

    initialize_param_and_rm( l1_bi, l1_bi_rm, 1, L1S, max_ini_v, min_ini_v);
    initialize_param_and_rm( l1_bf, l1_bf_rm, 1, L1S, max_ini_v, min_ini_v);
    initialize_param_and_rm( l1_bg, l1_bg_rm, 1, L1S, max_ini_v, min_ini_v);
    initialize_param_and_rm( l1_bo, l1_bo_rm, 1, L1S, max_ini_v, min_ini_v);

    initialize_param_and_rm( (FP*)l2_w, (FP*)l2_w_rm, L2S, L1S, max_ini_v, min_ini_v);
    initialize_param_and_rm( l2_b, l2_b_rm, 1, L2S, max_ini_v, min_ini_v);
}


void load_input_samples_to_xc(char file_name[])
{
    int count=0;
    FILE* ptr = fopen(file_name, "r");
    
    if (ptr == NULL) 
    {
        printf("[load_input_samples_to_xc]: No such file: %s.\n", file_name);    exit(1);
    }

    for(int j=0; j<BS; j++)
        for(int i=1; i<TS+1; i++)
            for(int k=0; k<L0S; k++) // the size of this dim is (L0S+L1S) though
            {
                fscanf(ptr, "%f", &xc[i][j][k] );
                count++;
            }
    
    // printf("%d\n", count);
    fclose(ptr);
}


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


void fill_l1_h_into_xc(int t)
{
    for(int i=0; i<BS; i++)
        for(int j=L0S; j<(L0S+L1S); j++)
            xc[t][i][j] = l1_h[t-1][i][j-L0S];
}

void forward(int seq_length)
{
    for(int t=1; t<=seq_length; t++)
    {
        // python code: self.xc[t] = np.hstack(( np.squeeze(xt[:, t-sp:t-sp+1, :], axis=1),  self.l1_h[t-1]))
        fill_l1_h_into_xc(t);

        // python code: self.l1_g_input[t] = np.dot(self.xc[t], self.param.l1_wg.T) + self.param.l1_bg
        mat_mul_b_T_add_bias( (FP*)&l1_g_input[t], (FP*)&xc[t], (FP*)l1_wg, BS, (L1S+L0S), L1S, (L1S+L0S), l1_bg);
        mat_mul_b_T_add_bias( (FP*)&l1_i_input[t], (FP*)&xc[t], (FP*)l1_wi, BS, (L1S+L0S), L1S, (L1S+L0S), l1_bi);
        mat_mul_b_T_add_bias( (FP*)&l1_f_input[t], (FP*)&xc[t], (FP*)l1_wf, BS, (L1S+L0S), L1S, (L1S+L0S), l1_bf);
        mat_mul_b_T_add_bias( (FP*)&l1_o_input[t], (FP*)&xc[t], (FP*)l1_wo, BS, (L1S+L0S), L1S, (L1S+L0S), l1_bo);

        // python code: self.l1_g[t] = np.tanh(self.l1_g_input[t])
        tanhf_on_matrix  ( (FP*)&l1_g[t], (FP*)&l1_g_input[t], BS, L1S);
        sigmoid_on_matrix( (FP*)&l1_i[t], (FP*)&l1_i_input[t], BS, L1S);
        sigmoid_on_matrix( (FP*)&l1_f[t], (FP*)&l1_f_input[t], BS, L1S);
        sigmoid_on_matrix( (FP*)&l1_o[t], (FP*)&l1_o_input[t], BS, L1S);

        // python code: self.l1_s[t] = self.l1_g[t] * self.l1_i[t] + self.l1_s[t-1] * self.l1_f[t]
        element_wise_mul( (FP*)&l1_s[t], (FP*)&l1_g[t], (FP*)&l1_i[t], BS, L1S);
        element_wise_mac( (FP*)&l1_s[t], (FP*)&l1_s[t-1], (FP*)&l1_f[t], BS, L1S);

        // python code: self.l1_h[t] = np.tanh(self.l1_s[t]) * self.l1_o[t]
        tanhf_on_matrix( (FP*)&l1_s_tanh[t], (FP*)l1_s[t], BS, L1S);
        element_wise_mul( (FP*)&l1_h[t], (FP*)&l1_s_tanh[t], (FP*)&l1_o[t], BS, L1S);


        // python code: self.l2_h[t] = np.dot(self.l1_h[t], self.param.l2_w.T) + self.param.l2_b 
        mat_mul_b_T_add_bias( (FP*)&l2_h[t], (FP*)&l1_h[t], (FP*)l2_w, BS, L1S, L2S, L1S, l2_b);
        softmax( (FP*)&l2_o[t], (FP*)&l2_h[t], BS, L2S);

    }
}

void print_network_out(int t)
{
    printf("Batch Size is %d\n", BS);

    printf("l2_h\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
            printf("%.8f  ", l2_h[t][i][j]);
        printf("\n");
    }

    printf("\n\nl2_o\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
            printf("%.8f  ", l2_o[t][i][j]);
        printf("\n");
    }
}


void element_wise_sub(FP* dst, FP* src1, FP* src2, int row, int col)
{
    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
            dst[i*col + j] = src1[i*col + j] - src2[i*col + j];
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

void find_ds(int t, int row, int col)
{
    // find the derivative of Loss w.r.t l1_s at time step t
    //python code: self.ds = self.state.l1_o[h_step] * (1 - (np.tanh(self.state.l1_s[h_step]))**2 ) *self.dl1_h

    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
            d_l1_s[i][j] = l1_o[t][i][j] * (1 - (l1_s_tanh[t][i][j])*(l1_s_tanh[t][i][j]) ) * d_l1_h[i][j];
}

void find_d_l1_ifgo_input(FP* dst, FP* src_a, FP* src_b, int row, int col, int check_g)
{
    int idx=0;

    if(check_g==1)
    {
        for(int i=0; i<row; i++)
            for(int j=0; j<col; j++)
            {   
                idx = i*col + j;
                dst[idx] = (1.0 - (src_a[idx]*src_a[idx])) * src_b[idx]; // derivative of tanh
            }
    }
    else 
    {
        for(int i=0; i<row; i++)
            for(int j=0; j<col; j++)
            {   
                idx = i*col + j;
                dst[idx] = src_a[idx] * (1.0 - src_a[idx]) * src_b[idx]; // derivative of tanh
            }
    }

}


void update_d_l1_h()
{
    FP tmp = 0;

    for(int i=0; i<BS; i++)
        for(int j=L0S; j<L0S+L1S; j++)
        {
            for(int k=0; k<L1S; k++)
                tmp += (di_input[i][k]*l1_wi[k][j] + df_input[i][k]*l1_wf[k][j] + dg_input[i][k]*l1_wg[k][j] + do_input[i][k]*l1_wo[k][j]);

            d_l1_h[i][j-L0S] = tmp;
            tmp = 0;
        }
}

void dbg_l1_w_b_o(int t)
{
    printf("t: %d, d_l1_h\t", t);
    for(int i=0; i<10; i++)
        printf("%.8f  ", d_l1_h[0][i]);
    printf("\n");

    printf("t: %d, d_l1_o\t", t);
    for(int i=0; i<10; i++)
        printf("%.8f  ", d_l1_o[0][i]);
    printf("\n");

    printf("t: %d, do_input\t", t);
    for(int i=0; i<10; i++)
        printf("%.8f  ", do_input[0][i]);
    printf("\n");

    printf("t: %d, l1_wo_grad\t", t);
    for(int i=0; i<10; i++)
        printf("%.8f  ", l1_wo_grad[0][i]);
    printf("\n");

    printf("t: %d, l1_bo_grad\t", t);
    for(int i=0; i<10; i++)
        printf("%.8f  ", l1_bo_grad[i]);
    printf("\n\n");
}


// Backpropagation from the given time step t
void backward(int t, int trunc_h, int trunc_s)
{
    int h_ep = 0; // earliest point(time step) to end up backtracking for l1_h
    int s_ep = 0; // earliest point(time step) to end up backtracking for l1_s

    element_wise_sub( (FP*)d_l2_h, (FP*)&l2_o[t], (FP*)label, BS, L2S);

    // python code: self.param.l2_w_diff =  (np.dot(self.dl2_h.T, self.state.l1_h[t]))/self.n_samples
    mat_mul_a_T_average( (FP*)l2_w_grad, (FP*)d_l2_h, (FP*)&l1_h[t], BS, L2S, BS, L1S, BS);

    // python code: self.param.l2_b_diff = (self.dl2_h.sum(axis=0))/self.n_samples # (10) = (200,10) 
    mat2vec_avr_sequeeze( (FP*)l2_b_grad, (FP*)d_l2_h, BS, L2S);

    // python code: self.dl1_h = np.dot(self.dl2_h, self.param.l2_w) # (200,128) = (200,10).(10, 128)
    mat_mul( (FP*)d_l1_h, (FP*)d_l2_h, (FP*)l2_w, BS, L2S, L2S, L1S);

    h_ep = (t-trunc_h>0) ? t-trunc_h : 0;
    for(int h_step=t; h_step>h_ep; h_step--)
    {
        find_ds(h_step, BS, L1S);

        // python: self.do = np.tanh(self.state.l1_s[h_step]) * self.dl1_h
        element_wise_mul( (FP*)d_l1_o, (FP*)&l1_s_tanh[h_step], (FP*)d_l1_h, BS, L1S);

        // python: self.do_input = sigmoid_derivative(self.state.l1_o[h_step]) * self.do 
        find_d_l1_ifgo_input( (FP*)do_input, (FP*)&l1_o[h_step], (FP*)d_l1_o, BS, L1S, 0);
        
        
        // python: self.param.l1_wo_diff += (np.dot(self.do_input.T, self.state.xc[h_step])) /self.n_samples
        mat_mul_a_T_average( (FP*)l1_wo_grad, (FP*)do_input, (FP*)&xc[h_step], BS, L1S, BS, (L1S+L0S), BS);
        
        // python: self.param.l1_bo_diff += (self.do_input.sum(axis=0)) /self.n_samples # (128) = (200,128)
        mat2vec_avr_sequeeze( (FP*)l1_bo_grad, (FP*)do_input, BS, L1S);
            
        dbg_l1_w_b_o(h_step);

        // python: s_ep = 0 if trunc_s is None else max(0, h_step -trunc_s)
        s_ep = (h_step-trunc_s>0) ? h_step-trunc_s : 0;

        // python: for s_step in np.arange(s_ep, h_step+1)[::-1]:
        for(int s_step=h_step; s_step>s_ep; s_step--)
        {
            // self.dg = self.state.l1_i[s_step] * self.ds
            element_wise_mul( (FP*)d_l1_g, (FP*)&l1_i[s_step], (FP*)d_l1_s, BS, L1S);
            element_wise_mul( (FP*)d_l1_i, (FP*)&l1_g[s_step], (FP*)d_l1_s, BS, L1S);
            element_wise_mul( (FP*)d_l1_f, (FP*)&l1_s[s_step-1], (FP*)d_l1_s, BS, L1S);

            // self.di_input = sigmoid_derivative(self.state.l1_i[s_step]) * self.di 
            find_d_l1_ifgo_input( (FP*)di_input, (FP*)&l1_i[s_step], (FP*)d_l1_i, BS, L1S, 0);
            find_d_l1_ifgo_input( (FP*)df_input, (FP*)&l1_f[s_step], (FP*)d_l1_f, BS, L1S, 0);
            find_d_l1_ifgo_input( (FP*)dg_input, (FP*)&l1_g[s_step], (FP*)d_l1_g, BS, L1S, 1);

            // self.param.l1_wi_diff += (np.dot(self.di_input.T, self.state.xc[s_step])) /self.n_samples
            mat_mul_a_T_average( (FP*)l1_wi_grad, (FP*)di_input, (FP*)&xc[s_step], BS, L1S, BS, (L1S+L0S), BS);
            mat_mul_a_T_average( (FP*)l1_wf_grad, (FP*)df_input, (FP*)&xc[s_step], BS, L1S, BS, (L1S+L0S), BS);
            mat_mul_a_T_average( (FP*)l1_wg_grad, (FP*)dg_input, (FP*)&xc[s_step], BS, L1S, BS, (L1S+L0S), BS);

            mat2vec_avr_sequeeze( (FP*)l1_bi_grad, (FP*)di_input, BS, L1S);
            mat2vec_avr_sequeeze( (FP*)l1_bf_grad, (FP*)df_input, BS, L1S);
            mat2vec_avr_sequeeze( (FP*)l1_bg_grad, (FP*)dg_input, BS, L1S);

            // self.ds= self.ds* self.state.l1_f[s_step]
            element_wise_mul( (FP*)d_l1_s, (FP*)d_l1_s, (FP*)&l1_f[s_step], BS, L1S);

            if(h_step == s_step)
                update_d_l1_h();
        }
    }

}

void SGD(FP* param, FP* grad, int row, int col)
{
    int idx;

    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
        {
            idx = i*col + j;
            param[idx] -= (LR * grad[idx]);
            grad[idx] = 0;
        }
}


void FPTT_SGD(FP* param, FP* grad, FP* rmean, FP* lbd, int row, int col)
{
    // python
    // self.l1_wg -= lr * (self.l1_wg_diff - self.l1_lbd_wg + alpha*(self.l1_wg - self.l1_rm_wg))
    // self.l1_lbd_wg -= alpha*(self.l1_wg - self.l1_rm_wg)
    // self.l1_rm_wg = 0.5*(self.l1_rm_wg + self.l1_wg) - (0.5/alpha)*self.l1_lbd_wg
    // self.l1_wg_diff = np.zeros_like(self.l1_wg)

    int idx;

    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
        {
            idx = i*col + j;
            param[idx] -= LR * (grad[idx] - lbd[idx] + ALPHA*(param[idx] - rmean[idx]) );
            lbd[idx] -= ALPHA * (param[idx] - rmean[idx]);
            rmean[idx] = 0.5*(rmean[idx] + param[idx]) - (0.5/ALPHA)*lbd[idx];
            grad[idx] = 0;
        }
}



void optimizer_and_zero_grad(int fptt_option)
{
    if(fptt_option==1)
    {
        FPTT_SGD( (FP*)l1_wi, (FP*)l1_wi_grad, (FP*)l1_wi_rm, (FP*)l1_wi_lbd, L1S, (L1S+L0S));
        FPTT_SGD( (FP*)l1_wf, (FP*)l1_wf_grad, (FP*)l1_wf_rm, (FP*)l1_wf_lbd, L1S, (L1S+L0S));
        FPTT_SGD( (FP*)l1_wg, (FP*)l1_wg_grad, (FP*)l1_wg_rm, (FP*)l1_wg_lbd, L1S, (L1S+L0S));
        FPTT_SGD( (FP*)l1_wo, (FP*)l1_wo_grad, (FP*)l1_wo_rm, (FP*)l1_wo_lbd, L1S, (L1S+L0S));

        FPTT_SGD( (FP*)l1_bi, (FP*)l1_bi_grad, (FP*)l1_bi_rm, (FP*)l1_bi_lbd, 1, L1S);
        FPTT_SGD( (FP*)l1_bf, (FP*)l1_bf_grad, (FP*)l1_bf_rm, (FP*)l1_bf_lbd, 1, L1S);
        FPTT_SGD( (FP*)l1_bg, (FP*)l1_bg_grad, (FP*)l1_bg_rm, (FP*)l1_bg_lbd, 1, L1S);
        FPTT_SGD( (FP*)l1_bo, (FP*)l1_bo_grad, (FP*)l1_bo_rm, (FP*)l1_bo_lbd, 1, L1S);

        FPTT_SGD( (FP*)l2_w, (FP*)l2_w_grad, (FP*)l2_w_rm, (FP*)l2_w_lbd, L2S, L1S);
        FPTT_SGD( (FP*)l2_b, (FP*)l2_b_grad, (FP*)l2_b_rm, (FP*)l2_b_lbd, 1, L2S);
    }
    else
    {
        SGD( (FP*)l1_wi, (FP*)l1_wi_grad, L1S, (L1S+L0S));
        SGD( (FP*)l1_wf, (FP*)l1_wf_grad, L1S, (L1S+L0S));
        SGD( (FP*)l1_wg, (FP*)l1_wg_grad, L1S, (L1S+L0S));
        SGD( (FP*)l1_wo, (FP*)l1_wo_grad, L1S, (L1S+L0S));

        SGD( (FP*)l1_bi, (FP*)l1_bi_grad, 1, L1S);
        SGD( (FP*)l1_bf, (FP*)l1_bf_grad, 1, L1S);
        SGD( (FP*)l1_bg, (FP*)l1_bg_grad, 1, L1S);
        SGD( (FP*)l1_bo, (FP*)l1_bo_grad, 1, L1S);

        SGD( (FP*)l2_w, (FP*)l2_w_grad, L2S, L1S);
        SGD( (FP*)l2_b, (FP*)l2_b_grad, 1, L2S);
    }
}





