#ifndef _LSTM_H_
#define _LSTM_H_

#include "matrix_ops.h"

#ifdef PLATFORM_X86
#include "network_params_x86.h"
#endif

#ifdef PLATFORM_RV
#include "network_params_riscv.h"
#endif



// DATA DEFINITIONS 
FP max_ini_v = sqrt(1/(FP)(L1S));
FP min_ini_v = -sqrt(1/(FP)(L1S)); 


//--------------------------------------
// FUNCTIONS DECLARATION
//--------------------------------------
void print_basic_config()
{
    printf("\n---------- Basic Configurations ----------\n");
    printf("Size of 3-layer network (input-LSTM-FC output): %dx%dx%d\n", L0S, L1S, L2S);
    printf("batch size: %d\n", BS);
    printf("learning rate: %f\n", LR);
    printf("Optimizer is SGD ");
    #ifdef MNIST_ONLINE_BY_PIXELS
        #ifdef K_VAL
            printf("and online formulation is enabled, K is %d, alpha: %f\n", K_VAL, ALPHA);
        #else
            printf("It is necessary to define K value when you want online formulation\n");
            exit(1);
        #endif

        #ifdef REGULARIZATION
            printf("Regularization is also enabled. Thus complete FPTT (online+reg).\n");
        #else
            printf("Regularization is NOT enabled\n");
        #endif
    #else
        printf("NOT using online formulation\n");
    #endif

#ifdef PLATFORM_RV
    printf("Running on RISC-V based platform, and ");
    #ifdef USE_GEMMINI_LIB
        printf("Gemmini is active\n");
    #else
        printf("Gemmini is NOT active\n");
    #endif
#endif

#ifdef PLATFORM_X86
    printf("Running on x86 based platform\n");
#endif

}


void print_static_memory_usage()
{
    int sum=0;
    int mem_params=0;
    int mem_states=0;
    int mem_grad=0;
    int mem_in_dat=0;

    mem_in_dat = sizeof(samples);
    mem_params = 4* (4*(sizeof(l1_wi) + sizeof(l1_bi)) + sizeof(l2_w) + sizeof(l2_b));
    mem_states = sizeof(xc) + sizeof(l1_i_input)*4 + sizeof(l1_i)*7 + sizeof(l2_h)*2;
    mem_grad = sizeof(d_l2_h) + sizeof(d_l1_h)*10;
    sum = mem_params + mem_states +mem_grad;

    printf(" \n---------- Memory Usage (static) ----------\n");
    printf("Network Input:\t %.2f MB (%d Bytes)\n", ((float)mem_in_dat/1024/1024), mem_in_dat);
    printf("Network parameters:\t %.2f MB (%d Bytes)\n", ((float)mem_params/1024/1024), mem_params);
    printf("Network states:\t\t %.2f MB (%d Bytes) \n", ((float)mem_states/1024/1024), mem_states);
    printf("Intermediate gradients:\t %.2f MB (%d Bytes) \n", ((float)mem_grad/1024/1024), mem_grad);
    printf("Total:\t\t\t %.2f MB (%d Bytes) \n", ((float)sum/1024/1024), sum);
    printf(" -------------------------------------------\n");
}


// load parameters used by PyTorch model from external files
// case 1: use the same initialization as PyTorch model
// case 2: re-train the pre-trained PyTorch model 
#ifndef BM
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
#endif

#ifndef BM
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
#endif


//  the running mean and the parameter are initialized as the same values
// Initialization of parameters and running mean
// in case training from scratch
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


// load input samples from external files
#ifndef BM
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
#endif

#ifndef BM
void load_input_samples(char file_name[])
{
   FILE* ptr = fopen(file_name, "r");
   if (ptr == NULL) 
   {
       printf("[load_input_samples]: No such file: %s.\n", file_name);    exit(1);
   }

   for(int i=0; i<BS; i++)
       for(int j=0; j<NUM_OF_P; j++)
           fscanf(ptr, "%f", &samples[i][j] );

   fclose(ptr);
}
#endif

// fill xc with data necessary for forward
// fill xc with a fraction of input samples of TS step (a piece)
void load_sub_seq_to_xc(int k_idx)
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

    /* todo: This index can be improved to be more applicable, i.e. for CIFAR-10 */
    for(int i=0; i<BS; i++)
        for(int j=1; j<=TS; j++)
            xc[j][i][0] = samples[i][k_idx*TS + j-1];

#ifdef PRINT_PERF
size_t end = rdcycle();
acc_load_sub_seq_to_xc += (end - start);
#endif

}

// link network states: l1_h[0] = l1_h[TS], if k_idx > 0
void relay_network_states()
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

    for(int i=0; i<BS; i++)
        for(int k=0; k<L1S; k++)
        {
            l1_h[0][i][k] = l1_h[TS][i][k]; 
            l1_s[0][i][k] = l1_s[TS][i][k]; 
        }

#ifdef PRINT_PERF
size_t end = rdcycle();
acc_relay_network_states += (end - start);
#endif
}


// xc is the concatenation of input and hidden states
void fill_l1_h_into_xc(int t)
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

    for(int i=0; i<BS; i++)
        for(int j=L0S; j<(L0S+L1S); j++)
            xc[t][i][j] = l1_h[t-1][i][j-L0S];

#ifdef PRINT_PERF
size_t end = rdcycle();
acc_fill_l1_h_into_xc += (end - start);
#endif

}

// Network forward path for [t] time steps
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
        element_wise_mul( (FP*)&l1_s[t], (FP*)&l1_g[t], (FP*)&l1_i[t], BS, L1S, "forward_pass");
        element_wise_mac( (FP*)&l1_s[t], (FP*)&l1_s[t-1], (FP*)&l1_f[t], BS, L1S);

        // python code: self.l1_h[t] = np.tanh(self.l1_s[t]) * self.l1_o[t]
        tanhf_on_matrix( (FP*)&l1_s_tanh[t], (FP*)l1_s[t], BS, L1S);
        element_wise_mul( (FP*)&l1_h[t], (FP*)&l1_s_tanh[t], (FP*)&l1_o[t], BS, L1S, "forward_pass");


        // python code: self.l2_h[t] = np.dot(self.l1_h[t], self.param.l2_w.T) + self.param.l2_b 
        mat_mul_b_T_add_bias( (FP*)&l2_h[t], (FP*)&l1_h[t], (FP*)l2_w, BS, L1S, L2S, L1S, l2_b);
        softmax( (FP*)&l2_o[t], (FP*)&l2_h[t], BS, L2S);

    }
}


#ifdef PRINT_DEBUG
void debug(int t)
{
    printf("\n\nl1_g_input\n");
    for(int i=0; i<BS; i++)
    {
        for(int j=0; j<L2S; j++)
            printf("%.8f  ", l1_g_input[t][i][j]);
        printf("\n");
    }

    printf("\n\nl1_g\n");
    for(int i=0; i<BS; i++)
    {
        for(int j=0; j<L2S; j++)
            printf("%.8f  ", l1_g[t][i][j]);
        printf("\n");
    }

    printf("\n\nl1_s\n");
    for(int i=0; i<BS; i++)
    {
        for(int j=0; j<L2S; j++)
            printf("%.8f  ", l1_s[t][i][j]);
        printf("\n");
    }

    printf("\n\nl1_h\n");
    for(int i=0; i<1; i++)
    {
        for(int j=0; j<L1S; j++)
            printf("%.8f  ", l1_h[t][i][j]);
        printf("\n");
    }
    printf("\n\nl2_w\n");
    for(int i=1; i<2; i++)
    {
        for(int j=0; j<L1S; j++)
            printf("%.8f  ", l2_w[i][j]);
        printf("\n");
    }

    printf("\n\nl2_b\n");
    for(int i=0; i<L2S; i++)
        printf("%.8f  ", l2_b[i]);
    printf("\n");

    printf("\n\nl2_h\n");
    for(int i=0; i<BS; i++)
    {
        for(int j=0; j<L2S; j++)
            printf("%.8f  ", l2_h[t][i][j]);
        printf("\n");
    }

    printf("\n\nl2_o\n");
    for(int i=0; i<BS; i++)
    {
        for(int j=0; j<L2S; j++)
            printf("%.8f  ", l2_o[t][i][j]);
        printf("\n");
    }
}

void print_network_out(int t)
{
    // printf("l2_h\n");
    // for(int i=0; i<BS; i++)
    // {
    //     printf("Sample no. %d: ", i);
    //     for(int j=0; j<L2S; j++)
    //         printf("%.8f  ", l2_h[t][i][j]);
    //     printf("\n");
    // }

    printf("\n\nl2_o\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
            printf("%.8f  ", l2_o[t][i][j]);
        printf("\n");
    }
}
#endif

// helper function: used only in [backward]
void find_ds(int t, int row, int col)
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

    // find the derivative of Loss w.r.t l1_s at time step t
    //python code: self.ds = self.state.l1_o[h_step] * (1 - (np.tanh(self.state.l1_s[h_step]))**2 ) *self.dl1_h
    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
            d_l1_s[i][j] = l1_o[t][i][j] * (1 - (l1_s_tanh[t][i][j])*(l1_s_tanh[t][i][j]) ) * d_l1_h[i][j];


#ifdef PRINT_PERF
size_t end = rdcycle();
acc_find_ds += (end - start);
#endif

#ifdef PRINT_COUNTER
// non-functional: operation counter below
bp_mul += row*col*3;
bp_sub += row*col;
#endif
}

// helper function: used only in [backward]
void find_d_l1_ifgo_input(FP* dst, FP* src_a, FP* src_b, int row, int col, int check_g)
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

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

#ifdef PRINT_PERF
size_t end = rdcycle();
acc_find_d_l1_ifgo_input += (end - start);
#endif

#ifdef PRINT_COUNTER
// non-functional: operation counter below
bp_mul += row*col*2;
bp_sub += row*col;
// index calculation
index_mul += row*col;
index_add += row*col;
#endif

}

// helper function: used only in [backward]
void update_d_l1_h()
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif
#ifndef USE_GEMMINI_LIB 
    FP tmp = 0;
    for(int i=0; i<BS; i++)
        for(int j=L0S; j<L0S+L1S; j++)
        {
            for(int k=0; k<L1S; k++)
                tmp += (di_input[i][k]*l1_wi[k][j] + df_input[i][k]*l1_wf[k][j] + dg_input[i][k]*l1_wg[k][j] + do_input[i][k]*l1_wo[k][j]);

            d_l1_h[i][j-L0S] = tmp;
            tmp = 0;
        }
#endif

#ifdef USE_GEMMINI_LIB 
      tiled_matmul_auto(BS, (L0S+L1S), L1S,
                    (FP*)di_input, (FP*)l1_wi, NULL, (FP*)tmp1,
                    L1S, (L0S+L1S), (L0S+L1S), (L0S+L1S),
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                    false, false,
                    false, !FULL_BIAS_WIDTH,
                    0,
                    WMM);

      tiled_matmul_auto(BS, (L0S+L1S), L1S,
                    (FP*)df_input, (FP*)l1_wf, NULL, (FP*)tmp2,
                    L1S, (L0S+L1S), (L0S+L1S), (L0S+L1S),
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                    false, false,
                    false, !FULL_BIAS_WIDTH,
                    0,
                    WMM);

      tiled_matmul_auto(BS, (L0S+L1S), L1S,
                    (FP*)dg_input, (FP*)l1_wg, NULL, (FP*)tmp3,
                    L1S, (L0S+L1S), (L0S+L1S), (L0S+L1S),
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                    false, false,
                    false, !FULL_BIAS_WIDTH,
                    0,
                    WMM);

      tiled_matmul_auto(BS, (L0S+L1S), L1S,
                    (FP*)do_input, (FP*)l1_wo, NULL, (FP*)tmp4,
                    L1S, (L0S+L1S), (L0S+L1S), (L0S+L1S),
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                    false, false,
                    false, !FULL_BIAS_WIDTH,
                    0,
                    WMM);

    for(int i=0; i<BS; i++)
        for(int j=L0S; j<L0S+L1S; j++)
            d_l1_h[i][j-L0S] = tmp1[i][j] +  tmp2[i][j] + tmp3[i][j] + tmp4[i][j]; 
#endif
#ifdef PRINT_PERF
size_t end = rdcycle();
acc_update_d_l1_h += (end - start);
#endif


#ifdef PRINT_COUNTER
// non-functional: operation counter below
bp_mul += (BS)*(L1S)*(L1S)*4;
bp_add += (BS)*((L1S)-1)*(L1S)*4;
// index calculation
index_sub += BS *L1S;
#endif
}

#ifdef PRINT_DEBUG
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
#endif


// Backpropagation from the given time step t
void backward(int t, int trunc_h, int trunc_s)
{
    static int cnt=0;
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
        cnt++;
        find_ds(h_step, BS, L1S);

        // python: self.do = np.tanh(self.state.l1_s[h_step]) * self.dl1_h
        element_wise_mul( (FP*)d_l1_o, (FP*)&l1_s_tanh[h_step], (FP*)d_l1_h, BS, L1S, "backward_pass");

        // python: self.do_input = sigmoid_derivative(self.state.l1_o[h_step]) * self.do 
        find_d_l1_ifgo_input( (FP*)do_input, (FP*)&l1_o[h_step], (FP*)d_l1_o, BS, L1S, 0);
        
        
        // python: self.param.l1_wo_diff += (np.dot(self.do_input.T, self.state.xc[h_step])) /self.n_samples
        mat_mul_a_T_average( (FP*)l1_wo_grad, (FP*)do_input, (FP*)&xc[h_step], BS, L1S, BS, (L1S+L0S), BS);
        
        // python: self.param.l1_bo_diff += (self.do_input.sum(axis=0)) /self.n_samples # (128) = (200,128)
        mat2vec_avr_sequeeze( (FP*)l1_bo_grad, (FP*)do_input, BS, L1S);
            
        // dbg_l1_w_b_o(h_step);

        // python: s_ep = 0 if trunc_s is None else max(0, h_step -trunc_s)
        s_ep = (h_step-trunc_s>0) ? h_step-trunc_s : 0;

        // python: for s_step in np.arange(s_ep, h_step+1)[::-1]:
        for(int s_step=h_step; s_step>s_ep; s_step--)
        {
            element_wise_mul( (FP*)d_l1_g, (FP*)&l1_i[s_step], (FP*)d_l1_s, BS, L1S, "backward_pass");
            element_wise_mul( (FP*)d_l1_i, (FP*)&l1_g[s_step], (FP*)d_l1_s, BS, L1S, "backward_pass");
            element_wise_mul( (FP*)d_l1_f, (FP*)&l1_s[s_step-1], (FP*)d_l1_s, BS, L1S, "backward_pass");

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
            element_wise_mul( (FP*)d_l1_s, (FP*)d_l1_s, (FP*)&l1_f[s_step], BS, L1S, "backward_pass");

            if(h_step == s_step)
                update_d_l1_h();

            // printf("h_step, s_step: %d, %d", h_step, s_step);
            // for(int i=0; i<10; i++)
            //     printf("%.6f  ", l1_wi_grad[0][i]);
            // printf("\n");
        }
    }
    // printf("counter:%d\n", cnt);
}

void SGD(FP* param, FP* grad, int row, int col)
{

#ifdef PRINT_PERF 
size_t start = rdcycle();
#endif

    int idx;
    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
        {
            idx = i*col + j;
            param[idx] -= (LR * grad[idx]);
            grad[idx] = 0;
        }

#ifdef PRINT_PERF 
size_t end = rdcycle();
acc_SGD += (end - start);
#endif

#ifdef PRINT_COUNTER
fptt_sub += row*col;
fptt_mul += row*col;
// index calculation
index_mul += row*col;
index_add += row*col;
#endif 

}

void FPTT_SGD(FP* param, FP* grad, FP* rmean, FP* lbd, int row, int col)
{
    // python
    // self.l1_wg -= lr * (self.l1_wg_diff - self.l1_lbd_wg + alpha*(self.l1_wg - self.l1_rm_wg))
    // self.l1_lbd_wg -= alpha*(self.l1_wg - self.l1_rm_wg)
    // self.l1_rm_wg = 0.5*(self.l1_rm_wg + self.l1_wg) - (0.5/alpha)*self.l1_lbd_wg
    // self.l1_wg_diff = np.zeros_like(self.l1_wg)

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

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

#ifdef PRINT_PERF
size_t end = rdcycle();
acc_FPTT_SGD += (end - start);
#endif

#ifdef PRINT_COUNTER
fptt_add += row*col*(2);
fptt_sub += row*col*(6);
fptt_mul += row*col*(5);
// index calculation
index_mul += row*col;
index_add += row*col;
#endif

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

#ifdef PRINT_DEBUG
void print_params_partly()
{
	printf("l2_w: ");
	for(int i=0; i<10; i++)
		printf("%.8f ", l2_w[0][i]);
	printf("\n");

	printf("l2_b: ");
	for(int i=0; i<10; i++)
		printf("%.8f ", l2_b[i]);
	printf("\n");

	printf("l1_wo: ");
	for(int i=0; i<10; i++)
		printf("%.8f ", l1_wo[0][i]);
	printf("\n");

	printf("l1_bo: ");
	for(int i=0; i<10; i++)
		printf("%.8f ", l1_bo[i]);
	printf("\n");

	printf("l1_wi: ");
	for(int i=0; i<10; i++)
		printf("%.8f ", l1_wi[0][i]);
	printf("\n");

	printf("l1_bi: ");
	for(int i=0; i<10; i++)
		printf("%.8f ", l1_bi[i]);
	printf("\n");

	printf("l1_wf: ");
	for(int i=0; i<10; i++)
		printf("%.8f ", l1_wf[0][i]);
	printf("\n");

	printf("l1_bf: ");
	for(int i=0; i<10; i++)
		printf("%.8f ", l1_bf[i]);
	printf("\n");

	printf("l1_wg: ");
	for(int i=0; i<10; i++)
		printf("%.8f ", l1_wg[0][i]);
	printf("\n");

	printf("l1_bg: ");
	for(int i=0; i<10; i++)
		printf("%.8f ", l1_bg[i]);
	printf("\n");
}
#endif

#endif //_LSTM_H_