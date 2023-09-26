#ifndef _LSTM_H_
#define _LSTM_H_

#include "matrix_ops.h"

// DATA DEFINITIONS 

extern FP max_ini_v;
extern FP min_ini_v;


//-------------------------------------
// define: Parameters (weights & biases)
//-------------------------------------
// weight definition of layer 1: LSTM
extern FP l1_wi[L1S][(L1S + L0S)];
extern FP l1_wf[L1S][(L1S + L0S)];
extern FP l1_wg[L1S][(L1S + L0S)];
extern FP l1_wo[L1S][(L1S + L0S)];

// bias definition of layer 1: LSTM
extern FP l1_bi[L1S];
extern FP l1_bf[L1S];
extern FP l1_bg[L1S];
extern FP l1_bo[L1S];

// weight definition of layer 2: FC 
extern FP l2_w[L2S][L1S];
// bias definition of layer 2: FC 
extern FP l2_b[L2S];


//-------------------------------------
// define: running mean (of Parameters (weights & biases))
//-------------------------------------
// weight running mean definition of layer 1: LSTM
extern FP l1_wi_rm[L1S][(L1S + L0S)];
extern FP l1_wf_rm[L1S][(L1S + L0S)];
extern FP l1_wg_rm[L1S][(L1S + L0S)];
extern FP l1_wo_rm[L1S][(L1S + L0S)];

// bias running mean definition of layer 1: LSTM
extern FP l1_bi_rm[L1S];
extern FP l1_bf_rm[L1S];
extern FP l1_bg_rm[L1S];
extern FP l1_bo_rm[L1S];

// weight running mean definition of layer 2: FC 
extern FP l2_w_rm[L2S][L1S];
// bias running mean definition of layer 2: FC 
extern FP l2_b_rm[L2S];


//-------------------------------------
// define: lambda (of Parameters (weights & biases))
//-------------------------------------
// weight lambda definition of layer 1: LSTM
extern FP l1_wi_lbd[L1S][(L1S + L0S)];
extern FP l1_wf_lbd[L1S][(L1S + L0S)];
extern FP l1_wg_lbd[L1S][(L1S + L0S)];
extern FP l1_wo_lbd[L1S][(L1S + L0S)];

// bias lambda definition of layer 1: LSTM
extern FP l1_bi_lbd[L1S];
extern FP l1_bf_lbd[L1S];
extern FP l1_bg_lbd[L1S];
extern FP l1_bo_lbd[L1S];

// weight lambda definition of layer 2: FC 
extern FP l2_w_lbd[L2S][L1S];
// bias lambda definition of layer 2: FC 
extern FP l2_b_lbd[L2S];


//-------------------------------------
// define: gradient (of Parameters (weights & biases))
//-------------------------------------
// weight gradient definition of layer 1: LSTM
extern FP l1_wi_grad[L1S][(L1S + L0S)];
extern FP l1_wf_grad[L1S][(L1S + L0S)];
extern FP l1_wg_grad[L1S][(L1S + L0S)];
extern FP l1_wo_grad[L1S][(L1S + L0S)];

// bias gradient definition of layer 1: LSTM
extern FP l1_bi_grad[L1S];
extern FP l1_bf_grad[L1S];
extern FP l1_bg_grad[L1S];
extern FP l1_bo_grad[L1S];

// weight gradient definition of layer 2: FC 
extern FP l2_w_grad[L2S][L1S];
// bias gradient definition of layer 2: FC 
extern FP l2_b_grad[L2S];


//--------------------------------------
// composed input to the LSTM
// L0S: input sample from the external
// L1S: hidden states of the last time step
//--------------------------------------
extern FP xc[TS+1][BS][(L0S+L1S)]; // range: 0 - TS

// the label/expected output
extern FP label[BS][L2S];

//--------------------------------------
// intermediate network states
//--------------------------------------
// l1_h and l1_s has the range: 0 - T, size T+1
// the other states has the range: 1 -T, size T
// but to make iterator more clear for now, they all have the size T+1
extern FP l1_i_input[TS+1][BS][L1S];
extern FP l1_f_input[TS+1][BS][L1S];
extern FP l1_g_input[TS+1][BS][L1S];
extern FP l1_o_input[TS+1][BS][L1S];


extern FP l1_i[TS+1][BS][L1S];
extern FP l1_f[TS+1][BS][L1S];
extern FP l1_g[TS+1][BS][L1S];
extern FP l1_o[TS+1][BS][L1S];
extern FP l1_s[TS+1][BS][L1S];
extern FP l1_s_tanh[TS+1][BS][L1S];
extern FP l1_h[TS+1][BS][L1S];


extern FP l2_h[TS+1][BS][L2S];
extern FP l2_o[TS+1][BS][L2S];

//--------------------------------------
// intermediate gradients
// variables used in backward path
//--------------------------------------
extern FP d_l2_h[BS][L2S];

extern FP d_l1_h[BS][L1S];
extern FP d_l1_s[BS][L1S];
extern FP d_l1_i[BS][L1S];
extern FP d_l1_f[BS][L1S];
extern FP d_l1_g[BS][L1S];
extern FP d_l1_o[BS][L1S];

extern FP di_input[BS][L1S];
extern FP df_input[BS][L1S];
extern FP dg_input[BS][L1S];
extern FP do_input[BS][L1S];




//--------------------------------------
// FUNCTIONS DECLARATION
//--------------------------------------
void print_basic_config();
void print_static_memory_usage();

// load parameters used by PyTorch model from external files
// case 1: use the same initialization as PyTorch model
// case 2: re-train the pre-trained PyTorch model 
void load_param_and_rm(FP* param, FP* rm, int row, int col, char file_dir[], char file_name[]);
void load_all_param_and_rm(char file_dir[]);

// Initialization of parameters and running mean
// in case training from scratch
void initialize_param_and_rm(FP* input_param, FP* input_rm, int row, int col, FP min_val, FP max_val);
void initialize_all_param_and_rm();

// load input samples from external files
void load_input_samples_to_xc(char file_name[]);

// xc is the concatenation of input and hidden states
void fill_l1_h_into_xc(int t);

// output the l2_o
void print_network_out(int t);

// Network forward path for [t] time steps
void forward(int seq_length);

void backward(int t, int trunc_h, int trunc_s);

void optimizer_and_zero_grad(int fptt_option);

void print_updated_params_partly();
#endif//_LSTM_H_
