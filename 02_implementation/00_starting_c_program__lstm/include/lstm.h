#ifndef _LSTM_H_
#define _LSTM_H_

#include "headers.h"

extern FP max_ini_v;
extern FP min_ini_v;


//-------------------------------------
// define: Parameters (weights & biases)
//-------------------------------------
// weight definition of layer 1: LSTM
extern FP l1_wi[L1_W_SIZE];
extern FP l1_wf[L1_W_SIZE];
extern FP l1_wg[L1_W_SIZE];
extern FP l1_wo[L1_W_SIZE];

// bias definition of layer 1: LSTM
extern FP l1_bi[L1_B_SIZE];
extern FP l1_bf[L1_B_SIZE];
extern FP l1_bg[L1_B_SIZE];
extern FP l1_bo[L1_B_SIZE];

// weight definition of layer 2: FC 
extern FP l2_w[L2_W_SIZE];
// bias definition of layer 2: FC 
extern FP l2_b[L2_B_SIZE];


//-------------------------------------
// define: running mean (of Parameters (weights & biases))
//-------------------------------------
// weight running mean definition of layer 1: LSTM
extern FP l1_wi_rm[L1_W_SIZE];
extern FP l1_wf_rm[L1_W_SIZE];
extern FP l1_wg_rm[L1_W_SIZE];
extern FP l1_wo_rm[L1_W_SIZE];

// bias running mean definition of layer 1: LSTM
extern FP l1_bi_rm[L1_B_SIZE];
extern FP l1_bf_rm[L1_B_SIZE];
extern FP l1_bg_rm[L1_B_SIZE];
extern FP l1_bo_rm[L1_B_SIZE];

// weight running mean definition of layer 2: FC 
extern FP l2_w_rm[L2_W_SIZE];
// bias running mean definition of layer 2: FC 
extern FP l2_b_rm[L2_B_SIZE];


//-------------------------------------
// define: lambda (of Parameters (weights & biases))
//-------------------------------------
// weight lambda definition of layer 1: LSTM
extern FP l1_wi_lbd[L1_W_SIZE];
extern FP l1_wf_lbd[L1_W_SIZE];
extern FP l1_wg_lbd[L1_W_SIZE];
extern FP l1_wo_lbd[L1_W_SIZE];

// bias lambda definition of layer 1: LSTM
extern FP l1_bi_lbd[L1_B_SIZE];
extern FP l1_bf_lbd[L1_B_SIZE];
extern FP l1_bg_lbd[L1_B_SIZE];
extern FP l1_bo_lbd[L1_B_SIZE];

// weight lambda definition of layer 2: FC 
extern FP l2_w_lbd[L2_W_SIZE];
// bias lambda definition of layer 2: FC 
extern FP l2_b_lbd[L2_B_SIZE];


//-------------------------------------
// define: gradient (of Parameters (weights & biases))
//-------------------------------------
// weight gradient definition of layer 1: LSTM
extern FP l1_wi_grad[L1_W_SIZE];
extern FP l1_wf_grad[L1_W_SIZE];
extern FP l1_wg_grad[L1_W_SIZE];
extern FP l1_wo_grad[L1_W_SIZE];

// bias gradient definition of layer 1: LSTM
extern FP l1_bi_grad[L1_B_SIZE];
extern FP l1_bf_grad[L1_B_SIZE];
extern FP l1_bg_grad[L1_B_SIZE];
extern FP l1_bo_grad[L1_B_SIZE];

// weight gradient definition of layer 2: FC 
extern FP l2_w_grad[L2_W_SIZE];
// bias gradient definition of layer 2: FC 
extern FP l2_b_grad[L2_B_SIZE];





extern FP input_samples[BS * ]

//--------------------------------------
// intermediate network states
//--------------------------------------
extern FP l1_i_input[TS * BS * L1S];
extern FP l1_f_input[TS * BS * L1S];
extern FP l1_g_input[TS * BS * L1S];
extern FP l1_o_input[TS * BS * L1S];


extern FP l1_i[TS * BS * L1S];
extern FP l1_f[TS * BS * L1S];
extern FP l1_g[TS * BS * L1S];
extern FP l1_o[TS * BS * L1S];
extern FP l1_s[TS * BS * L1S];
extern FP l1_h[TS * BS * L1S];


extern FP l2_h[TS * BS * L2S];
extern FP l2_o[TS * BS * L2S];









// load parameters used by PyTorch model
// case 1: use the same initialization as PyTorch model
// case 2: re-train the pre-trained PyTorch model 
void load_param_and_rm(FP* param, FP* rm, int size, char file_dir[], char file_name[]);
void load_all_param_and_rm(char file_dir[]);



// Initialization of parameters and running mean
// in case training from scratch
void initialize_param_and_rm(FP* input_param, FP* input_rm, int length, FP min_val, FP max_val);
void initialize_all_param_and_rm();


// Network forward path
void forward(int time_steps, FP* input_samples);




#endif//_LSTM_H_
