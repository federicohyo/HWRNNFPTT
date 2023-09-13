#ifndef _LSTM_H_
#define _LSTM_H_

#include "headers.h"

extern FP max_ini_v;
extern FP min_ini_v;


//-------------------------------------
// define: Parameters (weights & biases)
//-------------------------------------
// weight definition of layer 1: LSTM
extern FP l1_wi[L1S * (L1S+L0S)];
extern FP l1_wf[L1S * (L1S+L0S)];
extern FP l1_wg[L1S * (L1S+L0S)];
extern FP l1_wo[L1S * (L1S+L0S)];

// bias definition of layer 1: LSTM
extern FP l1_bi[L1S];
extern FP l1_bf[L1S];
extern FP l1_bg[L1S];
extern FP l1_bo[L1S];

// weight definition of layer 2: FC 
extern FP l2_w[L2S * L1S];
// bias definition of layer 2: FC 
extern FP l2_b[L2S];


//-------------------------------------
// define: running mean (of Parameters (weights & biases))
//-------------------------------------
// weight running mean definition of layer 1: LSTM
extern FP l1_wi_rm[L1S * (L1S+L0S)];
extern FP l1_wf_rm[L1S * (L1S+L0S)];
extern FP l1_wg_rm[L1S * (L1S+L0S)];
extern FP l1_wo_rm[L1S * (L1S+L0S)];

// bias running mean definition of layer 1: LSTM
extern FP l1_bi_rm[L1S];
extern FP l1_bf_rm[L1S];
extern FP l1_bg_rm[L1S];
extern FP l1_bo_rm[L1S];

// weight running mean definition of layer 2: FC 
extern FP l2_w_rm[L2S * L1S];
// bias running mean definition of layer 2: FC 
extern FP l2_b_rm[L2S];


//-------------------------------------
// define: lambda (of Parameters (weights & biases))
//-------------------------------------
// weight lambda definition of layer 1: LSTM
extern FP l1_wi_lbd[L1S * (L1S+L0S)];
extern FP l1_wf_lbd[L1S * (L1S+L0S)];
extern FP l1_wg_lbd[L1S * (L1S+L0S)];
extern FP l1_wo_lbd[L1S * (L1S+L0S)];

// bias lambda definition of layer 1: LSTM
extern FP l1_bi_lbd[L1S];
extern FP l1_bf_lbd[L1S];
extern FP l1_bg_lbd[L1S];
extern FP l1_bo_lbd[L1S];

// weight lambda definition of layer 2: FC 
extern FP l2_w_lbd[L2S * L1S];
// bias lambda definition of layer 2: FC 
extern FP l2_b_lbd[L2S];


//-------------------------------------
// define: gradient (of Parameters (weights & biases))
//-------------------------------------
// weight gradient definition of layer 1: LSTM
extern FP l1_wi_grad[L1S * (L1S+L0S)];
extern FP l1_wf_grad[L1S * (L1S+L0S)];
extern FP l1_wg_grad[L1S * (L1S+L0S)];
extern FP l1_wo_grad[L1S * (L1S+L0S)];

// bias gradient definition of layer 1: LSTM
extern FP l1_bi_grad[L1S];
extern FP l1_bf_grad[L1S];
extern FP l1_bg_grad[L1S];
extern FP l1_bo_grad[L1S];

// weight gradient definition of layer 2: FC 
extern FP l2_w_grad[L2S * L1S];
// bias gradient definition of layer 2: FC 
extern FP l2_b_grad[L2S];



// load parameters from the external, to fine tune the pre-trained model
void load_param();

// Initialization of parameters and running mean, in case training from scratch
void initialize_param_and_rm(FP* input_param, FP* input_rm, int length, FP min_val, FP max_val);
void initialize_all_param_and_rm();

#endif//_LSTM_H_
