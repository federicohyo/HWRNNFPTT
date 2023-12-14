#ifndef NETWORK_PARAMS_X86
#define NETWORK_PARAMS_X86

//-------------------------------------
// define: Parameters (weights & biases)
//-------------------------------------
// weight definition of layer 1: LSTM
// to be randomized
FP l1_wi[L1S][(L1S + L0S)];
FP l1_wf[L1S][(L1S + L0S)];
FP l1_wg[L1S][(L1S + L0S)];
FP l1_wo[L1S][(L1S + L0S)];
// bias definition of layer 1: LSTM
FP l1_bi[L1S];
FP l1_bf[L1S];
FP l1_bg[L1S];
FP l1_bo[L1S];
// weight definition of layer 2: FC 
FP l2_w[L2S][L1S];
// bias definition of layer 2: FC 
FP l2_b[L2S];


//-------------------------------------
// define: running mean (of Parameters (weights & biases))
//-------------------------------------
// weight running mean definition of layer 1: LSTM
// to be randomized
FP l1_wi_rm[L1S][(L1S + L0S)];
FP l1_wf_rm[L1S][(L1S + L0S)];
FP l1_wg_rm[L1S][(L1S + L0S)];
FP l1_wo_rm[L1S][(L1S + L0S)];
// bias running mean definition of layer 1: LSTM
FP l1_bi_rm[L1S];
FP l1_bf_rm[L1S];
FP l1_bg_rm[L1S];
FP l1_bo_rm[L1S];
// weight running mean definition of layer 2: FC 
FP l2_w_rm[L2S][L1S];
// bias running mean definition of layer 2: FC 
FP l2_b_rm[L2S];


//-------------------------------------
// define: lambda (of Parameters (weights & biases))
//-------------------------------------
// weight lambda definition of layer 1: LSTM
FP l1_wi_lbd[L1S][(L1S + L0S)] = {0};
FP l1_wf_lbd[L1S][(L1S + L0S)] = {0};
FP l1_wg_lbd[L1S][(L1S + L0S)] = {0};
FP l1_wo_lbd[L1S][(L1S + L0S)] = {0};
// bias lambda definition of layer 1: LSTM
FP l1_bi_lbd[L1S] = {0};
FP l1_bf_lbd[L1S] = {0};
FP l1_bg_lbd[L1S] = {0};
FP l1_bo_lbd[L1S] = {0};
// weight lambda definition of layer 2: FC 
FP l2_w_lbd[L2S][L1S] = {0};
// bias lambda definition of layer 2: FC 
FP l2_b_lbd[L2S] = {0};


//-------------------------------------
// define: gradient (of Parameters (weights & biases))
//-------------------------------------
// weight gradient definition of layer 1: LSTM
FP l1_wi_grad[L1S][(L1S + L0S)] = {0};
FP l1_wf_grad[L1S][(L1S + L0S)] = {0};
FP l1_wg_grad[L1S][(L1S + L0S)] = {0};
FP l1_wo_grad[L1S][(L1S + L0S)] = {0};
// bias gradient definition of layer 1: LSTM
FP l1_bi_grad[L1S] = {0};
FP l1_bf_grad[L1S] = {0};
FP l1_bg_grad[L1S] = {0};
FP l1_bo_grad[L1S] = {0};
// weight gradient definition of layer 2: FC 
FP l2_w_grad[L2S][L1S] = {0};
// bias gradient definition of layer 2: FC 
FP l2_b_grad[L2S] = {0};


//--------------------------------------
// composed input to the LSTM
// L0S: input sample from the external
// L1S: hidden states of the last time step
//--------------------------------------
FP samples[BS][NUM_OF_P];
FP xc[TS+1][BS][(L0S+L1S)] = {0};
// the label/expected output
FP label[BS][L2S] = {0};

//--------------------------------------
// intermediate network states
//--------------------------------------
// l1_h and l1_s has the range: 0 - T, size T+1
// the other states has the range: 1 -T, size T
// but to make iterator more clear for now, they all have the size T+1
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


//--------------------------------------
// intermediate gradients
// variables used in backward path
//--------------------------------------
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


#endif// NETWORK_PARAMS_X86