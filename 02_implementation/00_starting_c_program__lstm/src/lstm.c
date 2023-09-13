/*
    In lstm.h, see the definition
*/
#include "lstm.h"

FP max_ini_v = sqrt(1/(FP)(L1S));
FP min_ini_v = -sqrt(1/(FP)(L1S)); 

FP l1_wi[L1S * (L1S+L0S)];
FP l1_wf[L1S * (L1S+L0S)];
FP l1_wg[L1S * (L1S+L0S)];
FP l1_wo[L1S * (L1S+L0S)];
FP l1_bi[L1S];
FP l1_bf[L1S];
FP l1_bg[L1S];
FP l1_bo[L1S];
FP l2_w[L2S * L1S];
FP l2_b[L2S];


FP l1_wi_rm[L1S * (L1S+L0S)];
FP l1_wf_rm[L1S * (L1S+L0S)];
FP l1_wg_rm[L1S * (L1S+L0S)];
FP l1_wo_rm[L1S * (L1S+L0S)];
FP l1_bi_rm[L1S];
FP l1_bf_rm[L1S];
FP l1_bg_rm[L1S];
FP l1_bo_rm[L1S];
FP l2_w_rm[L2S * L1S];
FP l2_b_rm[L2S];


FP l1_wi_lbd[L1S * (L1S+L0S)] = {0};
FP l1_wf_lbd[L1S * (L1S+L0S)] = {0};
FP l1_wg_lbd[L1S * (L1S+L0S)] = {0};
FP l1_wo_lbd[L1S * (L1S+L0S)] = {0};
FP l1_bi_lbd[L1S] = {0};
FP l1_bf_lbd[L1S] = {0};
FP l1_bg_lbd[L1S] = {0};
FP l1_bo_lbd[L1S] = {0};
FP l2_w_lbd[L2S * L1S] = {0};
FP l2_b_lbd[L2S] = {0};


FP l1_wi_grad[L1S * (L1S+L0S)] = {0};
FP l1_wf_grad[L1S * (L1S+L0S)] = {0};
FP l1_wg_grad[L1S * (L1S+L0S)] = {0};
FP l1_wo_grad[L1S * (L1S+L0S)] = {0};
FP l1_bi_grad[L1S] = {0};
FP l1_bf_grad[L1S] = {0};
FP l1_bg_grad[L1S] = {0};
FP l1_bo_grad[L1S] = {0};
FP l2_w_grad[L2S * L1S] = {0};
FP l2_b_grad[L2S] = {0};



void load_params()
{

}


//  the running mean and the parameter are initialized as the same values
void initialize_param_and_rm(FP* input_param, FP* input_rm, int length, FP min_val, FP max_val)
{
    for (int i = 0; i < length; i++) 
    {
        // Scale the random integer to fit within the desired range
        FP scaledRand = ((FP)rand() / (FP)(RAND_MAX) ) * (max_val - min_val) + min_val;

        // Assign the scaled random value to the array element
        input_param[i] = scaledRand;
        input_rm[i] = scaledRand;
    }
} 

void initialize_all_param_and_rm()
{
    initialize_param_and_rm(l1_wi, l1_wi_rm, L1S*(L1S + L0S), max_ini_v, min_ini_v);
    initialize_param_and_rm(l1_wf, l1_wf_rm, L1S*(L1S + L0S), max_ini_v, min_ini_v);
    initialize_param_and_rm(l1_wg, l1_wg_rm, L1S*(L1S + L0S), max_ini_v, min_ini_v);
    initialize_param_and_rm(l1_wo, l1_wo_rm, L1S*(L1S + L0S), max_ini_v, min_ini_v);

    initialize_param_and_rm(l1_bi, l1_bi_rm, L1S, max_ini_v, min_ini_v);
    initialize_param_and_rm(l1_bf, l1_bf_rm, L1S, max_ini_v, min_ini_v);
    initialize_param_and_rm(l1_bg, l1_bg_rm, L1S, max_ini_v, min_ini_v);
    initialize_param_and_rm(l1_bo, l1_bo_rm, L1S, max_ini_v, min_ini_v);

    initialize_param_and_rm(l2_w, l2_w_rm, L1S*L2S, max_ini_v, min_ini_v);
    initialize_param_and_rm(l2_b, l2_b_rm, L1S*L2S, max_ini_v, min_ini_v);
}