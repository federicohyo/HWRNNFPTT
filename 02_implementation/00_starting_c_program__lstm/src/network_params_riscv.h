#ifndef NETWORK_PARAMS_RV
#define NETWORK_PARAMS_RV

// network_params_x86.h has more comments

static FP tmp1[BS][(L0S+L1S)] row_align(1);
static FP tmp2[BS][(L0S+L1S)] row_align(1);
static FP tmp3[BS][(L0S+L1S)] row_align(1);
static FP tmp4[BS][(L0S+L1S)] row_align(1);

static FP l1_wi[L1S][(L1S + L0S)] row_align(1);
static FP l1_wf[L1S][(L1S + L0S)] row_align(1);
static FP l1_wg[L1S][(L1S + L0S)] row_align(1);
static FP l1_wo[L1S][(L1S + L0S)] row_align(1);
static FP l1_bi[L1S] row_align(1);
static FP l1_bf[L1S] row_align(1);
static FP l1_bg[L1S] row_align(1);
static FP l1_bo[L1S] row_align(1);
static FP l2_w[L2S][L1S] row_align(1);
static FP l2_b[L2S] row_align(1);
static FP l1_wi_rm[L1S][(L1S + L0S)] row_align(1);
static FP l1_wf_rm[L1S][(L1S + L0S)] row_align(1);
static FP l1_wg_rm[L1S][(L1S + L0S)] row_align(1);
static FP l1_wo_rm[L1S][(L1S + L0S)] row_align(1);
static FP l1_bi_rm[L1S] row_align(1);
static FP l1_bf_rm[L1S] row_align(1);
static FP l1_bg_rm[L1S] row_align(1);
static FP l1_bo_rm[L1S] row_align(1);
static FP l2_w_rm[L2S][L1S] row_align(1);
static FP l2_b_rm[L2S] row_align(1);

static FP l1_wi_lbd[L1S][(L1S + L0S)] row_align(1);
static FP l1_wf_lbd[L1S][(L1S + L0S)] row_align(1);
static FP l1_wg_lbd[L1S][(L1S + L0S)] row_align(1);
static FP l1_wo_lbd[L1S][(L1S + L0S)] row_align(1);
static FP l1_bi_lbd[L1S] row_align(1);
static FP l1_bf_lbd[L1S] row_align(1);
static FP l1_bg_lbd[L1S] row_align(1);
static FP l1_bo_lbd[L1S] row_align(1);
static FP l2_w_lbd[L2S][L1S] row_align(1);
static FP l2_b_lbd[L2S] row_align(1);


static FP l1_wi_grad[L1S][(L1S + L0S)] row_align(1);
static FP l1_wf_grad[L1S][(L1S + L0S)] row_align(1);
static FP l1_wg_grad[L1S][(L1S + L0S)] row_align(1);
static FP l1_wo_grad[L1S][(L1S + L0S)] row_align(1);
static FP l1_bi_grad[L1S] row_align(1);
static FP l1_bf_grad[L1S] row_align(1);
static FP l1_bg_grad[L1S] row_align(1);
static FP l1_bo_grad[L1S] row_align(1);
static FP l2_w_grad[L2S][L1S] row_align(1);
static FP l2_b_grad[L2S] row_align(1);


static FP samples[BS][NUM_OF_P] row_align(1);
static FP xc[TS+1][BS][(L0S+L1S)] row_align(1);
static FP label[BS][L2S] row_align(1);

static FP l1_i_input[TS+1][BS][L1S] row_align(1);
static FP l1_f_input[TS+1][BS][L1S] row_align(1);
static FP l1_g_input[TS+1][BS][L1S] row_align(1);
static FP l1_o_input[TS+1][BS][L1S] row_align(1);

static FP l1_i[TS+1][BS][L1S] row_align(1);
static FP l1_f[TS+1][BS][L1S] row_align(1);
static FP l1_g[TS+1][BS][L1S] row_align(1);
static FP l1_o[TS+1][BS][L1S] row_align(1);
static FP l1_s[TS+1][BS][L1S] row_align(1);
static FP l1_s_tanh[TS+1][BS][L1S] row_align(1);
static FP l1_h[TS+1][BS][L1S] row_align(1);
static FP l2_h[TS+1][BS][L2S] row_align(1);
static FP l2_o[TS+1][BS][L2S] row_align(1);


static FP d_l2_h[BS][L2S] row_align(1);
static FP d_l1_h[BS][L1S] row_align(1);
static FP d_l1_s[BS][L1S] row_align(1);
static FP d_l1_i[BS][L1S] row_align(1);
static FP d_l1_f[BS][L1S] row_align(1);
static FP d_l1_g[BS][L1S] row_align(1);
static FP d_l1_o[BS][L1S] row_align(1);
static FP di_input[BS][L1S] row_align(1);
static FP df_input[BS][L1S] row_align(1);
static FP dg_input[BS][L1S] row_align(1);
static FP do_input[BS][L1S] row_align(1);

#endif// NETWORK_PARAMS_RV