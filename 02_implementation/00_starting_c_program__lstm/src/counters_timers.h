#ifndef _COUNTERS_TIMERS_
#define _COUNTERS_TIMERS_

//--------------------------------------
// Performance Counter
// fp: forward path
// bp: backward path
//--------------------------------------
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

size_t acc_mat_mul = 0;
size_t acc_mat_mul_b_T = 0;
size_t acc_mat_mul_a_T = 0;

size_t acc_mat2vec_avr_sequeeze =0;
size_t acc_element_wise_mul =0;
size_t acc_element_wise_mac =0;
size_t acc_element_wise_sub =0;
size_t acc_tanhf_on_matrix =0;
size_t acc_sigmoid_on_matrix =0;
size_t acc_softmax =0;
size_t acc_total=0;

size_t acc_load_sub_seq_to_xc =0;
size_t acc_relay_network_states=0;
size_t acc_fill_l1_h_into_xc=0;
size_t acc_find_ds =0;
size_t acc_find_d_l1_ifgo_input=0;
size_t acc_update_d_l1_h=0;
size_t acc_SGD=0;
size_t acc_FPTT_SGD=0;
size_t acc_optimizer=0;

#endif