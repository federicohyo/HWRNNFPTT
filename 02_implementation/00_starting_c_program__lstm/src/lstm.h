#ifndef _LSTM_H_
#define _LSTM_H_

#ifdef PLATFORM_X86
    #include "network_params_x86.h"
#endif
#ifdef PLATFORM_RV
    #include "network_params_riscv.h"
#endif

#include "matrix_ops.h"
#include "util_init.h"
#include "util_debug.h"



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
#ifndef RVMULTICORE
void forward(int seq_length)
#else
void forward(int cid, int seq_length)
#endif
{
    for(int t=1; t<=seq_length; t++)
    {
        // python code: self.xc[t] = np.hstack(( np.squeeze(xt[:, t-sp:t-sp+1, :], axis=1),  self.l1_h[t-1]))
        #ifdef RVMULTICORE
            if(cid==0) {
        #endif
            fill_l1_h_into_xc(t);
            // python code: self.l1_g_input[t] = np.dot(self.xc[t], self.param.l1_wg.T) + self.param.l1_bg
            mat_mul_b_T_add_bias( (FP*)&l1_g_input[t], (FP*)&xc[t], (FP*)l1_wg, BS, (L1S+L0S), L1S, (L1S+L0S), l1_bg);
            mat_mul_b_T_add_bias( (FP*)&l1_i_input[t], (FP*)&xc[t], (FP*)l1_wi, BS, (L1S+L0S), L1S, (L1S+L0S), l1_bi);
            mat_mul_b_T_add_bias( (FP*)&l1_f_input[t], (FP*)&xc[t], (FP*)l1_wf, BS, (L1S+L0S), L1S, (L1S+L0S), l1_bf);
            mat_mul_b_T_add_bias( (FP*)&l1_o_input[t], (FP*)&xc[t], (FP*)l1_wo, BS, (L1S+L0S), L1S, (L1S+L0S), l1_bo);
        #ifdef RVMULTICORE
            }
            barrier(NCORES);
        #endif
        
        #ifndef RVMULTICORE
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
        #else
            // // python code: self.l1_g[t] = np.tanh(self.l1_g_input[t])
            tanhf_on_matrix  (cid, (FP*)&l1_g[t], (FP*)&l1_g_input[t], BS, L1S);
            sigmoid_on_matrix(cid, (FP*)&l1_i[t], (FP*)&l1_i_input[t], BS, L1S);
            sigmoid_on_matrix(cid, (FP*)&l1_f[t], (FP*)&l1_f_input[t], BS, L1S);
            sigmoid_on_matrix(cid, (FP*)&l1_o[t], (FP*)&l1_o_input[t], BS, L1S);
            // // python code: self.l1_s[t] = self.l1_g[t] * self.l1_i[t] + self.l1_s[t-1] * self.l1_f[t]
            element_wise_mul(cid, (FP*)&l1_s[t], (FP*)&l1_g[t], (FP*)&l1_i[t], BS, L1S, "forward_pass");
            element_wise_mac(cid, (FP*)&l1_s[t], (FP*)&l1_s[t-1], (FP*)&l1_f[t], BS, L1S);
            // // python code: self.l1_h[t] = np.tanh(self.l1_s[t]) * self.l1_o[t]
            tanhf_on_matrix(cid, (FP*)&l1_s_tanh[t], (FP*)l1_s[t], BS, L1S);
            element_wise_mul(cid, (FP*)&l1_h[t], (FP*)&l1_s_tanh[t], (FP*)&l1_o[t], BS, L1S, "forward_pass");
        #endif

        #ifdef RVMULTICORE
            if(cid==0){ 
        #endif
            // python code: self.l2_h[t] = np.dot(self.l1_h[t], self.param.l2_w.T) + self.param.l2_b 
            mat_mul_b_T_add_bias( (FP*)&l2_h[t], (FP*)&l1_h[t], (FP*)l2_w, BS, L1S, L2S, L1S, l2_b);
            softmax( (FP*)&l2_o[t], (FP*)&l2_h[t], BS, L2S);
        #ifdef RVMULTICORE
            }
            barrier(NCORES);
        #endif
    }
}




// helper function: used only in [backward]
#ifndef RVMULTICORE
void find_ds(int t, int row, int col)
#else
void find_ds(int cid, int t, int row, int col)
#endif
{

#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t start = rdcycle();
    #else
        size_t start;
        if(cid==0)
            start = rdcycle();
        // barrier(NCORES);
    #endif
#endif

    // find the derivative of Loss w.r.t l1_s at time step t
//     for(int i=0; i<row; i++)
//         for(int j=0; j<col; j++)
//         {
// #ifndef ELEM_T_IS_LOWPREC_FLOAT
//          d_l1_s[i][j] = l1_o[t][i][j] * (1 - (l1_s_tanh[t][i][j])*(l1_s_tanh[t][i][j]) ) * d_l1_h[i][j];
// #else
//          d_l1_s[i][j] = bf16_mul(bf16_mul(l1_o[t][i][j], bf16_sub(0x3F80, bf16_mul(l1_s_tanh[t][i][j], l1_s_tanh[t][i][j]))), d_l1_h[i][j]);
// #endif
//         }

    //python code: self.ds = self.state.l1_o[h_step] * (1 - (np.tanh(self.state.l1_s[h_step]))**2 ) *self.dl1_h
    int base=t*row*col;
    #ifndef RVMULTICORE
    for(int i=0; i<row*col; i++)
    #else
    for(int i=cid; i<row*col; i+=NCORES)
    #endif
    {
        #ifndef ELEM_T_IS_LOWPREC_FLOAT
        M2O(d_l1_s)[i] = M2O(l1_o)[base+i] * (1 - (M2O(l1_s_tanh)[base+i]*M2O(l1_s_tanh)[base+i])) * M2O(d_l1_h)[i];
        #else
        M2O(d_l1_s)[i] = bf16_mul(bf16_mul(M2O(l1_o)[base+i], bf16_sub(0x3F80, bf16_mul(M2O(l1_s_tanh)[base+i], M2O(l1_s_tanh)[base+i]))), M2O(d_l1_h)[i]);
        #endif
    }

#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t end = rdcycle();
        acc_find_ds += (end - start);
    #else
        size_t end;
        if(cid==0)
        {
            end = rdcycle();
            acc_find_ds += (end - start);
        }
        // barrier(NCORES);
    #endif
#endif

#ifdef PRINT_COUNTER
    #ifdef RVMULTICORE
    if(cid==0)
    {
    #endif
        // non-functional: operation counter below
        bp_mul += row*col*3;
        bp_sub += row*col;
    #ifdef RVMULTICORE
    }
    #endif
#endif
}

// helper function: used only in [backward]
#ifndef RVMULTICORE
void find_d_l1_ifgo_input(FP* dst, FP* src_a, FP* src_b, int row, int col, int check_g)
#else
void find_d_l1_ifgo_input(int cid, FP* dst, FP* src_a, FP* src_b, int row, int col, int check_g)
#endif
{

#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t start = rdcycle();
    #else
        size_t start;
        if(cid==0)
            start = rdcycle();
        // barrier(NCORES);
    #endif
#endif


if(check_g==1)
    #ifndef RVMULTICORE
    for(int i=0; i<row*col; i++)
    #else
    for(int i=cid; i<row*col; i+=NCORES)
    #endif
    {
        #ifndef ELEM_T_IS_LOWPREC_FLOAT
            dst[i] = (1.0 - (src_a[i]*src_a[i])) * src_b[i]; // derivative of tanh
        #else
            dst[i] = bf16_mul(bf16_sub(0x3F80, bf16_mul(src_a[i], src_a[i])), src_b[i]);
        #endif
    }
else
    #ifndef RVMULTICORE
    for(int i=0; i<row*col; i++)
    #else
    for(int i=cid; i<row*col; i+=NCORES)
    #endif
    {
        #ifndef ELEM_T_IS_LOWPREC_FLOAT
            dst[i] = src_a[i] * (1.0 - src_a[i]) * src_b[i]; // derivative of tanh
        #else
            dst[i] = bf16_mul(bf16_mul(src_a[i], bf16_sub(0x3F80, src_a[i])), src_b[i]);
        #endif
    }

#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t end = rdcycle();
        acc_find_d_l1_ifgo_input += (end - start);
    #else
        size_t end;
        if(cid==0)
        {
            end = rdcycle();
            acc_find_d_l1_ifgo_input += (end - start);
        }
        // barrier(NCORES);
    #endif
#endif

#ifdef PRINT_COUNTER
    #ifdef RVMULTICORE
    if(cid==0)
    {
    #endif
        // non-functional: operation counter below
        bp_mul += row*col*2;
        bp_sub += row*col;
        // index calculation
        index_mul += row*col;
        index_add += row*col;
    #ifdef RVMULTICORE
    }
    #endif
#endif
}

// helper function: used only in [backward]
void update_d_l1_h()
{

#ifdef PRINT_PERF
size_t start = rdcycle();
#endif

    mat_mul((FP*)tmp1, (FP*)di_input, (FP*)l1_wi, BS, L1S, L1S, (L0S+L1S));
    mat_mul((FP*)tmp2, (FP*)df_input, (FP*)l1_wf, BS, L1S, L1S, (L0S+L1S));
    mat_mul((FP*)tmp3, (FP*)dg_input, (FP*)l1_wg, BS, L1S, L1S, (L0S+L1S));
    mat_mul((FP*)tmp4, (FP*)do_input, (FP*)l1_wo, BS, L1S, L1S, (L0S+L1S));

    for(int i=0; i<BS; i++)
        for(int j=L0S; j<L0S+L1S; j++)
        {
            #ifndef ELEM_T_IS_LOWPREC_FLOAT
                d_l1_h[i][j-L0S] = tmp1[i][j] +  tmp2[i][j] + tmp3[i][j] + tmp4[i][j]; 
            #else
                d_l1_h[i][j-L0S] = bf16_add(bf16_add(tmp1[i][j], tmp2[i][j]), bf16_add(tmp3[i][j], tmp4[i][j]));
            #endif
        }

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




// Backpropagation from the given time step t
#ifndef RVMULTICORE
void backward(int t, int trunc_h, int trunc_s)
#else
void backward(int cid, int t, int trunc_h, int trunc_s)
#endif
{
    int h_ep; // earliest point(time step) to end up backtracking for l1_h
    int s_ep; // earliest point(time step) to end up backtracking for l1_s

    #ifndef RVMULTICORE
        element_wise_sub( (FP*)d_l2_h, (FP*)&l2_o[t], (FP*)labels, BS, L2S);
        // python code: self.param.l2_w_diff =  (np.dot(self.dl2_h.T, self.state.l1_h[t]))/self.n_samples
        mat_mul_a_T_average( (FP*)l2_w_grad, (FP*)d_l2_h, (FP*)&l1_h[t], BS, L2S, BS, L1S, BS);
        // python code: self.param.l2_b_diff = (self.dl2_h.sum(axis=0))/self.n_samples # (10) = (200,10) 
        mat2vec_avr_sequeeze( (FP*)l2_b_grad, (FP*)d_l2_h, BS, L2S);
        // python code: self.dl1_h = np.dot(self.dl2_h, self.param.l2_w) # (200,128) = (200,10).(10, 128)
        mat_mul( (FP*)d_l1_h, (FP*)d_l2_h, (FP*)l2_w, BS, L2S, L2S, L1S);
        h_ep = (t-trunc_h>0) ? t-trunc_h : 0;
    #else
    element_wise_sub(cid, (FP*)d_l2_h, (FP*)&l2_o[t], (FP*)labels, BS, L2S);
    if(cid==0) // only core0 do these
    {
        // python code: self.param.l2_w_diff =  (np.dot(self.dl2_h.T, self.state.l1_h[t]))/self.n_samples
        mat_mul_a_T_average( (FP*)l2_w_grad, (FP*)d_l2_h, (FP*)&l1_h[t], BS, L2S, BS, L1S, BS);
        // python code: self.param.l2_b_diff = (self.dl2_h.sum(axis=0))/self.n_samples # (10) = (200,10) 
        mat2vec_avr_sequeeze( (FP*)l2_b_grad, (FP*)d_l2_h, BS, L2S);
        // python code: self.dl1_h = np.dot(self.dl2_h, self.param.l2_w) # (200,128) = (200,10).(10, 128)
        mat_mul( (FP*)d_l1_h, (FP*)d_l2_h, (FP*)l2_w, BS, L2S, L2S, L1S);
        h_ep = (t-trunc_h>0) ? t-trunc_h : 0;
    }
    barrier(NCORES);
    #endif


    for(int h_step=t; h_step>h_ep; h_step--)
    {

        #ifndef RVMULTICORE
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
        #else
            find_ds(cid, h_step, BS, L1S);
            // python: self.do = np.tanh(self.state.l1_s[h_step]) * self.dl1_h
            element_wise_mul(cid, (FP*)d_l1_o, (FP*)&l1_s_tanh[h_step], (FP*)d_l1_h, BS, L1S, "backward_pass");
            // python: self.do_input = sigmoid_derivative(self.state.l1_o[h_step]) * self.do 
            find_d_l1_ifgo_input(cid, (FP*)do_input, (FP*)&l1_o[h_step], (FP*)d_l1_o, BS, L1S, 0);
            if(cid==0)
            {
                // python: self.param.l1_wo_diff += (np.dot(self.do_input.T, self.state.xc[h_step])) /self.n_samples
                mat_mul_a_T_average( (FP*)l1_wo_grad, (FP*)do_input, (FP*)&xc[h_step], BS, L1S, BS, (L1S+L0S), BS);
                // python: self.param.l1_bo_diff += (self.do_input.sum(axis=0)) /self.n_samples # (128) = (200,128)
                mat2vec_avr_sequeeze( (FP*)l1_bo_grad, (FP*)do_input, BS, L1S);
                // dbg_l1_w_b_o(h_step);
                // python: s_ep = 0 if trunc_s is None else max(0, h_step -trunc_s)
                s_ep = (h_step-trunc_s>0) ? h_step-trunc_s : 0;
            }
            barrier(NCORES);
        #endif
        
        
        // python: for s_step in np.arange(s_ep, h_step+1)[::-1]:
        for(int s_step=h_step; s_step>s_ep; s_step--)
        {
            #ifndef RVMULTICORE
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
            #else
                element_wise_mul(cid, (FP*)d_l1_g, (FP*)&l1_i[s_step], (FP*)d_l1_s, BS, L1S, "backward_pass");
                element_wise_mul(cid, (FP*)d_l1_i, (FP*)&l1_g[s_step], (FP*)d_l1_s, BS, L1S, "backward_pass");
                element_wise_mul(cid, (FP*)d_l1_f, (FP*)&l1_s[s_step-1], (FP*)d_l1_s, BS, L1S, "backward_pass");
                // self.di_input = sigmoid_derivative(self.state.l1_i[s_step]) * self.di 
                find_d_l1_ifgo_input(cid, (FP*)di_input, (FP*)&l1_i[s_step], (FP*)d_l1_i, BS, L1S, 0);
                find_d_l1_ifgo_input(cid, (FP*)df_input, (FP*)&l1_f[s_step], (FP*)d_l1_f, BS, L1S, 0);
                find_d_l1_ifgo_input(cid, (FP*)dg_input, (FP*)&l1_g[s_step], (FP*)d_l1_g, BS, L1S, 1);
                // self.param.l1_wi_diff += (np.dot(self.di_input.T, self.state.xc[s_step])) /self.n_samples
                if(cid==0)
                {
                    mat_mul_a_T_average( (FP*)l1_wi_grad, (FP*)di_input, (FP*)&xc[s_step], BS, L1S, BS, (L1S+L0S), BS);
                    mat_mul_a_T_average( (FP*)l1_wf_grad, (FP*)df_input, (FP*)&xc[s_step], BS, L1S, BS, (L1S+L0S), BS);
                    mat_mul_a_T_average( (FP*)l1_wg_grad, (FP*)dg_input, (FP*)&xc[s_step], BS, L1S, BS, (L1S+L0S), BS);
                    mat2vec_avr_sequeeze( (FP*)l1_bi_grad, (FP*)di_input, BS, L1S);
                    mat2vec_avr_sequeeze( (FP*)l1_bf_grad, (FP*)df_input, BS, L1S);
                    mat2vec_avr_sequeeze( (FP*)l1_bg_grad, (FP*)dg_input, BS, L1S);
                }
                barrier(NCORES);
                // self.ds= self.ds* self.state.l1_f[s_step]
                element_wise_mul(cid, (FP*)d_l1_s, (FP*)d_l1_s, (FP*)&l1_f[s_step], BS, L1S, "backward_pass");
                if(cid==0){
                    if(h_step == s_step)
                        update_d_l1_h();
                }
                barrier(NCORES);
            #endif

        }
        #ifdef RVMULTICORE
        barrier(NCORES);
        #endif

        // print_grad(h_step);
    }
}

#ifndef RVMULTICORE
void SGD(FP* param, FP* grad, int row, int col)
#else
void SGD(int cid, FP* param, FP* grad, int row, int col)
#endif
{
#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t start = rdcycle();
    #else
        size_t start;
        if(cid==0)
            start = rdcycle();
        // barrier(NCORES);
    #endif
#endif

#ifndef RVMULTICORE // x86 or single-core RV
    for(int i=0; i<row*col; i++)
#else // multi-core RV
    for(int i=cid; i<row*col; i+=NCORES)
#endif
    {
        #ifndef ELEM_T_IS_LOWPREC_FLOAT
            param[i] -= (LR * grad[i]);
            grad[i] = 0;
        #else
            param[i] = bf16_sub(param[i], bf16_mul(LR, grad[i]) );
            grad[i] = 0;
        #endif
    }
#ifdef RVMULTICORE
    barrier(NCORES);
#endif

#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t end = rdcycle();
        acc_SGD += (end - start);
    #else
        size_t end;
        if(cid==0)
        {
            end = rdcycle();
            acc_SGD += (end - start);
        }
        // barrier(NCORES);
    #endif
#endif

#ifdef PRINT_COUNTER
    #ifdef RVMULTICORE
    if(cid==0)
    {
    #endif
        fptt_sub += row*col;
        fptt_mul += row*col;
        // index calculation
        index_mul += row*col;
        index_add += row*col;
    #ifdef RVMULTICORE
    }
    #endif
#endif 
}

#ifndef RVMULTICORE
void FPTT_SGD(FP* param, FP* grad, FP* rmean, FP* lbd, int row, int col)
#else
void FPTT_SGD(int cid, FP* param, FP* grad, FP* rmean, FP* lbd, int row, int col)
#endif
{
    // python
    // self.l1_wg -= lr * (self.l1_wg_diff - self.l1_lbd_wg + alpha*(self.l1_wg - self.l1_rm_wg))
    // self.l1_lbd_wg -= alpha*(self.l1_wg - self.l1_rm_wg)
    // self.l1_rm_wg = 0.5*(self.l1_rm_wg + self.l1_wg) - (0.5/alpha)*self.l1_lbd_wg
    // self.l1_wg_diff = np.zeros_like(self.l1_wg)
#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t start = rdcycle();
    #else
        size_t start;
        if(cid==0)
            start = rdcycle();
        // barrier(NCORES);
    #endif
#endif

#ifndef RVMULTICORE // x86 or single-core RV
    for(int i=0; i<row*col; i++)
#else // multi-core RV
    for(int i=cid; i<row*col; i+=NCORES)
#endif
    {
        #ifndef ELEM_T_IS_LOWPREC_FLOAT
            param[i] -= LR * (grad[i] - lbd[i] + ALPHA*(param[i] - rmean[i]) );
            lbd[i] -= ALPHA * (param[i] - rmean[i]);
            rmean[i] = 0.5*(rmean[i] + param[i]) - (0.5/ALPHA)*lbd[i];
            grad[i] = 0;
        #else
            param[i] = bf16_sub(param[i], bf16_mul(LR, bf16_add(bf16_sub(grad[i],lbd[i]), bf16_mul(ALPHA, bf16_sub(param[i], rmean[i])) )) );
            lbd[i] = bf16_sub(lbd[i], bf16_mul(ALPHA, bf16_sub(param[i], rmean[i])));
            rmean[i] = bf16_sub(bf16_mul(P1, bf16_add(rmean[i], param[i])), bf16_mul(P2, lbd[i]));
            grad[i] = 0;
        #endif
    }

#ifdef RVMULTICORE
    barrier(NCORES);
#endif

#ifdef PRINT_PERF
    #ifndef RVMULTICORE
        size_t end = rdcycle();
        acc_FPTT_SGD += (end - start);
    #else
        size_t end;
        if(cid==0)
        {
            end = rdcycle();
            acc_FPTT_SGD += (end - start);
        }
        // barrier(NCORES);
    #endif
#endif

#ifdef PRINT_COUNTER
    #ifdef RVMULTICORE
    if(cid==0)
    {
    #endif
        fptt_add += row*col*(2);
        fptt_sub += row*col*(6);
        fptt_mul += row*col*(5);
        // index calculation
        index_mul += row*col;
        index_add += row*col;
    #ifdef RVMULTICORE
    }
    #endif
#endif
}

#ifndef RVMULTICORE
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
#else
void optimizer_and_zero_grad(int cid, int fptt_option)
{
    if(fptt_option==1)
    {
        FPTT_SGD(cid, (FP*)l1_wi, (FP*)l1_wi_grad, (FP*)l1_wi_rm, (FP*)l1_wi_lbd, L1S, (L1S+L0S));
        FPTT_SGD(cid, (FP*)l1_wf, (FP*)l1_wf_grad, (FP*)l1_wf_rm, (FP*)l1_wf_lbd, L1S, (L1S+L0S));
        FPTT_SGD(cid, (FP*)l1_wg, (FP*)l1_wg_grad, (FP*)l1_wg_rm, (FP*)l1_wg_lbd, L1S, (L1S+L0S));
        FPTT_SGD(cid, (FP*)l1_wo, (FP*)l1_wo_grad, (FP*)l1_wo_rm, (FP*)l1_wo_lbd, L1S, (L1S+L0S));

        FPTT_SGD(cid, (FP*)l1_bi, (FP*)l1_bi_grad, (FP*)l1_bi_rm, (FP*)l1_bi_lbd, 1, L1S);
        FPTT_SGD(cid, (FP*)l1_bf, (FP*)l1_bf_grad, (FP*)l1_bf_rm, (FP*)l1_bf_lbd, 1, L1S);
        FPTT_SGD(cid, (FP*)l1_bg, (FP*)l1_bg_grad, (FP*)l1_bg_rm, (FP*)l1_bg_lbd, 1, L1S);
        FPTT_SGD(cid, (FP*)l1_bo, (FP*)l1_bo_grad, (FP*)l1_bo_rm, (FP*)l1_bo_lbd, 1, L1S);

        FPTT_SGD(cid, (FP*)l2_w, (FP*)l2_w_grad, (FP*)l2_w_rm, (FP*)l2_w_lbd, L2S, L1S);
        FPTT_SGD(cid, (FP*)l2_b, (FP*)l2_b_grad, (FP*)l2_b_rm, (FP*)l2_b_lbd, 1, L2S);
    }
    else
    {
        SGD(cid, (FP*)l1_wi, (FP*)l1_wi_grad, L1S, (L1S+L0S));
        SGD(cid, (FP*)l1_wf, (FP*)l1_wf_grad, L1S, (L1S+L0S));
        SGD(cid, (FP*)l1_wg, (FP*)l1_wg_grad, L1S, (L1S+L0S));
        SGD(cid, (FP*)l1_wo, (FP*)l1_wo_grad, L1S, (L1S+L0S));

        SGD(cid, (FP*)l1_bi, (FP*)l1_bi_grad, 1, L1S);
        SGD(cid, (FP*)l1_bf, (FP*)l1_bf_grad, 1, L1S);
        SGD(cid, (FP*)l1_bg, (FP*)l1_bg_grad, 1, L1S);
        SGD(cid, (FP*)l1_bo, (FP*)l1_bo_grad, 1, L1S);

        SGD(cid, (FP*)l2_w, (FP*)l2_w_grad, L2S, L1S);
        SGD(cid, (FP*)l2_b, (FP*)l2_b_grad, 1, L2S);
    }
}
#endif


void cross_entropy()
{
    float loss=0; // loss is always represented in fp32
    uint32_t tmp_out=0;
    uint32_t tmp_exp=0;
    
    for(int i=0; i<BS; i++)
        for(int j=0; j<L2S; j++)
        {
            #ifndef ELEM_T_IS_LOWPREC_FLOAT
                loss -= log(l2_o[TS][i][j])*labels[i][j];
            #else
                tmp_out = u16_to_u32(l2_o[TS][i][j]);
                tmp_exp = u16_to_u32(labels[i][j]); 

                loss -= log(u32_to_fp32(tmp_out))*u32_to_fp32(tmp_exp); 
            #endif
        }
    
    printf("Cross Entropy loss: %f (0x%X)\n", loss, *(uint32_t*)&loss);
}



#endif //_LSTM_H_