#ifndef _UTIL_DEBUG_H_
#define _UTIL_DEBUG_H_

// configurations of the workload & the platform
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

        if(REGULARIZATION==1)
            printf("Regularization is also enabled. Thus complete FPTT (online+reg).\n");
        else
            printf("Regularization is NOT enabled\n");
    #else
        printf("NOT using online formulation\n");
    #endif

#ifdef PLATFORM_RV
    #ifndef RVMULTICORE
        printf("Running on single-core RISC-V based platform, and ");
    #else
        printf("Running on %d-core RISC-V based platform, and ", NCORES);
    #endif

    #ifdef USE_GEMMINI_LIB
        printf("one Gemmini accelerator is active\n");
    #else
        printf("Gemmini is NOT active\n");
    #endif
#endif

#ifdef PLATFORM_X86
    printf("Running on x86 based platform\n");
#endif

}

// memory usage of network parameters
// (defined as global static memory)
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




// some prints of network states to debug
// ignore


#ifndef ELEM_T_IS_LOWPREC_FLOAT // if the network is in FP32
void print_network_out(int t)
{
    printf("l1_i_input\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
            printf("%.8f  %X", l1_i_input[t][i][j], f2hex(l1_i_input[t][i][j]));
        printf("\n");
    }

    printf("l1_f_input\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
            printf("%.8f  %X", l1_f_input[t][i][j], f2hex(l1_f_input[t][i][j]));
        printf("\n");
    }

    printf("l1_g_input\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
            printf("%.8f  %X", l1_g_input[t][i][j], f2hex(l1_g_input[t][i][j]));
        printf("\n");
    }

    printf("l1_o_input\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
            printf("%.8f  %X", l1_o_input[t][i][j], f2hex(l1_o_input[t][i][j]));
        printf("\n");
    }

    printf("l1_h\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
            printf("%.8f  %X", l1_h[t][i][j], f2hex(l1_h[t][i][j]));
        printf("\n");
    }
    printf("l2_h\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
            printf("%.8f  %X", l2_h[t][i][j], f2hex(l2_h[t][i][j]));
        printf("\n");
    }

    printf("l2_o\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
            printf("%.8f  %X", l2_o[t][i][j], f2hex(l2_o[t][i][j]));
        printf("\n");
    }
}


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
	printf("\n\n");


	printf("l2_w: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", f2hex(l2_w[0][i]));
	printf("\n");

	printf("l2_b: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", f2hex(l2_b[i]));
	printf("\n");

	printf("l1_wo: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", f2hex(l1_wo[0][i]));
	printf("\n");

	printf("l1_bo: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", f2hex(l1_bo[i]));
	printf("\n");

	printf("l1_wi: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", f2hex(l1_wi[0][i]));
	printf("\n");

	printf("l1_bi: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", f2hex(l1_bi[i]));
	printf("\n");

	printf("l1_wf: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", f2hex(l1_wf[0][i]));
	printf("\n");

	printf("l1_bf: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", f2hex(l1_bf[i]));
	printf("\n");

	printf("l1_wg: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", f2hex(l1_wg[0][i]));
	printf("\n");

	printf("l1_bg: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", f2hex(l1_bg[i]));
	printf("\n");

}


void print_grad(int t)
{
    printf("t: %d, d_l1_h\t", t);
    for(int i=0; i<10; i++)
        printf("%.8f  %X", d_l1_h[0][i], f2hex(d_l1_h[0][i]));
    printf("\n");

    printf("t: %d, d_l1_o\t", t);
    for(int i=0; i<10; i++)
        printf("%.8f  %X", d_l1_o[0][i], f2hex(d_l1_o[0][i]));
    printf("\n");

    printf("t: %d, do_input\t", t);
    for(int i=0; i<10; i++)
        printf("%.8f  %X", do_input[0][i], f2hex(do_input[0][i]));
    printf("\n");

    printf("t: %d, l1_wo_grad\t", t);
    for(int i=0; i<10; i++)
        printf("%.8f  %X", l1_wo_grad[0][i], f2hex(l1_wo_grad[0][i]));
    printf("\n");

    printf("t: %d, l1_bo_grad\t", t);
    for(int i=0; i<10; i++)
        printf("%.8f  %X", l1_bo_grad[i], f2hex(l1_bo_grad[i]));
    printf("\n\n");
}


#else // if the network is in BF16
void print_network_out(int t)
{
    uint32_t tmp=0;
    printf("l1_i_input\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
        {
            tmp = u16_to_u32(l1_i_input[t][i][j]);
            printf("%.8f  ", u32_to_fp32(tmp));
        }
        printf("\n");
    }

    printf("l1_f_input\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
        {
            tmp = u16_to_u32(l1_f_input[t][i][j]);
            printf("%.8f  ", u32_to_fp32(tmp));
        }
        printf("\n");
    }

    printf("l1_g_input\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
        {
            tmp = u16_to_u32(l1_g_input[t][i][j]);
            printf("%.8f  ", u32_to_fp32(tmp));
        }
        printf("\n");
    }

    printf("l1_o_input\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
        {
            tmp = u16_to_u32(l1_o_input[t][i][j]);
            printf("%.8f  ", u32_to_fp32(tmp));
        }
        printf("\n");
    }

    printf("l1_h\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
        {
            tmp = u16_to_u32(l1_h[t][i][j]);
            printf("%.8f  ", u32_to_fp32(tmp));
        }
        printf("\n");
    }
    printf("l2_h\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
        {
            tmp = u16_to_u32(l2_h[t][i][j]);
            printf("%.8f  ", u32_to_fp32(tmp));
        }
        printf("\n");
    }

    printf("l2_o\n");
    for(int i=0; i<BS; i++)
    {
        printf("Sample no. %d: ", i);
        for(int j=0; j<L2S; j++)
        {
            tmp = u16_to_u32(l2_o[t][i][j]);
            printf("%.8f  ", u32_to_fp32(tmp));
        }
        printf("\n");
    }
}


void print_params_partly()
{
    uint32_t tmp=0;

	printf("l2_w: ");
	for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(l2_w[0][i]);
		printf("%.8f ", u32_to_fp32(tmp));
    }
	printf("\n");

	printf("l2_b: ");
	for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(l2_b[i]);
		printf("%.8f ", u32_to_fp32(tmp));
    }
	printf("\n");

	printf("l1_wo: ");
	for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(l1_wo[0][i]);
		printf("%.8f ", u32_to_fp32(tmp));
    }
	printf("\n");

	printf("l1_bo: ");
	for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(l1_bo[i]);
		printf("%.8f ", u32_to_fp32(tmp));
    }
	printf("\n");

	printf("l1_wi: ");
	for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(l1_wi[0][i]);
		printf("%.8f ", u32_to_fp32(tmp));
    }
	printf("\n");

	printf("l1_bi: ");
	for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(l1_bi[i]);
		printf("%.8f ", u32_to_fp32(tmp));
    }
	printf("\n");

	printf("l1_wf: ");
	for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(l1_wf[0][i]);
		printf("%.8f ", u32_to_fp32(tmp));
    }
	printf("\n");

	printf("l1_bf: ");
	for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(l1_bf[i]);
		printf("%.8f ", u32_to_fp32(tmp));
    }
	printf("\n");

	printf("l1_wg: ");
	for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(l1_wg[0][i]);
		printf("%.8f ", u32_to_fp32(tmp));
    }
	printf("\n");

	printf("l1_bg: ");
	for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(l1_bg[i]);
		printf("%.8f ", u32_to_fp32(tmp));
    }
	printf("\n\n");


	printf("l2_w: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", l2_w[0][i]);
	printf("\n");

	printf("l2_b: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", l2_b[i]);
	printf("\n");

	printf("l1_wo: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", l1_wo[0][i]);
	printf("\n");

	printf("l1_bo: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", l1_bo[i]);
	printf("\n");

	printf("l1_wi: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", l1_wi[0][i]);
	printf("\n");

	printf("l1_bi: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", l1_bi[i]);
	printf("\n");

	printf("l1_wf: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", l1_wf[0][i]);
	printf("\n");

	printf("l1_bf: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", l1_bf[i]);
	printf("\n");

	printf("l1_wg: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", l1_wg[0][i]);
	printf("\n");

	printf("l1_bg: ");
	for(int i=0; i<10; i++)
		printf("0x%X ", l1_bg[i]);
	printf("\n");

}

void print_grad(int t)
{
    uint32_t tmp=0; 

    printf("t: %d, d_l1_h\t", t);
    for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(d_l1_h[0][i]);
        printf("%.8f  ", u32_to_fp32(tmp));
    }
    printf("\n");

    printf("t: %d, d_l1_o\t", t);
    for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(d_l1_o[0][i]);
        printf("%.8f  ", u32_to_fp32(tmp));
    }
    printf("\n");

    printf("t: %d, do_input\t", t);
    for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(do_input[0][i]);
        printf("%.8f  ", u32_to_fp32(tmp));
    }
    printf("\n");

    printf("t: %d, l1_wo_grad\t", t);
    for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(l1_wo_grad[0][i]);
        printf("%.8f  ", u32_to_fp32(tmp));
    }
    printf("\n");

    printf("t: %d, l1_bo_grad\t", t);
    for(int i=0; i<10; i++)
    {
        tmp = u16_to_u32(l1_bo_grad[i]);
        printf("%.8f  ", u32_to_fp32(tmp));
    }
    printf("\n\n");
}
#endif





#endif // _UTIL_DEBUG_H_
