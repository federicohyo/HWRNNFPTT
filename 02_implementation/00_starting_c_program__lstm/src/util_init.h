#ifndef _UTIL_INIT_H_
#define _UTIL_INIT_H_

/*
    This header file handles
    1. initialization of the network parameters (weights/biases), with three options available
        a. [run-time, need OS] load from external text files
        b. [run-time, for OS/baremetal] randomize in-suit
        c. [compile-time, for baremetal/OS] use predefined data array, see directory data/c_headers
        
    2. loading of network input (samples) and labels
        a. [run-time, need OS] load from external text files
        b. [compile-time, for baremetal/OS] use predefined data array, see directory data/c_headers

    for case 1.a & 1.b
        network parameters are the same, from a pre-trained PyTorch LSTM Model
*/



/*------------------------------
    initialize parameters: 1.a
------------------------------*/
#ifndef BM // to run these functions, OS is necessary
void load_param_and_rm(FP* param, FP* rm, int row, int col, char file_dir[], char file_name[])
{
    char file[50];  
    strcpy(file, file_dir);
    strcat(file, file_name);

    float tmp=0;
    FILE* ptr = fopen(file, "r");
   
    if (ptr == NULL) 
    {
        printf("[load_param_and_rm]: No such file: %s.\n", file);    exit(1);
    }

    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
        {
            fscanf(ptr, "%f", &tmp);

            #ifndef ELEM_T_IS_LOWPREC_FLOAT
                param[i*col +j] = tmp;
                rm[i*col +j] = tmp; 
            #else
                param[i*col +j] = fp32_to_u16(tmp);
                rm[i*col +j] = fp32_to_u16(tmp);
            #endif
        }
    fclose(ptr);
}

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


/*------------------------------
    initialize parameters: 1.b
------------------------------*/
// DATA DEFINITIONS 
float max_ini_v = sqrt(1/(float)(L1S));
float min_ini_v = -sqrt(1/(float)(L1S)); 

void randomize_param_and_rm(FP* input_param, FP* input_rm, int row, int col, float min_val, float max_val)
{
    for(int i = 0; i < row; i++) 
        for(int j = 0; j < col; j++) 
        {
            // Scale the random integer to fit within the desired range
            float scaledRand = ((float)rand() / (float)(RAND_MAX) ) * (max_val - min_val) + min_val;

            // Assign the scaled random value to the array element
            #ifndef ELEM_T_IS_LOWPREC_FLOAT
                input_param[i*col + j] = scaledRand;
                input_rm[i*col + j] = scaledRand;
            #else
                input_param[i*col + j] = fp32_to_u16(scaledRand);
                input_rm[i*col + j] = fp32_to_u16(scaledRand);
            #endif
       }
} 

void randomize_all_param_and_rm()
{
    // srand(time(NULL));
    srand(0); // specify the seed so that the reuslt can be re-produced

    randomize_param_and_rm( (FP*)l1_wi, (FP*)l1_wi_rm, L1S, (L1S + L0S), max_ini_v, min_ini_v);
    randomize_param_and_rm( (FP*)l1_wf, (FP*)l1_wf_rm, L1S, (L1S + L0S), max_ini_v, min_ini_v);
    randomize_param_and_rm( (FP*)l1_wg, (FP*)l1_wg_rm, L1S, (L1S + L0S), max_ini_v, min_ini_v);
    randomize_param_and_rm( (FP*)l1_wo, (FP*)l1_wo_rm, L1S, (L1S + L0S), max_ini_v, min_ini_v);

    randomize_param_and_rm( l1_bi, l1_bi_rm, 1, L1S, max_ini_v, min_ini_v);
    randomize_param_and_rm( l1_bf, l1_bf_rm, 1, L1S, max_ini_v, min_ini_v);
    randomize_param_and_rm( l1_bg, l1_bg_rm, 1, L1S, max_ini_v, min_ini_v);
    randomize_param_and_rm( l1_bo, l1_bo_rm, 1, L1S, max_ini_v, min_ini_v);

    randomize_param_and_rm( (FP*)l2_w, (FP*)l2_w_rm, L2S, L1S, max_ini_v, min_ini_v);
    randomize_param_and_rm( l2_b, l2_b_rm, 1, L2S, max_ini_v, min_ini_v);
}



/*------------------------------
    initialize parameters: 1.c
------------------------------*/
#ifdef BM

#include "baremetal_initialization.h"

void bm_initialize_param_and_rm(FP* param, FP* rmean, float* src, int dim1, int dim2)
{

    for(int i=0; i<dim1; i++)
        for(int j=0; j<dim2; j++)
        {
            #ifndef ELEM_T_IS_LOWPREC_FLOAT
                param[i*dim2+j] = src[i*dim2+j];
                rmean[i*dim2+j] = src[i*dim2+j];
            #else 
                param[i*dim2+j] = fp32_to_u16(src[i*dim2+j]);
                rmean[i*dim2+j] = fp32_to_u16(src[i*dim2+j]);
            #endif
        }
}

void bm_initialize_all_param_and_rm()
{
    bm_initialize_param_and_rm((FP*)l1_wi, (FP*)l1_wi_rm, param_l1_wi, L1S, (L0S+L1S));
    bm_initialize_param_and_rm((FP*)l1_wf, (FP*)l1_wf_rm, param_l1_wf, L1S, (L0S+L1S));
    bm_initialize_param_and_rm((FP*)l1_wg, (FP*)l1_wg_rm, param_l1_wg, L1S, (L0S+L1S));
    bm_initialize_param_and_rm((FP*)l1_wo, (FP*)l1_wo_rm, param_l1_wo, L1S, (L0S+L1S));

    bm_initialize_param_and_rm((FP*)l1_bi, (FP*)l1_bi_rm, param_l1_bi, 1, L1S);
    bm_initialize_param_and_rm((FP*)l1_bf, (FP*)l1_bf_rm, param_l1_bf, 1, L1S);
    bm_initialize_param_and_rm((FP*)l1_bg, (FP*)l1_bg_rm, param_l1_bg, 1, L1S);
    bm_initialize_param_and_rm((FP*)l1_bo, (FP*)l1_bo_rm, param_l1_bo, 1, L1S);

    bm_initialize_param_and_rm((FP*)l2_w, (FP*)l2_w_rm, param_l2_w, L2S, L1S);
    bm_initialize_param_and_rm((FP*)l2_b, (FP*)l2_b_rm, param_l2_b, 1, L2S);
}

#endif



/*------------------------------
    load samples & labels: 2.a
------------------------------*/
#ifndef BM // to run these functions, OS is necessary
void load_input_samples(char file_name[])
{
    float tmp=0;
    FILE* ptr = fopen(file_name, "r");

    if (ptr == NULL) 
    { printf("[load_input_samples]: No such file: %s.\n", file_name);    exit(1); }

    for(int i=0; i<BS; i++)
        for(int j=0; j<NUM_OF_P; j++)
            {
                fscanf(ptr, "%f", &tmp);
                #ifndef ELEM_T_IS_LOWPREC_FLOAT
                    samples[i][j] = tmp;
                #else
                    samples[i][j] = fp32_to_u16(tmp);
                #endif
            }

    fclose(ptr);
}

void load_labels(char file_name[])
{
    float tmp=0;
    FILE* ptr = fopen(file_name, "r");
    
    if (ptr == NULL) 
    { printf("[load_labels]: No such file: %s.\n", file_name);    exit(1); }

    for(int i=0; i<BS; i++)
        for(int j=0; j<L2S; j++)
            {
                fscanf(ptr, "%f", &tmp);
                #ifndef ELEM_T_IS_LOWPREC_FLOAT
                    labels[i][j] = tmp;
                #else
                    labels[i][j] = fp32_to_u16(tmp);
                #endif
            }

    fclose(ptr);
}
#endif

/*------------------------------
    load samples & labels: 2.b
------------------------------*/
#ifdef BM
void bm_load_input_samples(FP* dst, float* src)
{
    for(int i=0; i<BS; i++)
        for(int j=0; j<NUM_OF_P; j++)
        {
            #ifndef ELEM_T_IS_LOWPREC_FLOAT
                dst[i*NUM_OF_P+ j] = src[i*NUM_OF_P+ j];
            #else
                dst[i*NUM_OF_P+ j] = fp32_to_u16(src[i*NUM_OF_P+ j]);
            #endif
        }
}

void bm_load_labels(FP* dst, float* src)
{
    for(int i=0; i<BS; i++)
        for(int j=0; j<L2S; j++)
        {
            #ifndef ELEM_T_IS_LOWPREC_FLOAT
                dst[i*L2S + j] = src[i*L2S + j];
            #else
                dst[i*L2S + j] = fp32_to_u16(src[i*L2S + j]);
            #endif
        }

}
#endif

void initialization(char dir_load_param[])
{
    static char param_dir[50]; 

	strcpy(param_dir, "../../data/params/1x128x10/");
	strcat(param_dir, dir_load_param);
	strcat(param_dir, "/");

	#ifndef BM 
		printf("load parameters and data from files\n");
		load_all_param_and_rm(param_dir);
		load_input_samples("../../data/input/samples.txt");
		load_labels("../../data/input/labels.txt");
	#else
		bm_initialize_all_param_and_rm();
		bm_load_input_samples((FP*)samples, samples_arr);
		bm_load_labels((FP*)labels, labels_arr);
	#endif
		// randomize_all_param_and_rm();
}

#endif //_UTIL_INIT_H_