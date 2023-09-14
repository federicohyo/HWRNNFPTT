/*
    In lstm.h, see the definition
*/
#include "lstm.h"

FP max_ini_v = sqrt(1/(FP)(L1S));
FP min_ini_v = -sqrt(1/(FP)(L1S)); 

FP l1_wi[L1_W_SIZE];
FP l1_wf[L1_W_SIZE];
FP l1_wg[L1_W_SIZE];
FP l1_wo[L1_W_SIZE];
FP l1_bi[L1_B_SIZE];
FP l1_bf[L1_B_SIZE];
FP l1_bg[L1_B_SIZE];
FP l1_bo[L1_B_SIZE];
FP l2_w[L2_W_SIZE];
FP l2_b[L2_B_SIZE];


FP l1_wi_rm[L1_W_SIZE];
FP l1_wf_rm[L1_W_SIZE];
FP l1_wg_rm[L1_W_SIZE];
FP l1_wo_rm[L1_W_SIZE];
FP l1_bi_rm[L1_B_SIZE];
FP l1_bf_rm[L1_B_SIZE];
FP l1_bg_rm[L1_B_SIZE];
FP l1_bo_rm[L1_B_SIZE];
FP l2_w_rm[L2_W_SIZE];
FP l2_b_rm[L2_B_SIZE];


FP l1_wi_lbd[L1_W_SIZE] = {0};
FP l1_wf_lbd[L1_W_SIZE] = {0};
FP l1_wg_lbd[L1_W_SIZE] = {0};
FP l1_wo_lbd[L1_W_SIZE] = {0};
FP l1_bi_lbd[L1_B_SIZE] = {0};
FP l1_bf_lbd[L1_B_SIZE] = {0};
FP l1_bg_lbd[L1_B_SIZE] = {0};
FP l1_bo_lbd[L1_B_SIZE] = {0};
FP l2_w_lbd[L2_W_SIZE] = {0};
FP l2_b_lbd[L2_B_SIZE] = {0};


FP l1_wi_grad[L1_W_SIZE] = {0};
FP l1_wf_grad[L1_W_SIZE] = {0};
FP l1_wg_grad[L1_W_SIZE] = {0};
FP l1_wo_grad[L1_W_SIZE] = {0};
FP l1_bi_grad[L1_B_SIZE] = {0};
FP l1_bf_grad[L1_B_SIZE] = {0};
FP l1_bg_grad[L1_B_SIZE] = {0};
FP l1_bo_grad[L1_B_SIZE] = {0};
FP l2_w_grad[L2_W_SIZE] = {0};
FP l2_b_grad[L2_B_SIZE] = {0};


void load_param_and_rm(FP* param, FP* rm, int size, char file_dir[], char file_name[])
{
    char file[50];  
    strcpy(file, file_dir);
    strcat(file, file_name);

    FILE* ptr = fopen(file, "r");
    
    if (ptr == NULL) {
        printf("No such file: %s.\n", file);
        exit(1);
    }

    for(int i=0; i<size; i++)
    {
        fscanf(ptr, "%f", &param[i]);
        // fscanf(ptr, "%f", &rm[i]);
        rm[i] = param[i];
    }

    fclose(ptr);
}

void load_all_param_and_rm(char file_dir[])
{

    load_param_and_rm(l1_wi, l1_wi_rm, L1_W_SIZE, file_dir, "param_l1_wi.txt");
    load_param_and_rm(l1_wf, l1_wf_rm, L1_W_SIZE, file_dir, "param_l1_wf.txt");
    load_param_and_rm(l1_wg, l1_wg_rm, L1_W_SIZE, file_dir, "param_l1_wg.txt");
    load_param_and_rm(l1_wo, l1_wo_rm, L1_W_SIZE, file_dir, "param_l1_wo.txt");

    load_param_and_rm(l1_bi, l1_bi_rm, L1_B_SIZE, file_dir, "param_l1_bi.txt");
    load_param_and_rm(l1_bf, l1_bf_rm, L1_B_SIZE, file_dir, "param_l1_bf.txt");
    load_param_and_rm(l1_bg, l1_bg_rm, L1_B_SIZE, file_dir, "param_l1_bg.txt");
    load_param_and_rm(l1_bo, l1_bo_rm, L1_B_SIZE, file_dir, "param_l1_bo.txt");

    load_param_and_rm(l2_w, l2_w_rm, L2_W_SIZE, file_dir, "param_l2_w.txt");
    load_param_and_rm(l2_b, l2_b_rm, L2_B_SIZE, file_dir, "param_l2_b.txt");
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
    // srand(time(NULL));
    srand(0); // specify the seed so that the reuslt can be re-produced

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


void forward(int time_steps, FP* input_samples)
{

}