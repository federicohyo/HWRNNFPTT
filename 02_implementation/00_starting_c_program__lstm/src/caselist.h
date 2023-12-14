#ifndef _CASELIST_H_
#define _CASELIST_H_

#include "lstm.h"

// not the focus of this project
// being list here since it is the starting point
// see README.md the definition of offline
void learning_mnist_offline(char way_of_processing[], char dir_load_param[])
{

	char param_dir[50]; 


	/* specify the directory to load weights*/ 
	if(strcmp(way_of_processing,"streaming_by_rows")==0)
	{
		strcpy(param_dir, "../../data/params/28x128x10/");
		strcat(param_dir, dir_load_param);
		strcat(param_dir, "/");
	}
	else if(strcmp(way_of_processing,"streaming_by_pixels")==0)
	{
		strcpy(param_dir, "../../data/params/1x128x10/");
		strcat(param_dir, dir_load_param);
		strcat(param_dir, "/");
	}
	else
	{ printf("No such way of processing\n"); exit(1); }


#ifndef BM // specify this macro in the compilation command
	/* load weights and input samples*/
	load_all_param_and_rm(param_dir);
	load_input_samples_to_xc("../../data/input/samples.txt");
#endif

	printf("\noriginal parameters (partly)\n");
	print_params_partly();

	forward(TS);
	print_network_out(TS);
	backward(TS, TS, 2);
	optimizer_and_zero_grad(REGULARIZATION); 

	printf("\nupdated parameters (partly)\n");
	print_params_partly();
}


// see README.md the definition of offline
void learning_mnist_online(char way_of_processing[], char dir_load_param[], int reg_option)
{
	char param_dir[50]; 
  
	/* specify the directory to load weights*/ 
	if(strcmp(way_of_processing, "streaming_by_pixels")!=0)
	{ printf("Training online is only expected to be with streaming by pixels\n"); exit(1); }
	else
	{
		strcpy(param_dir, "../../data/params/1x128x10/");
		strcat(param_dir, dir_load_param);
		strcat(param_dir, "/");
	}

// load files from the external file system (linux)
// only when the platform is not baremetal
// specify this macro in the compilation command
#ifndef BM 
	printf("load parameters and data from files\n");
	/* load weights and input samples*/
	load_all_param_and_rm(param_dir);
	// initialize_all_param_and_rm();
	load_input_samples("../../data/input/samples.txt");
#endif

#ifdef PRINT_DEBUG
	printf("\noriginal parameters (partly)\n");
	print_params_partly();
#endif


#ifdef PRINT_PERF
	#ifdef PLATFORM_X86 
		clock_t begin = clock();
	#endif
	#ifdef PLATFORM_RV
		size_t sc = rdcycle();
	#endif
#endif

	// core computation
	// online formulation the original sequence is seen as K subsequences of TS (length)
	// T = TS * K 
	for(int i=0; i<K_VAL; i++)
	{
		#ifndef BM
		printf("K: %d\n", i);
		#endif
		load_sub_seq_to_xc(i);
		relay_network_states();
		forward(TS);
		backward(TS, TS, 2);
		optimizer_and_zero_grad(reg_option); 
	}

#ifdef PRINT_PERF
	#ifdef PLATFORM_X86 
		clock_t end = clock();
		double time_spent = ((double)(end - begin))/ 4e7;
		printf("[clock] cycles taken: %ld, Latency: %f\n", (end - begin), time_spent);
	#endif
	#ifdef PLATFORM_RV
		size_t ec = rdcycle();
		size_t cycles = ec -sc;
		printf("[rdcycle]: cycles taken: %ld, latency(in sec): %ld\n", cycles, (int)(((double)cycles)/4e7) );
	#endif
#endif


#ifdef PRINT_DEBUG
	print_network_out(TS);
	printf("\nupdated parameters (partly)\n");
	print_params_partly();
#endif

}



#endif//_CASELIST_H_
