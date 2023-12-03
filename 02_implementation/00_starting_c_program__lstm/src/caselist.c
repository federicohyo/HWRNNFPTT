/*
    In the directory include/caselist.h
    see data/function declaration 
*/

#include "caselist.h"




void learning_mnist_offline(char way_of_processing[], char dir_load_param[], char reg_option[])
{

	char param_dir[50]; 
	int if_calculate_regularizer=0;

	if(strcmp(reg_option, "calculate_regularizer")==0)
		if_calculate_regularizer=1;


	if(strcmp(way_of_processing,"streaming_by_rows")==0)
	{
		strcpy(param_dir, "./data/params/28x128x10/");
		strcat(param_dir, dir_load_param);
		strcat(param_dir, "/");
	}
	else if(strcmp(way_of_processing,"streaming_by_pixels")==0)
	{
		strcpy(param_dir, "./data/params/1x128x10/");
		strcat(param_dir, dir_load_param);
		strcat(param_dir, "/");
	}
	else
	{
		printf("No such way of processing\n"); exit(1);
	}
	printf("\n%s\nif_calculate_regularizer: %d\n", param_dir, if_calculate_regularizer);

	load_all_param_and_rm(param_dir);
	load_input_samples_to_xc("./data/input/samples.txt");
	print_params_partly();

	forward(TS);
	print_network_out(TS);

	backward(TS, TS, 3);
	optimizer_and_zero_grad(if_calculate_regularizer); 

	printf("\nprint updated parameters\n");
	print_params_partly();
}

void learning_mnist_online(char way_of_processing[], char dir_load_param[], char reg_option[])
{
	char param_dir[50]; 
	int if_calculate_regularizer=0;

	if(strcmp(reg_option, "calculate_regularizer")==0)
		if_calculate_regularizer=1;

	if(strcmp(way_of_processing, "streaming_by_pixels")!=0)
	{
		printf("Training online is only expected to be with streaming by pixels\n"); exit(1);
	}
	else
	{
		strcpy(param_dir, "./data/params/1x128x10/");
		strcat(param_dir, dir_load_param);
		strcat(param_dir, "/");
	}

	load_all_param_and_rm(param_dir);
	// initialize_all_param_and_rm();
	load_input_samples("./data/input/samples.txt");
	print_params_partly();

	// online formulation
	// the original sequence is seen as
	// K subsequences of TS (length)
	// T = TS * K

	clock_t begin = clock();
	for(int i=0; i<K; i++)
	{
		load_sub_seq_to_xc(i);
		relay_network_states();
		forward(TS);

		backward(TS, TS, 2);
		optimizer_and_zero_grad(if_calculate_regularizer); 
	}
	clock_t end = clock();
	double time_spent = (double)(end - begin)/ CLOCKS_PER_SEC ;

	print_network_out(TS);
	printf("\nprint updated parameters\n");
	print_params_partly();

	printf("Latency: %.2f\n", time_spent);
}



/* ---------------------------------
	history solutions below 
	(not used, just for debugging reference)
-----------------------------------*/
void case1()
{
	/*
		MNIST processing: streaming by rows, BPTT
	*/

	// initialize_all_param_and_rm();
	load_all_param_and_rm("./data/params/28x128x10/initial/");
	load_input_samples_to_xc("./data/input/samples.txt");

	forward(28);
	print_network_out(28);

	backward(28, 28, 3);
	optimizer_and_zero_grad(1); 
	
	printf("\nprint updated parameters\n");
	print_params_partly();
}

void case2()
{
	/*
		MNIST processing: streaming by pixels, BPTT
	*/

	// load_all_param_and_rm("./data/params/1x128x10/trained/");
	load_all_param_and_rm("./data/params/1x128x10/initial/");
	load_input_samples_to_xc("./data/input/samples.txt");
	print_params_partly();

	forward(784);
	print_network_out(784);

	backward(784, 784, 3);
	optimizer_and_zero_grad(1); 

	printf("\nprint updated parameters\n");
	print_params_partly();
}