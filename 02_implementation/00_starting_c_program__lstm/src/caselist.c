/*
    In the directory include/caselist.h
    see data/function declaration 
*/

#include "caselist.h"

void case1()
{
	/*
		MNIST processing: streaming by rows
	*/

	// initialize_all_param_and_rm();
	load_all_param_and_rm("./data/params/initial/");
	load_input_samples_to_xc("./data/input/samples.txt");

	forward(28);
	print_network_out(28);
	backward(28, 28, 3);
	optimizer_and_zero_grad(1); // FPTT as an option in the optimizer, 1 is on
	print_updated_params_partly();

}