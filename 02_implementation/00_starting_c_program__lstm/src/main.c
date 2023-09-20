#include "lstm.h"



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

}



int main()
{
	print_static_memory_usage();
	case1();
	// for(int i=0; i<10; i++)
		// printf("%.8f ", l1_wo[0][i]);
	
	// for(int j=0; j<BS; j++)
	// {
	// 	for(int i=0; i<28; i++)
	// 		printf("%.8f ", xc[2][j][i]);

	// 	printf("\n\n\n");
	// }
	printf("Finished\n");
	return 0;
}