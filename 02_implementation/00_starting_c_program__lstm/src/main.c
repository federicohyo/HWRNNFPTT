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

	backward(28, 28, 3);

	optimizer_and_zero_grad(1); // FPTT as an option in the optimizer, 1 is on

	printf("\nprint updated parameters\n");
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
	printf("\n");


}



int main()
{
	print_static_memory_usage();
	case1();
	
	printf("Finished\n");
	return 0;
}