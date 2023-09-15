#include "lstm.h"



void test1()
{
	/*
		MNIST processing: streaming by rows
	*/

	// initialize_all_param_and_rm();
	load_all_param_and_rm("./data/params/initial/");

	for(int i=0; i<10; i++)
		printf("%f\n", l2_b[i]);

	load_input_samples_to_xc("./data/input/samples.txt");
}



int main()
{
	test1();

	printf("Finished\n");
	return 0;
}