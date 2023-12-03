#include "caselist.h"


int main()
{
	print_basic_config();
	print_static_memory_usage();

	#ifdef MNIST_OFFLINE_BY_ROWS
		learning_mnist_offline("streaming_by_rows", "randomized", "calculate_regularizer");
	#endif

	#ifdef MNIST_OFFLINE_BY_PIXELS
		learning_mnist_offline("streaming_by_pixels", "pre-trained", "calculate_regularizer");
		// learning_mnist_offline("streaming_by_pixels", "randomized", "calculate_regularizer");
	#endif

	#ifdef MNIST_ONLINE_BY_PIXELS // specify K in ./include/headers.h
		learning_mnist_online("streaming_by_pixels", "pre-trained", "calculate_regularizer");
	#endif

	print_operation_count();

	printf("Finished\n");
	return 0;
}