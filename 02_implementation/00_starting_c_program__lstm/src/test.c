#include "headers.h"
#include "caselist.h"


int main()
{
#ifdef PRINT_CONFIG
  	print_basic_config();
  	print_static_memory_usage();
#endif
  
#ifdef MNIST_OFFLINE_BY_ROWS
  	learning_mnist_offline("streaming_by_rows", "randomized");
#endif
  
#ifdef MNIST_OFFLINE_BY_PIXELS
  	learning_mnist_offline("streaming_by_pixels", "pre-trained");
  	// learning_mnist_offline("streaming_by_pixels", "randomized", REGULARIZATION);
#endif
  
#ifdef MNIST_ONLINE_BY_PIXELS /* specify K in Makefile */
    learning_mnist_online("streaming_by_pixels", "pre-trained", REGULARIZATION);
#endif
  
#ifdef PRINT_PERF
    print_function_acc_time();
#endif
#ifdef PRINT_COUNTER
  	print_operation_count();
#endif

	printf("Finished\n");
	return 0;
}