#include "headers.h"
#include "caselist.h"


void __main(void)
{

#ifdef RVMULTICORE
 	size_t mhartid = read_csr(mhartid);
 	if (mhartid >= NCORES) while (1);

 	// for(int i=0; i<NCORES; i++)
 	// {
	// 	barrier(NCORES);
 	// 	if(mhartid==i)
	// 		printf("Core %ld launched \n", mhartid);
 	// }
	// barrier(NCORES);
#endif

  
#ifdef PRINT_CONFIG
	#ifndef RVMULTICORE 
	  	print_basic_config();
	  	print_static_memory_usage();
	#else
	if(mhartid==0) 
	{
	  	print_basic_config();
	  	print_static_memory_usage();
	}
	barrier(NCORES);
	#endif
#endif
  
// #ifdef MNIST_OFFLINE_BY_ROWS
//   	learning_mnist_offline("streaming_by_rows", "randomized");
// #endif
  
// #ifdef MNIST_OFFLINE_BY_PIXELS
//   	learning_mnist_offline("streaming_by_pixels", "pre-trained");
//   	// learning_mnist_offline("streaming_by_pixels", "randomized", REGULARIZATION);
// #endif
  
#ifdef MNIST_ONLINE_BY_PIXELS /* specify K in Makefile */
	#ifndef RVMULTICORE
   		learning_mnist_online("pre-trained", REGULARIZATION);
	#else
   		learning_mnist_online(mhartid, "pre-trained", REGULARIZATION);
	#endif
#endif
 
#ifdef PRINT_PERF
	#ifdef RVMULTICORE 
		if(mhartid==0) 
	#endif
	print_function_acc_time();
#endif
#ifdef PRINT_COUNTER
	#ifdef RVMULTICORE 
		if(mhartid==0) 
	#endif
 	print_operation_count();
#endif

#ifdef RVMULTICORE 
	if(mhartid==0) 
		printf("Finished\n");
	if (mhartid > 0) while (1);
#else
	printf("Finished\n");
#endif
}

int main(void)
{
	__main();
	return 0;
}