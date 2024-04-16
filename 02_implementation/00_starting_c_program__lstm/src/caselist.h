#ifndef _CASELIST_H_
#define _CASELIST_H_

#include "lstm.h"




// see README.md the definition of offline
#ifndef RVMULTICORE
void learning_mnist_online(char dir_load_param[], int reg_option)
#else
void learning_mnist_online(int cid, char dir_load_param[], int reg_option)
#endif
{

#ifdef RVMULTICORE 
if(cid==0) // if multicore riscv, make sure the function is executed only on core 0
{	
#endif 
	initialization(dir_load_param);
#ifdef RVMULTICORE
}// closing bracket 
// barrier(NCORES);
#endif 

#ifdef PRINT_DEBUG
	#ifndef RVMULTICORE
		printf("\noriginal parameters (partly)\n");
		print_params_partly();
	#else
		if(cid==0)
		{
			printf("\noriginal parameters (partly)\n");
			print_params_partly();
		}
		// barrier(NCORES);
	#endif
#endif

#ifdef PRINT_PERF
	#ifdef PLATFORM_X86 
		clock_t begin = clock();
	#endif
	#ifdef PLATFORM_RV
		#ifndef RVMULTICORE
			size_t sc = rdcycle();
		#else
			size_t sc;
			if(cid==0)
			{
				sc = rdcycle();
			}
			barrier(NCORES);// sync all threads(cores)
		#endif
	#endif
#endif



	// core computation
	// online formulation the original sequence is seen as K subsequences of TS (length)
	// T = TS * K 
	for(int i=0; i<K_VAL; i++)
	{
		// #ifndef BM
			#ifdef RVMULTICORE// if multi-core, only core-0 needs to execute the code below 
			if(cid==1)
			#endif
				printf("K: %d\n", i);
		// #endif
				// printf("K: %d\n", i);

		#ifdef RVMULTICORE// if multi-core, only core-0 needs to execute the code in between 
			if(cid==0) 
			{
		#endif
				load_sub_seq_to_xc(i);
				relay_network_states();
		#ifdef RVMULTICORE
			}
			barrier(NCORES);// sync all threads(cores)
		#endif
		
		#ifndef RVMULTICORE
			forward(TS);
			backward(TS, TS, 2);
			optimizer_and_zero_grad(reg_option); 
			// cross_entropy();
			
			// print_network_out(1);
			// if(i==0){
			// 	print_network_out(1);
			// 	// print_network_out(TS);
			// }
		#else
			forward(cid, TS);
			// barrier(NCORES);// sync all threads(cores)
			backward(cid, TS, TS, 2);
			// barrier(NCORES);// sync all threads(cores)
			optimizer_and_zero_grad(cid, reg_option); 
			// barrier(NCORES);// sync all threads(cores)
		#endif
	}

#ifdef PRINT_PERF
	#ifdef PLATFORM_X86 
		clock_t end = clock();
		double time_spent = ((double)(end - begin))/ 4e7;
		printf("[clock] cycles taken: %ld, Latency: %f\n", (end - begin), time_spent);
	#endif
	#ifdef PLATFORM_RV
		#ifndef RVMULTICORE
			size_t ec = rdcycle();
			size_t cycles = ec-sc;
			printf("[rdcycle]: cycles taken: %ld, latency(in sec): %ld\n", cycles, (int)(((double)cycles)/4e7) );
		#else// if multi-core, only core-0 needs to execute the code in between 
			size_t ec, cycles;
			if(cid==0)
			{
				ec = rdcycle();
				cycles = ec-sc;
				printf("[rdcycle]: cycles taken: %ld, latency(in sec): %ld\n", cycles, (int)(((double)cycles)/4e7) );
			}
			// barrier(NCORES);	// sync all threads(cores)
		#endif
	#endif
#endif


#ifdef PRINT_DEBUG
	#ifdef RVMULTICORE // if multi-core, only core-0 needs to execute the code in between 
	if(cid==0)
	{
	#endif
		cross_entropy();
		printf("\nupdated parameters (partly)\n");
		print_params_partly();
	#ifdef RVMULTICORE// if multi-core, only core-0 needs to execute the code in between 
	}
	#endif
#endif

}



#endif//_CASELIST_H_
