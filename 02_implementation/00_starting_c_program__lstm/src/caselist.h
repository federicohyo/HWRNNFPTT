#ifndef _CASELIST_H_
#define _CASELIST_H_

#include "lstm.h"


/*
    offline means network parameters won't be updated until 
    the entire input sequence has been streamed into the network
    i.e., forward TS steps, then do backward
*/
void learning_mnist_offline(char way_of_processing[], char dir_load_param[], char reg_option[]);


/*
    online training is the opposite of the offline training. 
    assuming that the input sequence arrives in a continuous way.
    The network can do backward path after a subsequence is completed with the forward path.
    In other words, mulitple times of parameter update and interleaved forward/backward path.
*/
void learning_mnist_online(char way_of_processing[], char dir_load_param[], char reg_option[]);

#endif//_CASELIST_H_