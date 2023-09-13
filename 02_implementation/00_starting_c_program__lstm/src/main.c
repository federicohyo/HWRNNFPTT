#include "lstm.h"

int main()
{
    srand(time(NULL));
	initialize_all_param_and_rm();

	printf("%f, %f, %f\n", max_ini_v, min_ini_v, l2_b[0]);

	for (int i=0; i<10; i++)
		printf("%f\n", l2_b[i]);

	printf("Functional\n");
	return 0;
}