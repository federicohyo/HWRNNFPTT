#include <stdio.h>
#include <stdlib.h>

void print_matrix(int** a, int row, int col)
{
    for(int i=0; i< row; i++)
    {
        for(int j=0; j<col; j++)
            printf("a[%d][%d]=%d  ", i,j, *((int*)a + i*col +j ));

        printf("\n");
    }
}

int TwoDimArraySum4(int row, int col, int twoDimAr[row][col])
{
   int result = 0;

   for (int i = 0; i < row; i++)
   {
       for (int j = 0; j < col; j++)
       {
           //下面两种寻址方式都行
           result += twoDimAr[i][j];
           // result += *(*(twoDimAr+ i) + j);
       }
   }
   return result;
}


void printf4(int *p, int n, int col)
{
 
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < col; j++)
        {
            // printf("第%d行%d列:%d\n", i, j, *(p+i*col+j));//3为每行的元素个数
            printf("第%d行%d列:%d\n", i, j, p[i*col+j] );//3为每行的元素个数
            // printf("第%d行%d列:%d\n", i, j, ((int**)p)[i][j]);//3为每行的元素个数
        }
    }
}

void mat_mul(int * dst, int* src_a, int* src_b, int a_row, int a_col, int b_row, int b_col)
{
    if (a_col != b_row)
    {
        printf("[mat_mul]: Size not matched!\n"); exit(1);
    }

    for(int i=0; i<a_row; i++)
        for(int j=0; j<b_col; j++)
            for(int k=0; k<a_col; k++)
                dst[i*b_col + j] += (src_a[i*a_col + k] * src_b[j + k*b_col]);
}

void mat_mul_b_T_add_bias(int* dst, int* src_a, int* src_b, int a_row, int a_col, int b_row, int b_col, int* bias)
{
    if (a_col != b_col)
    {
        printf("[mat_mul_b_T]: Size not matched!\n"); exit(1);
    }

    for(int i=0; i<a_row; i++)
        for(int j=0; j<b_row; j++)
            for(int k=0; k<a_col; k++)
                dst[i*b_row + j] += src_a[i*a_col + k] * src_b[j*a_col + k];

    for(int i=0; i<a_row; i++)
        for(int j=0; j<b_row; j++)
            dst[i*b_row + j] += bias[j];
}

int main()
{
    int g[2][2][3] = {{{1,2,3},{4,5,6}}, {{7,8,9}, {10,11,12}}};
    int m[2][3]={{1,2,3}, {4,5,6}};
    int n[4] = {7,8,9,10};
    int result = 0;

    int dst[2][4] = {0};
    int src_a[2][3] = {{1,2,3}, {4,5,6}};
    int src_b[3][4] = {{1,2,3,4},{5,6,7,8}, {9,10,11,12}};
    int src_b_T[4][3] = {{1,5,9}, {2,6,10}, {3,7,11}, {4,8,12}};
    int bias[4] = {1,1,1,1};

    print_matrix((int**)dst, 2, 4);
    mat_mul_b_T_add_bias(&dst[0][0], &src_a[0][0], &src_b_T[0][0], 2,3,4,3, bias);

    print_matrix((int**)dst, 2, 4);
    // printf4(&m[0][0], 2, 3);
    // printf4(&n[0], 1, 4);
    // printf4(&g[0][0][0], 2, 3);
    // printf("\n\n\n");
    // printf4(&g[1][0][0], 2, 3);
    // print_matrix((int**)m, 2, 3);
    
    // result = TwoDimArraySum4(2, 3, m);
    // printf("Sum4函数结果：%d\n", result);
    return 0;
}