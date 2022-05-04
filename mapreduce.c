#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#define DTYPE int

typedef DTYPE (*map_func_t)(DTYPE);
typedef DTYPE (*reduce_func_t)(DTYPE, DTYPE);


DTYPE *init(DTYPE *in, const int len, const DTYPE val)
{
    assert(in && len>0);
    for(int idx=0; idx<len; ++idx)
    {
        in[idx] = val;
    }
    return in;
}

DTYPE *map(register const DTYPE *in, register DTYPE *out, register const int len)
{
    assert(in && out && len>0);	//depurar errores
    for(register int idx=0; idx<len; ++idx)
    {
        out[idx] = in[idx] * in[idx];
    }
    return out;
}

DTYPE reduce(register const DTYPE *in, register const int len)
{
    assert(in && len>0);	//depurar errores
    DTYPE res = 0;
    for(register int idx=0; idx<len; ++idx)
    {
        DTYPE e = in[idx];
        res = res + e;
    }
    return res;
}

int main(int argc, char *argv[])
{
    int use_thread_num = omp_get_num_threads();
    int current_thread_idx;
    double start_time = 0.0;
    double end_time = 0.0;
    printf("use_thread_num = %d\n", use_thread_num);

    // init
    const int n = 1e6;
    printf("n = %d\n", n);
    DTYPE *a = calloc(n, sizeof(DTYPE));
    DTYPE *b = calloc(n, sizeof(DTYPE));
    DTYPE *c = calloc(n, sizeof(DTYPE));

    printf("start init\n");
    a = init(a, n, 1);
    b = init(b, n, 2);
    c = init(c, n, 3);


	// ++++MAP++++
    printf("start map\n");
    start_time = omp_get_wtime(); 
    DTYPE *mapped_a = map(a, a, n);
    end_time = omp_get_wtime(); 
    printf("mapped_a:%.4f\n", end_time-start_time);

    start_time = omp_get_wtime(); 
    DTYPE *mapped_b = map(b, b, n);
    end_time = omp_get_wtime(); 
    printf("mapped_b:%.4f\n", end_time-start_time);

    start_time = omp_get_wtime(); 
    DTYPE *mapped_c = map(c, c, n);
    end_time = omp_get_wtime(); 
    printf("mapped_c:%.4f\n", end_time-start_time);


	// ++++REDUCE++++
    printf("start reduce\n");
    start_time = omp_get_wtime(); 
    DTYPE reduced_a = reduce(a, n);
    end_time = omp_get_wtime();
    
    printf("finish reduced_a:%.4f\n", end_time-start_time);

    start_time = omp_get_wtime(); 
    DTYPE reduced_b = reduce(b, n);
    end_time = omp_get_wtime();
    printf("finish reduced_b:%.4f\n", end_time-start_time);

    start_time = omp_get_wtime(); 
    DTYPE reduced_c = reduce(c, n);
    end_time = omp_get_wtime();
    printf("finish reduced_b:%.4f\n", end_time-start_time);

    printf("reduced_a:%.2f\n", (float)reduced_a);
    printf("reduced_b:%.2f\n", (float)reduced_b);
    printf("reduced_c:%.2f\n", (float)reduced_c);

    // free
    if(a) free(a); a = NULL;
    if(b) free(b); b = NULL;
    if(c) free(c); c = NULL;

    return 0;
}
