#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "mmpriv.h" // declare functions in this header
#include "kalloc.h"
#include "krmq.h"

/* 

Parallel chaining helper functions with CUDA

*/

// template kernel code

#define NUM_THREADS	32

__global__ void temp_func(unsigned int opt) {
    __shared__ uint s_data[NUM_THREADS];

    int tid = threadIdx.x;

    // for (int i=0; i < 12; ++i) {
    //     t = tid%12 + opt;
    //     s_data[tid]
    // }
    s_data[tid] = opt;

    __syncthreads();

    s_data[(tid+2)%12] += opt;

}

void range_selection() {
    // fprintf(stderr, "[M::%s] testing\n", __func__);
    // dim3 DimBlock(NUM_THREADS,1,1);
    // dim3 DimGrid(1, 1, 1);

    // // Run kernel
    // temp_func<<<DimGrid, DimBlock>>>(1);
    // cudaDeviceSynchronize();
    // fprintf(stderr, "[M::%s] success\n", __func__);
}

void forward_dp() {
    // fprintf(stderr, "[M::%s] testing\n", __func__);
}


