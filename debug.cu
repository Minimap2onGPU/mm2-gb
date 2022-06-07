#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

__global__ void temp_func(uint *input, unsigned int opt) {
    __shared__ uint s_data[12];

    int tid = threadIdx.x;
    int t;

    for (int i=0; i < 12; ++i) {
        t = tid%12;
    }

    __syncthreads();

}



