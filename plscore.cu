#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "plchain.h"
#include "debug.h"

/* 

Parallel chaining helper functions with CUDA

*/

__constant__ Misc misc;

/* arithmetic functions begin */

__device__ static inline float cuda_mg_log2(float x) // NB: this doesn't work when x<2
{
	union { float f; uint32_t i; } z = { x };
	float log_2 = ((z.i >> 23) & 255) - 128;
	z.i &= ~(255 << 23);
	z.i += 127 << 23;
	log_2 += (-0.34484843f * z.f + 2.02466578f) * z.f - 0.67487759f;
	return log_2;
}

__device__ static inline int32_t comput_sc(const int64_t ai_x, const int64_t ai_y, const int64_t aj_x, const int64_t aj_y,
                                int32_t max_dist_x, int32_t max_dist_y,
                                int32_t bw, float chn_pen_gap,
                                float chn_pen_skip, int is_cdna, int n_seg) {
    int32_t dq = (int32_t)ai_y - (int32_t)aj_y, dr, dd, dg, q_span, sc;
    int32_t sidi = (ai_y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;
    int32_t sidj = (aj_y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;
    if (dq <= 0 || dq > max_dist_x) return INT32_MIN;
    dr = (int32_t)(ai_x - aj_x);
    if (sidi == sidj && (dr == 0 || dq > max_dist_y)) return INT32_MIN;
    dd = dr > dq ? dr - dq : dq - dr;
    if (sidi == sidj && dd > bw) return INT32_MIN;
    if (n_seg > 1 && !is_cdna && sidi == sidj && dr > max_dist_y)
        return INT32_MIN;  // nseg = 1 by default
    dg = dr < dq ? dr : dq;
    q_span = aj_y >> 32 & 0xff;
    sc = q_span < dg ? q_span : dg;
    if (dd || dg > q_span) {
        float lin_pen, log_pen;
        lin_pen = chn_pen_gap * (float)dd + chn_pen_skip * (float)dg;
        log_pen =
            dd >= 1 ? cuda_mg_log2(dd + 1) : 0.0f;  // mg_log2() only works for dd>=2
        if (is_cdna || sidi != sidj) {
            if (sidi != sidj && dr == 0)
                ++sc;  // possibly due to overlapping paired ends; give a minor
                       // bonus
            else if (dr > dq || sidi != sidj)
                sc -=
                    (int)(lin_pen < log_pen ? lin_pen
                                            : log_pen);  // deletion or jump
                                                         // between paired ends
            else
                sc -= (int)(lin_pen + .5f * log_pen);
        } else
            sc -= (int)(lin_pen + .5f * log_pen);
    }
    return sc;
}

/* arithmetic functions end */

/* kernels begin */

__global__ void score_generation_naive(const int64_t* anchors_x, const int64_t* anchors_y, int32_t *range,
                        size_t *seg_start_arr, 
                        int32_t* f, uint16_t* p, size_t total_n, size_t seg_count) {

    // NOTE: each block deal with one batch 
    // the number of threads in a block is fixed, so we need to calculate iter
    // n = end_idx_arr - start_idx_arr
    // iter = (range[i] - 1) / num_threads + 1

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    /* calculate the segement for current block */
    size_t start_idx = SIZE_MAX;
    int segid = bid * NUM_SEG_PER_BLOCK;
    while (start_idx == SIZE_MAX){
        if (segid >= seg_count) return;
        start_idx = seg_start_arr[segid];
        segid++;
    }
    segid = (bid + 1) *NUM_SEG_PER_BLOCK;
    size_t end_idx = SIZE_MAX;
    while (end_idx == SIZE_MAX) {
        if (segid >= seg_count) {
            end_idx = total_n;
            break;
        }
        end_idx = seg_start_arr[segid];
        segid++;
    }
    // if (tid == 0) {
    //     printf("bid=%d %d - %d\n", bid, start_idx, end_idx);
    // }

    Misc blk_misc = misc;

    // init f and p
    for (size_t i=start_idx+tid; i < end_idx; i += NUM_THREADS_RANGE) {
        f[i] = anchors_y[i] >> 32 & 0xff;
        p[i] = 0;
    }
    
    for (size_t i=start_idx; i < end_idx; ++i) {
        int32_t range_i = range[i];
        for (int32_t j=tid; j < range_i; j+=NUM_THREADS_SCORE) {
            int32_t sc = comput_sc(anchors_x[i+j+1], anchors_y[i+j+1], anchors_x[i], anchors_y[i],
                                blk_misc.max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap, 
                                blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg);
            if (sc == INT32_MIN) continue;
            sc += f[i];
            if (sc >= f[i+j+1] && sc != (anchors_y[i+j+1]>>32 & 0xff)) {
                f[i+j+1] = sc;
                p[i+j+1] = j+1;
            }
        }
        __syncthreads();
    }
}

/* kernels end */

/* host functions begin */

void upload_misc(input_iter input_arr[INPUT_BATCH_SIZE]) {
    // FIXME: remove this after input data processing is updated
    Misc temp_misc[INPUT_BATCH_SIZE];
    for (int i = 0; i < INPUT_BATCH_SIZE; ++i) {
        memcpy(temp_misc+i, &(input_arr[i].misc), sizeof(Misc));
    }
#ifdef USEHIP
    hipMemcpyToSymbol(HIP_SYMBOL(misc), &input_arr[0].misc, sizeof(Misc));
#else
    cudaMemcpyToSymbol(misc, temp_misc, sizeof(Misc));
#endif
    cudaCheck();
}

void p_rel2idx(const uint16_t* rel, int64_t* p, size_t n) {
    for (int i = 0; i < n; ++i){
        if (rel[i] == 0)
            p[i] = -1;
        else
            p[i] = i - rel[i];
    }
} 

void stream_score_generation(input_iter input_arr[INPUT_BATCH_SIZE], 
                        hostMemPtr* host_mem_ptr, deviceMemPtr* device_mem_ptr, int size, void* stream_, void* event_) {

    cudaStream_t* stream = (cudaStream_t*) stream_; 
    cudaEvent_t* event = (cudaEvent_t*) event_;

    // upload_misc(input_arr); // FIXME: Put this at the beginning when input read is fixed
    #ifdef USEHIP
    hipMemcpyToSymbolAsync(HIP_SYMBOL(misc), &input_arr[0].misc, sizeof(Misc), 0, cudaMemcpyHostToDevice, *stream);
    #else
    cudaMemcpyToSymbolAsync(misc, &input_arr[0].misc, sizeof(Misc), 0, cudaMemcpyHostToDevice, *stream);
    #endif
    
    cudaCheck();
    dim3 DimBlock(NUM_THREADS_SCORE, 1, 1);

    /* reorganize anchor data */
    size_t total_n = device_mem_ptr->total_n;
    size_t cut_num = device_mem_ptr->num_cut;
    
    size_t griddim = (cut_num-1)/NUM_SEG_PER_BLOCK + 1;

    // FIXME: temperally disable this check
    // // sanity check
    // if (griddim > MAX_GRID_SCORE) {
    //     fprintf(stderr, "exceed max grid %zu\n", griddim);
    //     exit(0);
    // }

    dim3 DimGrid(griddim, 1, 1);

    score_generation_naive<<<DimGrid, DimBlock, 0, *stream>>>(device_mem_ptr->d_ax, device_mem_ptr->d_ay, device_mem_ptr->d_range, 
                                            device_mem_ptr->d_cut, device_mem_ptr->d_f, device_mem_ptr->d_p, total_n, cut_num);
    cudaCheck();
#ifdef DEBUG_VERBOSE
    fprintf(stderr, "[M::%s] score generation success\n", __func__);
#endif

#ifdef DEBUG_CHECK
    cudaMemcpyAsync(host_mem_ptr->f, device_mem_ptr->d_f, sizeof(int32_t)*total_n, cudaMemcpyDeviceToHost, *stream);
    cudaMemcpyAsync(host_mem_ptr->p, device_mem_ptr->d_p, sizeof(uint16_t)*total_n, cudaMemcpyDeviceToHost, *stream);
#endif

    cudaCheck();

    cudaEventRecord(*event, *stream);
    cudaCheck();

}

void forward_dp(input_iter input_arr[INPUT_BATCH_SIZE], int size, a_pass pass_ptr) {
    // fprintf(stderr, "[M::%s] testing\n", __func__);

    upload_misc(input_arr); // FIXME: Put this at the beginning when input read is fixed
    dim3 DimBlock(NUM_THREADS_SCORE, 1, 1);

    /* reorganize anchor data */
    size_t total_n = pass_ptr.total_n;
    size_t cut_num = pass_ptr.num_cut;

    size_t griddim = (cut_num-1)/NUM_SEG_PER_BLOCK + 1;
#ifdef DEBUG_VERBOSE
    printf("Score Launch grid %d\n", griddim);
#endif // DEBUG_VERBOSE
    // sanity check
    if (griddim > MAX_GRID_SCORE) {
        fprintf(stderr, "exceed max grid\n");
        exit(0);
    }

    dim3 DimGrid(griddim, 1, 1);

    // copy data to gpu
    int32_t *d_f; // score
    uint16_t *d_p; // predecessor

    cudaMalloc(&d_f, sizeof(int32_t)*total_n);
    cudaMalloc(&d_p, sizeof(uint16_t)*total_n);
    cudaCheck();


    // Run kernel
    // printf("Grid Dim, %d\n", DimGrid.x);
    score_generation_naive<<<DimGrid, DimBlock>>>(pass_ptr.x, pass_ptr.y, pass_ptr.range, 
                                            pass_ptr.cut,
                                            d_f, d_p, total_n, cut_num);
    cudaCheck();
    cudaDeviceSynchronize();
    cudaCheck();
#ifdef DEBUG_VERBOSE
    fprintf(stderr, "[M::%s] score generation success\n", __func__);
#endif
    
    int32_t* f = (int32_t*)malloc(sizeof(int32_t) * total_n);
    uint16_t* p_rel = (uint16_t*)malloc(sizeof(uint16_t) * total_n);
    cudaMemcpy(f, d_f, sizeof(int32_t)*total_n, cudaMemcpyDeviceToHost);
    cudaMemcpy(p_rel, d_p, sizeof(uint16_t)*total_n, cudaMemcpyDeviceToHost);
    cudaCheck();

#ifdef DEBUG_CHECK
    //check score
    int64_t read_start = 0;
    for (int i = 0; i < size; i++){
        int64_t* p = (int64_t*)malloc(sizeof(int64_t)*input_arr[i].n);
        p_rel2idx(p_rel + read_start, p, input_arr[i].n);
#ifdef DEBUG_VERBOSE
        // fprintf(stderr, "Read#%d\n", i);
        // debug_print_score(f+read_start, input_arr[i].n);
        // fprintf(stderr, "Reference score: ");
        // debug_print_score(input_arr[i].f, input_arr[i].n);
#endif
        debug_check_score(p, f+read_start, input_arr[i].p, input_arr[i].f, input_arr[i].n);
        read_start += input_arr[i].n;
    }
#endif
    cudaFree(pass_ptr.x);
    cudaFree(pass_ptr.y);
    cudaFree(pass_ptr.range);

    cudaFree(pass_ptr.cut);
    cudaFree(d_f);
    cudaFree(d_p);

}

/* host functions end */


