#ifndef __PLKERNEL_H__
#define __PLKERNEL_H__

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "plchain.h"

// __constant__ Misc misc[NUM_STREAMS];
__constant__ Misc misc[NUM_STREAMS];

/* Range selection kernels begin */

inline __device__ int64_t range_binary_search(const int64_t* ax, int64_t i, int64_t st_end){
    int64_t st_high = st_end, st_low=i;
    while (st_high != st_low) {
        int64_t mid = (st_high + st_low -1) / 2+1;
        if (ax[i] >> 32 != ax[mid] >> 32 || ax[mid] > ax[i] + MAX_DIST_X) {
            st_high = mid -1;
        } else {
            st_low = mid;
        }
    }
    return st_high;
}

__global__ void range_selection_kernel_naive(const int64_t* ax, size_t *start_idx_arr, size_t *read_end_idx_arr, int32_t *range, size_t* cut, size_t* cut_start_idx){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    size_t start_idx = start_idx_arr[bid];
    size_t read_end_idx = read_end_idx_arr[bid];
    size_t end_idx = start_idx + MAX_ANCHOR_PER_BLOCK;
    end_idx = end_idx > read_end_idx ? read_end_idx : end_idx;
    size_t cut_idx = cut_start_idx[bid];
    if(tid == 0 && (bid == 0 || read_end_idx_arr[bid-1] != read_end_idx)){
        cut[cut_idx] = start_idx;
    }
    cut_idx++;
    const int range_op[7] = {16, 512, 1024, 2048, 3072, 4096, MAX_ITER};  // Range Options
    for (size_t i = start_idx + tid; i < end_idx; i += NUM_THREADS_RANGE){
        size_t st_max = i + MAX_ITER;
        st_max = st_max < read_end_idx ? st_max : read_end_idx -1;
        size_t st;
        for (int j=0; j<7; ++j){
            st = i + range_op[j];
            st = st <= st_max ? st : st_max;
            if (st > i && (ax[i] >> 32 != ax[st] >> 32 || ax[st] > ax[i] + MAX_DIST_X)){
                break;
            }
        }
        st = range_binary_search(ax, i, st);
        range[i] = st - i;

        if (tid >= NUM_THREADS_RANGE - CUT_CHECK_ANCHORS && NUM_THREADS_RANGE - tid + i <= end_idx){
            if (st == i) cut[cut_idx] = i+1;
        }
        cut_idx++;
    }
}

__global__ void range_selection_kernel(const int64_t* ax, size_t *start_idx_arr, size_t *read_end_idx_arr, int32_t *range){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    size_t start_idx = start_idx_arr[bid];
    size_t read_end_idx = read_end_idx_arr[bid];
    size_t end_idx = start_idx + MAX_ANCHOR_PER_BLOCK;
    end_idx = end_idx > read_end_idx ? read_end_idx : end_idx;

    size_t load_anchor_idx = 100;
    size_t load_smem_idx;
    size_t cal_idx = start_idx + threadIdx.x;
    int32_t cal_smem = tid;
    __shared__ int64_t smem[NUM_ANCHOR_IN_SMEM];

    /* prefetch anchors */
    load_smem_idx = tid;
    load_anchor_idx = start_idx + tid;
    // if (tid == 20) printf("load_smem_idx %d, load_anchor_idx %lu\n", load_smem_idx, load_anchor_idx);
    for (int i = 0; i < PREFETCH_ANCHORS_RANGE/NUM_THREADS_RANGE && load_anchor_idx < read_end_idx; ++i){
        // if (tid == 20) printf("load_smem_idx %d, load_anchor_idx %lu\n", load_smem_idx, load_anchor_idx);
        smem[load_smem_idx] = ax[load_anchor_idx];
        load_smem_idx += NUM_THREADS_RANGE;
        load_anchor_idx += NUM_THREADS_RANGE;
    }

    int iter = (NUM_ANCHOR_IN_SMEM - PREFETCH_ANCHORS_RANGE)/NUM_THREADS_RANGE; // iterations before another load is needed
    while (cal_idx < end_idx) { // tail threads may skip this loop
        /* load anchors */
        load_smem_idx = load_smem_idx >= NUM_ANCHOR_IN_SMEM ? load_smem_idx - NUM_ANCHOR_IN_SMEM : load_smem_idx;
        for (int i = 0; i < iter && load_anchor_idx < end_idx + PREFETCH_ANCHORS_RANGE; ++i){
            // if (tid == 20) printf("load it load_smem_idx %d, load_anchor_idx %lu\n", load_smem_idx, load_anchor_idx);
            smem[load_smem_idx] = ax[load_anchor_idx];
            load_smem_idx += NUM_THREADS_RANGE;
            load_anchor_idx += NUM_THREADS_RANGE;
            load_smem_idx = load_smem_idx >= NUM_ANCHOR_IN_SMEM ? load_smem_idx - NUM_ANCHOR_IN_SMEM : load_smem_idx;
        }

        __syncthreads();
        
        /* calculate sucessor range */
        for (int i = 0; i < iter && cal_idx < end_idx; ++i){
            int64_t anchor = smem[cal_smem];

            size_t st = cal_idx + PREFETCH_ANCHORS_RANGE < read_end_idx ? cal_idx + PREFETCH_ANCHORS_RANGE : read_end_idx-1;
            int32_t st_smem = cal_smem + st - cal_idx;
            st_smem = st_smem >= NUM_ANCHOR_IN_SMEM ? st_smem - NUM_ANCHOR_IN_SMEM : st_smem;
            // if (tid == 20) printf("cal idx %lu, cal_mem %d, st %lu, st_smem %d\n", cal_idx, cal_smem, st,st_smem);

            // if (tid == 20) printf("anchor.x %d, smem[st_smem] %d, anchor.x+MAX_DIST_X%d\n", anchor, smem[st_smem], anchor+MAX_DIST_X);

            while (st > cal_idx && 
                        (anchor>> 32 != smem[st_smem] >> 32 ||
                            smem[st_smem] > anchor + MAX_DIST_X
                        )
                    ){
                // if (bid == 25)
                // printf("while 0 bid %d tid %d cal_idx %d\n", bid, tid, cal_idx);
                --st;
                if (st_smem == 0) st_smem = NUM_ANCHOR_IN_SMEM-1;
                else --st_smem;
            }
            
            /* NOTE: fallback: succussor is not prefetched */
            if (st >= PREFETCH_ANCHORS_RANGE + cal_idx){
                st = cal_idx + MAX_ITER < read_end_idx ? i + MAX_ITER : read_end_idx-1;
                while(
                    anchor >> 32 != ax[st] >> 32 || 
                    ax[st] > anchor + MAX_DIST_X // check from global memory
                ){
                    --st;
                    // if (bid == 25)
                    // printf("while 1 bid %d tid %d\n", bid, tid);
                }

            }
            range[cal_idx] = st - cal_idx;
            cal_smem += NUM_THREADS_RANGE;
            cal_smem = cal_smem >= NUM_ANCHOR_IN_SMEM ? cal_smem - NUM_ANCHOR_IN_SMEM : cal_smem;
            cal_idx += NUM_THREADS_RANGE;
            // if (bid == 25)
            // printf("for loop i %d bid %d tid %d\n", i, bid, tid);
        }
        // if (bid == 25)
        // printf("outer while bid %d tid %d\n", bid, tid);
        __syncthreads();

    }
    
}

/* Range selection kernels end */

/* Score generation kernels begin */

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

    Misc blk_misc = misc[0];

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

/* Score generation kernels end */

/* Host functions begin */


void upload_misc_stream(int stream_idx, Misc *input_misc, cudaStream_t *stream) {
    #ifdef USEHIP
    hipMemcpyToSymbolAsync(HIP_SYMBOL(misc[stream_idx]), &input_misc, sizeof(Misc), 0, cudaMemcpyHostToDevice, *stream);
    #else
    cudaMemcpyToSymbolAsync(misc[stream_idx], &input_misc, sizeof(Misc), 0, cudaMemcpyHostToDevice, *stream);
    #endif
    
    cudaCheck();
}

void upload_misc(Misc *misc_info) {
    #ifdef USEHIP
    hipMemcpyToSymbol(HIP_SYMBOL(misc[0]), misc_info, sizeof(Misc), 0, cudaMemcpyHostToDevice);
    #else
    cudaMemcpyToSymbol(misc[0], misc_info, sizeof(Misc), 0, cudaMemcpyHostToDevice);
    #endif
    
    cudaCheck();
}


// void p_rel2idx(const uint16_t* rel, int64_t* p, size_t n) {
//     for (int i = 0; i < n; ++i){
//         if (rel[i] == 0)
//             p[i] = -1;
//         else
//             p[i] = i - rel[i];
//     }
// } 


/* Host functions end */

#endif // !__PLKERNEL_H__