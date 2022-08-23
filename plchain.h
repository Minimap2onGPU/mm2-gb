#ifndef _PLCHAIN_H_
#define _PLCHAIN_H_

#include <assert.h>
#include "minimap.h"
#include "bseq.h"
#include "kseq.h"
#include "hipify.h"
#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/* kernel parameters */

#define INPUT_BATCH_SIZE 32

#define MAX_NUM_ANCHORS        33554432 // 2^25
// #define MAX_NUM_ANCHORS        3554432 // 2^25
#define MAX_NUM_CUT            65536 // 2^16
#define MAX_GRID               10000
#define MAX_GRID_SCORE         655360
#define ANCHORS_PER_BLOCK (256) // max_grid = anchors / anchors_per_block
#define ANCHORS_PER_CUT (256) // max_cut = anchors / anchors_per_cut

#ifndef MAX_IT_BLOCK_RANGE
#define MAX_IT_BLOCK_RANGE	       64 // target number of iteration per block for range selection kernel
#endif

#ifndef NUM_THREADS_RANGE
#define NUM_THREADS_RANGE      512 // number of thread per block for range selection kernel
#endif

#ifndef PREFETCH_ANCHORS_RANGE
#define PREFETCH_ANCHORS_RANGE 512 // number of anchors prefetched for range selection kernel
#endif

#ifndef CUT_CHECK_ANCHORS
#define CUT_CHECK_ANCHORS  10
#endif

#define NUM_THREADS_SCORE      512 // number of thread per block for score generation kernel
#define NUM_SEG_PER_BLOCK      3

#define MAX_ANCHOR_PER_BLOCK (MAX_IT_BLOCK_RANGE* NUM_THREADS_RANGE)

#define MAX_ITER 5000
#define MAX_DIST_X 5000
#define NUM_ANCHOR_IN_SMEM 3072
#define NUM_STREAMS 4

typedef __int32_t int32_t;

/* functions declaration */
double dynamic_stream_chain_loop(input_iter* input_arr, int total_reads);

void stream_chain_loop(input_iter* input_arr, int total_iter);

void stream_range_selection(input_iter input_arr[INPUT_BATCH_SIZE], 
                        hostMemPtr* host_mem_ptr, deviceMemPtr* device_mem_ptr, int size, void* stream_);

void stream_score_generation(input_iter input_arr[INPUT_BATCH_SIZE], 
                        hostMemPtr* host_mem_ptr, deviceMemPtr* device_mem_ptr, int size, void* stream_, void* event_);

void p_rel2idx(const uint16_t* rel, int64_t* p, size_t n);
a_pass range_selection(input_iter input_arr[INPUT_BATCH_SIZE], int size);
void forward_dp(input_iter input_arr[INPUT_BATCH_SIZE], int size, a_pass pass_ptr);

// CPU functions
void forward_range_selection_cpu(mm128_t* a, int64_t n, int max_dist_x,
                                 int max_iter,  // input  max_detection_range
                                 int32_t* range);
void forward_chaining_cpu(input_iter*);

#ifdef __cplusplus
}
#endif

#endif  // _PLCHAIN_H_


