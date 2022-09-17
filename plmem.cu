#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "plchain.h"
#include "debug.h"
#include <time.h>

hostMemPtr host_mem_ptrs[NUM_STREAMS];
deviceMemPtr device_mem_ptrs[NUM_STREAMS];
// cudaStream_t streams[NUM_STREAMS]; // init streams
// cudaEvent_t events[NUM_STREAMS];

// i = __sync_fetch_and_add(i, delta); // NOTE: atomic add, i starts from thread_idx
int total_frags = 0;
int total_tasks = 0;
int max_awaiting_tasks = 0;
int awaiting_tasks = 0;
pthread_mutex_t pltask_lock; // lock for task append
pthread_cond_t pltask_cv;

size_t total_n = 0; // NOTE: index where the array is stored 
int size = 0;
int grid_dim = 0;
int cut_num = 0;

// TODO: put this into plscore?
__constant__ Misc misc;

int64_t *get_p(int64_t n, int i) {
    uint16_t *rel = host_mem_ptrs[0].p+i;
    int64_t* p = (int64_t*)malloc(sizeof(int64_t)*n);
    for (int i = 0; i < n; ++i) {
        if (rel[i] == 0)
            p[i] = -1;
        else
            p[i] = i - rel[i];
    }
    return p; // TODO: make sure p is freed after using it
}

int32_t *get_f(int64_t n, int i) {
    return host_mem_ptrs[0].f+i;
}

int pltask_init(int num_threads, int num_seqs) { 

    size = num_threads;
    max_awaiting_tasks = num_threads;
    total_frags = num_seqs;
    // NOTE: allocate pin memory for each stream
    size_t avail_mem_stream = MEM_GPU;
    avail_mem_stream = MEM_GPU/NUM_STREAMS * 1e9; // split memory for each stream
    // memory per anchor = ax + ay + range + f + p + (start_idx + read_end_idx + cut_start_idx + cut)
    // size: F1 = ax + ay + range + f + p; F2 = start_idx + read_end_idx + cut_start_idx; F3 = cut
    int64_t F1 = 8+8+4+4+2, F2 = 8+8+8, F3 = 8;
    int64_t P1 = ANCHORS_PER_BLOCK, P2 = ANCHORS_PER_CUT;
    // avail_memory = (F1 + F2/ANCHORS_PER_BLOCK + F3/ANCHORS_PER_CUT) * num_anchors
    int64_t max_anchors_stream = (avail_mem_stream*P1*P2) / (F1*P1*P2 + F2*P2 + F3*P1); // ignore misc as anchors cannot just fit whole memory
    int64_t max_grid = max_anchors_stream / ANCHORS_PER_BLOCK;
    int64_t max_num_cut = max_anchors_stream / ANCHORS_PER_CUT;

    for (int i = 0; i < NUM_STREAMS; ++i) {
        // cudaStreamCreate(&streams[i]);
        // cudaEventCreate(&events[i]);
        cudaCheck();
        // set up host memory pointers
        host_mem_ptrs[i].index = -1; // -1 means unused stream
        cudaMallocHost((void**)&host_mem_ptrs[i].ax, max_anchors_stream * sizeof(int64_t));
        cudaMallocHost((void**)&host_mem_ptrs[i].ay, max_anchors_stream * sizeof(int64_t));
        cudaMallocHost((void**)&host_mem_ptrs[i].f, max_anchors_stream * sizeof(int32_t));
        cudaMallocHost((void**)&host_mem_ptrs[i].p, max_anchors_stream * sizeof(uint16_t));
        cudaMallocHost((void**)&host_mem_ptrs[i].start_idx, max_grid * sizeof(size_t));
        cudaMallocHost((void**)&host_mem_ptrs[i].read_end_idx, max_grid * sizeof(size_t));
        cudaMallocHost((void**)&host_mem_ptrs[i].cut_start_idx, max_grid * sizeof(size_t));
        // set up GPU memory pointers
        cudaMalloc(&device_mem_ptrs[i].d_ax, max_anchors_stream * sizeof(int64_t));
        cudaMalloc(&device_mem_ptrs[i].d_ay, max_anchors_stream * sizeof(int64_t));
        cudaMalloc(&device_mem_ptrs[i].d_range, max_anchors_stream * sizeof(int32_t));
        cudaMalloc(&device_mem_ptrs[i].d_f, max_anchors_stream * sizeof(int32_t));
        cudaMalloc(&device_mem_ptrs[i].d_p, max_anchors_stream * sizeof(uint16_t));    
        cudaMalloc(&device_mem_ptrs[i].d_cut, max_num_cut * sizeof(size_t));
        cudaMalloc(&device_mem_ptrs[i].d_start_idx, sizeof(size_t) * max_grid);
        cudaMalloc(&device_mem_ptrs[i].d_read_end_idx, sizeof(size_t) * max_grid);
        cudaMalloc(&device_mem_ptrs[i].d_cut_start_idx, sizeof(size_t) * max_grid);
        cudaCheck();
    }
    return 0;
}

size_t pltask_append(int64_t n, mm128_t *a, int i) {  
    // NOTE: This function must be called inside a critical section
    // FIXME: so far i is never used 

    size_t idx = total_n;
    
    total_n += n;

    hostMemPtr *host_mem_ptr = host_mem_ptrs + 0;
    deviceMemPtr *device_mem_ptr = device_mem_ptrs + 0;

    int block_num = (n - 1) / MAX_ANCHOR_PER_BLOCK + 1;

    host_mem_ptr->start_idx[grid_dim] = idx;
    size_t end_idx = idx + MAX_ANCHOR_PER_BLOCK;
    host_mem_ptr->read_end_idx[grid_dim] = idx + n;
    host_mem_ptr->cut_start_idx[grid_dim] = cut_num;
    for (int j = 1; j < block_num; j++) {
        cut_num += MAX_IT_BLOCK_RANGE;
        host_mem_ptr->start_idx[grid_dim + j] = end_idx;
        end_idx = host_mem_ptr->start_idx[grid_dim + j] + MAX_ANCHOR_PER_BLOCK;
        host_mem_ptr->read_end_idx[grid_dim + j] = idx + n;
        host_mem_ptr->cut_start_idx[grid_dim + j] = cut_num;
    }

    cut_num += (n - (block_num - 1) * MAX_ANCHOR_PER_BLOCK - 1) / NUM_THREADS_RANGE + 1;

    grid_dim += block_num;

    // copy anchors to pin memory
    for (int j = 0; j < n; j++){
        host_mem_ptr->ax[idx] = a[j].x;
        host_mem_ptr->ay[idx] = a[j].y; 
        ++idx;
    }

    return total_n-n;
}

int pltask_launch(int max_dist_x, int max_dist_y, int bw, int max_skip, int max_iter,
    float chn_pen_gap, float chn_pen_skip, int is_cdna,
    int n_seg) {

    hostMemPtr *host_mem_ptr = host_mem_ptrs;
    deviceMemPtr *device_mem_ptr = device_mem_ptrs;
    device_mem_ptr->num_cut = cut_num;

    dim3 DimBlock0(NUM_THREADS_RANGE, 1, 1);
    dim3 DimGrid0(grid_dim, 1, 1);

    cudaMemcpy(device_mem_ptr->d_ax, host_mem_ptr->ax, sizeof(int64_t)*total_n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_mem_ptr->d_ay, host_mem_ptr->ay, sizeof(int64_t)*total_n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_mem_ptr->d_start_idx, host_mem_ptr->start_idx, sizeof(size_t)*grid_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(device_mem_ptr->d_read_end_idx, host_mem_ptr->read_end_idx, sizeof(size_t)*grid_dim, cudaMemcpyHostToDevice);
    
    cudaMemcpy(device_mem_ptr->d_cut_start_idx, host_mem_ptr->cut_start_idx, sizeof(size_t)*grid_dim, cudaMemcpyHostToDevice);
    cudaMemset(device_mem_ptr->d_cut, 0xff, sizeof(size_t)*cut_num);
    cudaCheck();

    range_selection_kernel_naive<<<DimGrid0, DimBlock0>>>(device_mem_ptr->d_ax, device_mem_ptr->d_start_idx, device_mem_ptr->d_read_end_idx, 
                                                                device_mem_ptr->d_range, device_mem_ptr->d_cut, device_mem_ptr->d_cut_start_idx);
    cudaCheck();

    Misc misc_info;
    misc_info.bw = bw;
    misc_info.max_skip = max_skip;
    misc_info.max_iter = max_iter;
    misc_info.max_dist_x = max_dist_x;
    misc_info.max_dist_y = max_dist_y;
    misc_info.is_cdna = is_cdna;
    misc_info.chn_pen_gap = chn_pen_gap;
    misc_info.chn_pen_skip = chn_pen_skip;
    misc_info.n_seg = n_seg;

    #ifdef USEHIP
    hipMemcpyToSymbol(HIP_SYMBOL(misc), &misc_info, sizeof(Misc), 0, cudaMemcpyHostToDevice);
    #else
    cudaMemcpyToSymbol(misc, &misc_info, sizeof(Misc), 0, cudaMemcpyHostToDevice);
    #endif

    cudaCheck();
    int griddim = (cut_num-1)/NUM_SEG_PER_BLOCK + 1;
    dim3 DimBlock1(NUM_THREADS_SCORE, 1, 1);
    dim3 DimGrid1(griddim, 1, 1);
    score_generation_naive<<<DimGrid1, DimBlock1>>>(device_mem_ptr->d_ax, device_mem_ptr->d_ay, device_mem_ptr->d_range, 
                                            device_mem_ptr->d_cut, device_mem_ptr->d_f, device_mem_ptr->d_p, total_n, cut_num);
    cudaCheck();

    // copy f and p back to host
    cudaMemcpy(host_mem_ptr->f, device_mem_ptr->d_f, sizeof(int32_t)*total_n, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_mem_ptr->p, device_mem_ptr->d_p, sizeof(uint16_t)*total_n, cudaMemcpyDeviceToHost);
    
    // NOTE: reset the grid size 
    grid_dim = 0;
    cut_num = 0;
    
    return 0;
}


mm128_t *mg_plchain_dp(
    int max_dist_x, int max_dist_y, int bw, int max_skip, int max_iter,
    int min_cnt, int min_sc, float chn_pen_gap, float chn_pen_skip, int is_cdna,
    int n_seg, int64_t n,  // NOTE: n is number of anchors
    mm128_t *a,            // NOTE: a is ptr to anchors.
    int *n_u_, uint64_t **_u,
    void *km) {  // TODO: make sure this works when n has more than 32 bits

    pthread_mutex_lock(&pltask_lock);
	pltask_append(n, a, awaiting_tasks); // NOTE: append anchors to pin memory
    awaiting_tasks ++; 
    total_tasks ++;
    if (awaiting_tasks == size || total_tasks == total_frags) { // TODO: when it is not a multiple of n_threads
        fprintf(stderr, "[M: %s] Launch chaining kernel\n", __func__);
		pltask_launch(max_dist_x, max_dist_y, bw, max_skip, max_iter, chn_pen_gap, chn_pen_skip, is_cdna, n_seg);
        awaiting_tasks = 0;
        pthread_cond_broadcast(&pltask_cv);
    } else {
        pthread_cond_wait(&pltask_cv, &pltask_lock);
    }
    pthread_mutex_unlock(&pltask_lock);
    fprintf(stderr, "[M: %s] ready to continue\n", __func__);

}