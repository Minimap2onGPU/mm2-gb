#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "plchain.h"
#include "debug.h"

/* 

Parallel chaining helper functions with CUDA

*/

/* kernels begin */

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

// __global__ void range_selection_kernel_naive(const int64_t* ax, size_t *start_idx_arr, size_t *end_idx_arr, size_t *read_end_idx_arr, int32_t *range){
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;

//     size_t start_idx = start_idx_arr[bid];
//     size_t end_idx = end_idx_arr[bid];
//     size_t read_end_idx = read_end_idx_arr[bid];
    
//     for (size_t i = start_idx + tid; i < end_idx; i += NUM_THREADS_RANGE){
//         size_t st = i + MAX_ITER;
//         st = i + MAX_ITER < read_end_idx ? st : read_end_idx -1;
//         while (st > i && (ax[i] >> 32 != ax[st] >> 32  // NOTE: different prefix cannot become predecessor 
//                 || ax[st] > ax[i] + MAX_DIST_X)) { // NOTE: same prefix compare the value
//             --st;
//         }
//         range[i] = st - i;
//     }
// }

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

/* kernels end */

/* host functions begin */

void stream_range_selection(input_iter input_arr[INPUT_BATCH_SIZE], 
                        hostMemPtr* host_mem_ptr, deviceMemPtr* device_mem_ptr, int size, void* stream_) {
    
    cudaStream_t* stream = (cudaStream_t*) stream_; 

    dim3 DimBlock(NUM_THREADS_RANGE, 1, 1);

    /* reorganize anchor data */
    size_t total_n = 0;
    for (int i=0; i < size; i++){
        total_n += input_arr[i].n;
    }
    device_mem_ptr->total_n = total_n;

    size_t griddim = 0;

    size_t idx = 0;
    size_t cut_num = 0;
    for (int i=0; i<size; i++){
        int n = input_arr[i].n;
        int block_num = (n - 1) / MAX_ANCHOR_PER_BLOCK + 1;

        host_mem_ptr->start_idx[griddim] = idx;
        size_t end_idx = idx + MAX_ANCHOR_PER_BLOCK;
        host_mem_ptr->read_end_idx[griddim] = idx + n;
        host_mem_ptr->cut_start_idx[griddim] = cut_num;
        for (int j=1; j<block_num; j++){
            cut_num += MAX_IT_BLOCK_RANGE;
            host_mem_ptr->start_idx[griddim + j] = end_idx;
            end_idx = host_mem_ptr->start_idx[griddim + j] + MAX_ANCHOR_PER_BLOCK;
            host_mem_ptr->read_end_idx[griddim + j] = idx + n;
            host_mem_ptr->cut_start_idx[griddim + j] = cut_num;
        }
        cut_num += (n - (block_num -1) * MAX_ANCHOR_PER_BLOCK - 1) / NUM_THREADS_RANGE + 1;
        end_idx = idx + n;

        griddim += block_num;
        // FIXME: temperally disable this
        // // sanity check
        // if (griddim >= MAX_GRID) break;

        for (int j =0; j < n; j++){
            host_mem_ptr->ax[idx] = input_arr[i].a[j].x;
            host_mem_ptr->ay[idx] = input_arr[i].a[j].y; 
            ++idx;
        }
    }
    device_mem_ptr->num_cut = cut_num;

    // FIXME: temperally disable this
    // // sanity check
    // if (griddim >= MAX_GRID) {
    //     fprintf(stderr, "exceed max grid, MAX_GRID=%d, need %zu\n", MAX_GRID, griddim);
    //     exit(0);
    // }

    dim3 DimGrid(griddim,1,1);
    // copy to device memory
    cudaMemcpyAsync(device_mem_ptr->d_ax, host_mem_ptr->ax, sizeof(int64_t)*total_n, cudaMemcpyHostToDevice, *stream);
    cudaMemcpyAsync(device_mem_ptr->d_ay, host_mem_ptr->ay, sizeof(int64_t)*total_n, cudaMemcpyHostToDevice, *stream);
    cudaMemcpyAsync(device_mem_ptr->d_start_idx, host_mem_ptr->start_idx, sizeof(size_t)*griddim, cudaMemcpyHostToDevice, *stream);
    cudaMemcpyAsync(device_mem_ptr->d_read_end_idx, host_mem_ptr->read_end_idx, sizeof(size_t)*griddim, cudaMemcpyHostToDevice, *stream);
    
    cudaMemcpyAsync(device_mem_ptr->d_cut_start_idx, host_mem_ptr->cut_start_idx, sizeof(size_t)*griddim, cudaMemcpyHostToDevice, *stream);
    cudaMemsetAsync(device_mem_ptr->d_cut, 0xff, sizeof(size_t)*cut_num, *stream);
    cudaCheck();

    // Run kernel
#ifdef DEBUG_VERBOSE
    printf("Grim Dim: %d Cut: %zu Anchors: %zu\n", DimGrid.x, cut_num, total_n);
#endif
    // range_selection_kernel<<<DimGrid, DimBlock>>>(d_ax, d_start_idx, d_end_idx, d_read_end_idx, d_range);
    range_selection_kernel_naive<<<DimGrid, DimBlock, 0, *stream>>>(device_mem_ptr->d_ax, device_mem_ptr->d_start_idx, device_mem_ptr->d_read_end_idx, 
                                                                device_mem_ptr->d_range, device_mem_ptr->d_cut, device_mem_ptr->d_cut_start_idx);
#ifdef DEBUG_VERBOSE
    printf("Kernel Launched\n");
#endif
    cudaCheck();
    // fprintf(stderr, "[M::%s] cut_num: %zu, total_n: %zu\n", __func__, cut_num, total_n);

    return;
}

a_pass range_selection(input_iter input_arr[INPUT_BATCH_SIZE], int size) {
    dim3 DimBlock(NUM_THREADS_RANGE,1,1);

    /* reorganize anchor data */
    size_t total_n = 0;
    for (int i=0; i<size; i++){
        total_n += input_arr[i].n;
    }
    
    int64_t* ax = (int64_t*)malloc(total_n * sizeof(int64_t));
    int64_t* ay = (int64_t*)malloc(total_n * sizeof(int64_t));

    size_t griddim = 0;
    size_t start_idx[MAX_GRID];
    size_t read_end_idx[MAX_GRID];
    size_t cut_start_idx[MAX_GRID];

    size_t idx = 0;
    size_t cut_num = 0;
    for (int i=0; i<size; i++){
        int n = input_arr[i].n;
        int block_num = (n - 1) / MAX_ANCHOR_PER_BLOCK + 1;

        start_idx[griddim] = idx;
        size_t end_idx = idx + MAX_ANCHOR_PER_BLOCK;
        read_end_idx[griddim] = idx + n;
        cut_start_idx[griddim] = cut_num;
        for (int j=1; j<block_num; j++){
            cut_num += MAX_IT_BLOCK_RANGE;
            start_idx[griddim + j] = end_idx;
            end_idx = start_idx[griddim + j] + MAX_ANCHOR_PER_BLOCK;
            read_end_idx[griddim + j] = idx + n;
            cut_start_idx[griddim + j] = cut_num;
        }
        cut_num += (n - (block_num -1) * MAX_ANCHOR_PER_BLOCK - 1) / NUM_THREADS_RANGE + 1;
        end_idx = idx + n;

        griddim += block_num;
        //sanity check
        if (griddim >= MAX_GRID) break;

        for (int j =0; j < n; j++){
            ax[idx] = input_arr[i].a[j].x;
            ay[idx] = input_arr[i].a[j].y; 
            ++idx;
        }
    }

    //DEBUG:
    // for (int i = 0; i < griddim; ++i){
    //     printf("block#%d, %lu - %lu, read end %lu\n", i, start_idx[i], end_idx[i], read_end_idx[i]);
    // }

    // sanity check
    if (griddim >= MAX_GRID) {
        fprintf(stderr, "exceed max grid, MAX_GRID=%d, need %zu\n", MAX_GRID, griddim);
        exit(0);
    }

    dim3 DimGrid(griddim,1,1);
    //DEBUG:

    // copy data to gpu
    int64_t *d_ax;
    int64_t *d_ay;
    size_t *d_start_idx, *d_read_end_idx;
    int32_t *d_range;
    cudaMalloc(&d_ax, sizeof(int64_t)*total_n);
    cudaMalloc(&d_ay, sizeof(int64_t)*total_n);
    cudaMalloc(&d_start_idx, sizeof(size_t)*griddim);
    cudaMalloc(&d_read_end_idx, sizeof(size_t)*griddim);
    cudaMalloc(&d_range, sizeof(int32_t)*total_n);
    cudaCheck();
    
    cudaMemcpy(d_ax, ax, sizeof(int64_t)*total_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ay, ay, sizeof(int64_t)*total_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_idx, start_idx, sizeof(size_t)*griddim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_read_end_idx, read_end_idx, sizeof(size_t)*griddim, cudaMemcpyHostToDevice);
    
    cudaCheck();

    size_t *d_cut_start_idx, *d_cut;
    cudaMalloc(&d_cut_start_idx, sizeof(size_t)*griddim);
    cudaMalloc(&d_cut, sizeof(size_t)*cut_num);
    cudaMemcpy(d_cut_start_idx, cut_start_idx, sizeof(size_t)*griddim, cudaMemcpyHostToDevice);
    cudaMemset(d_cut, 0xff, sizeof(size_t)*cut_num);
    cudaCheck();

    // Run kernel
#ifdef DEBUG_VERBOSE
    printf("Grim Dim: %d Cut: %zu Anchors: %zu\n", DimGrid.x, cut_num, total_n);
#endif
    // range_selection_kernel<<<DimGrid, DimBlock>>>(d_ax, d_start_idx, d_read_end_idx, d_range);
    range_selection_kernel_naive<<<DimGrid, DimBlock>>>(d_ax, d_start_idx, d_read_end_idx, d_range, d_cut, d_cut_start_idx);
#ifdef DEBUG_VERBOSE
    printf("Kernel Launched\n");
#endif
    cudaCheck();
    cudaDeviceSynchronize();
    cudaCheck();
#ifdef DEBUG_VERBOSE
    fprintf(stderr, "[M::%s] range calculation success\n", __func__);
#endif




#ifdef DEBUG_CHECK
    //check range

    int32_t* range = (int32_t*)malloc(sizeof(int32_t) * total_n);
    cudaMemcpy(range, d_range, sizeof(int32_t)*total_n, cudaMemcpyDeviceToHost);
#ifdef DEBUG_VERBOSE
    size_t* cut = (size_t*)malloc(sizeof(size_t)*cut_num);
    cudaMemcpy(cut, d_cut, sizeof(size_t)*cut_num, cudaMemcpyDeviceToHost);
    for (int readid=0, cid=0, idx=0; readid<size; readid++){
        // printf("Read#%d, start %lu, read_end %lu cut:", readid, idx, idx + input_arr[readid].n);
        while(cid < cut_num && (cut[cid] < idx + input_arr[readid].n || cut[cid] == SIZE_MAX)){
            // printf("%zu", cut[cid]);
            if(cut[cid] != SIZE_MAX){
                if (range[cut[cid]-1] != 0 && cut[cid] != 0) printf("[debug] Error Cut at %d (%d), (Read %d, %d - %d)\n", cut[cid], range[cut[cid]-1], readid, idx, idx + input_arr[readid].n);
                // printf("(%d) ", range[cut[cid]-1]);
            } else {
                // printf("(x) ");
                ;
            }
            cid++;
        }
        // printf("\n");
        idx += input_arr[readid].n;
    }
#endif
    int64_t read_start = 0;
    for (int i = 0; i<size; i++){
        debug_print_successor_range(range+read_start, input_arr[i].n);
        debug_check_range(range + read_start, input_arr[i].range, input_arr[i].n);
        read_start += input_arr[i].n;
    }
    free(range);
#endif
    a_pass pass_ptr;
    // cudaFree(d_ax);
    // cudaFree(d_ay);
    

    cudaFree(d_start_idx);
    cudaFree(d_read_end_idx);
    cudaFree(d_cut_start_idx);
    // cudaFree(d_range);
    
    pass_ptr.x = d_ax;
    pass_ptr.y = d_ay;
    pass_ptr.range = d_range;
    pass_ptr.cut = d_cut;
    pass_ptr.num_cut = cut_num;
    pass_ptr.total_n = total_n;

    free(ax);
    free(ay);

    return pass_ptr;
}

/* host functions end */
