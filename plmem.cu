#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>

#include "plchain.h"
#include "plthread.h"
#include "debug.h"

/*
    chaining task types and task lists
*/

enum Status {
    EMPTY, // empty slot
    IDLE, // task hasn't started yet
    TASK_ON, // task is running
    TASK_END, // task is done
};

struct task_chain_t {
    int64_t *a;
	int index; // index where anchors were appended to and where to find f, p 
    int size; // batch size
    Status status; // status of the task
};

struct task_align_t {
	// TODO: waiting to be filled
    Status status; // status of the task
};

static task_chain_t *chaining_tasks;
static task_align_t *alignment_tasks;
static int64_t chain_index; // += size for every chaining task
static int64_t align_index; // += size for every alignment task
static bool gpu_busy;

int pltask_init(int num_seqs) {
    // TODO: compute gpu memory size and initialize the tasks list 
    gpu_busy = false;
    chain_index = align_index = 0;
    chaining_tasks = (task_chain_t *) malloc(sizeof(task_chain_t)*num_seqs);
    alignment_tasks = (task_align_t *) malloc(sizeof(task_align_t)*num_seqs);
}

/*********************** Thread function calls start ************************/

int plchain_append(int64_t n_a) {

    // TODO: increment the index 
    // NOTE: No consistence violation as sequence index on each cpu thread is independent

    return 0;
}

int plchain_check(size_t index) {

    Status status = chaining_tasks[index].status;
    switch (status)
    {
    case IDLE:
        // TODO: start gpu if gpu_busy is false
        // change gpu_busy to true
        return -1;

    case TASK_ON:
        return 1;

    case TASK_END:
        // TODO: misc after gpu is done
        return 0;
    default:
        break;
    }


    return 0; 
    // TODO: return 0 if gpu is done,
    // return -1 if gpu just starts
    // return 1 if gpu has start but not finished yet  
}

// TODO: alignment tasks append and check




/*********************** Thread function calls end ************************/

/*********************** GPU loops start ************************/

double dynamic_stream_chain_loop(input_iter* input_arr, int total_reads) {
    // NOTE: return duration of this cpu batch
    // NUM_STREAMS must be more than one
    assert(NUM_STREAMS > 1);
    hostMemPtr host_mem_ptrs[NUM_STREAMS];
    deviceMemPtr device_mem_ptrs[NUM_STREAMS];
    double dura = 0.0;
    clock_t clk_start, clk_end;
    clk_start = time(NULL);
    // clk_start = clock();

    // set up stream and memory pointers
    cudaStream_t * streams = new cudaStream_t[NUM_STREAMS]; // init streams
    cudaEvent_t * events = new cudaEvent_t[NUM_STREAMS];
    // FIXME: is it possible for a read to be too long to fit in one stream?
    size_t avail_mem_stream = MEM_GPU;
    avail_mem_stream = MEM_GPU/NUM_STREAMS * 1e9; // split memory for each stream
#ifdef DEBUG_CHECK
    fprintf(stderr, "[M: %s] memory per stream: %zuB\n", __func__, avail_mem_stream);    
#endif // DEBUG_CHECK
    // memory per anchor = ax + ay + range + f + p + (start_idx + read_end_idx + cut_start_idx + cut)
    // size: F1 = ax + ay + range + f + p; F2 = start_idx + read_end_idx + cut_start_idx; F3 = cut
    int64_t F1 = 8+8+4+4+2, F2 = 8+8+8, F3 = 8;
    int64_t P1 = ANCHORS_PER_BLOCK, P2 = ANCHORS_PER_CUT;
    // avail_memory = (F1 + F2/ANCHORS_PER_BLOCK + F3/ANCHORS_PER_CUT) * num_anchors
    int64_t max_anchors_stream = (avail_mem_stream*P1*P2) / (F1*P1*P2 + F2*P2 + F3*P1); // ignore misc as anchors cannot just fit whole memory
    int64_t max_grid = max_anchors_stream / ANCHORS_PER_BLOCK;
    int64_t max_num_cut = max_anchors_stream / ANCHORS_PER_CUT;
#ifdef DEBUG_CHECK
    fprintf(stderr, "[M: %s] per stream max_anchors: %ld, max_grid: %ld, max_cut: %ld\n", __func__, max_anchors_stream, max_grid, max_num_cut);    
#endif // DEBUG_CHECK

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
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

    // schedule streams until all reads finished 
    int processed_reads = 0; // reads processed so far
    input_iter* curr_arr = input_arr;
    int batch_count = 0; // batchs launched so far
    while (processed_reads < total_reads) {
        int read_idx;
        int64_t num_anchors = 0;
        // assign anchors for stream
        for (read_idx = processed_reads; read_idx < total_reads; ++read_idx) {
            if (num_anchors + input_arr[read_idx].n > max_anchors_stream) {
            #ifdef DEBUG_CHECK
                fprintf(stderr, "[M: %s] stream_mem fills up after %d reads and %ld anchors\n", __func__, read_idx - processed_reads, num_anchors);    
            #endif // DEBUG_CHECK
                break;
            }
            num_anchors += input_arr[read_idx].n; // increment only when memory is available
        }

        int stream_idx = batch_count;
        // find usable stream
        if (batch_count >= NUM_STREAMS) {
            stream_idx = -1;
            while (stream_idx == -1) {
                for (int t = 0; t < NUM_STREAMS; ++t) {
                    if (!cudaEventQuery(events[t])) {
                        stream_idx = t;
                        // FIXME: unnecessary recreate?
                        cudaEventDestroy(events[stream_idx]);
                        cudaEventCreate(&events[stream_idx]);
                        break;
                    }
                }
            }
            cudaCheck();
#ifdef DEBUG_CHECK
            int sync_iter = stream_idx;
            // TODO: check correctness
            int index = host_mem_ptrs[sync_iter].index; // index here is read index
            // fprintf(stderr, "[M::%s] correctness check index: %d\n", __func__, index);
            int64_t read_start = 0;
            input_iter *prev_arr = input_arr + index;
            // range check
            if (is_debug_range() != -1) {
                int32_t* range = (int32_t*)malloc(sizeof(int32_t) * device_mem_ptrs[sync_iter].total_n);
                cudaMemcpy(range, device_mem_ptrs[sync_iter].d_range, sizeof(int32_t)*device_mem_ptrs[sync_iter].total_n, cudaMemcpyDeviceToHost);
                // fprintf(stderr, "[M::%s] Start range check\n", __func__);
                for (int i = 0; i < host_mem_ptrs[sync_iter].size; i++){
                    debug_print_successor_range(range+read_start, prev_arr[i].n);
                    debug_check_range(range + read_start, prev_arr[i].range, prev_arr[i].n);
                    read_start += prev_arr[i].n;
                }
                free(range);
            }
            // score check
            fprintf(stderr, "[M::%s] Start score check of %d reads\n", __func__, host_mem_ptrs[sync_iter].size);
            read_start = 0;
            for (int i = 0; i < host_mem_ptrs[sync_iter].size; i++) {
                // int64_t* p = (int64_t*)malloc(sizeof(int64_t)*prev_arr[i].n);
                // p_rel2idx(host_mem_ptrs[sync_iter].p + read_start, p, prev_arr[i].n);
                // debug_check_score(p, host_mem_ptrs[sync_iter].f + read_start, prev_arr[i].p, prev_arr[i].f, prev_arr[i].n);
                debug_check_score_auto_trans(host_mem_ptrs[sync_iter].p + read_start, host_mem_ptrs[sync_iter].f + read_start, prev_arr[i].p, prev_arr[i].f, prev_arr[i].n);
                read_start += prev_arr[i].n;
            }
            fprintf(stderr, "[M::%s] End score check\n", __func__);
#endif
            // TODO: read p, f
            // host_mem_ptrs[sync_iter].f
            // host_mem_ptrs[sync_iter].p
        }
        batch_count++;
        int batch_size = read_idx - processed_reads;
        host_mem_ptrs[stream_idx].index = processed_reads; // record read index
        host_mem_ptrs[stream_idx].size = batch_size; // record batch size (num of reads processed)
        device_mem_ptrs[stream_idx].total_n = num_anchors;
        fprintf(stderr, "[M::%s] Start stream with %d reads\n", __func__, batch_size);
        stream_range_selection(curr_arr, host_mem_ptrs + stream_idx, device_mem_ptrs + stream_idx, batch_size, (void *)(&streams[stream_idx]));
        cudaCheck();
        stream_score_generation(curr_arr, host_mem_ptrs + stream_idx, device_mem_ptrs + stream_idx, batch_size, (void *)(&streams[stream_idx]), (void *)(&events[stream_idx]));
        cudaCheck();
        curr_arr += batch_size;
        processed_reads = read_idx;
#ifdef DEBUG_CHECK
        fprintf(stderr, "[M: %s] -----reads in progress %d/%d----- \n", __func__, processed_reads, total_reads);    
#endif // DEBUG_CHECK
    }

    fprintf(stderr, "[M::%s] Sync up all streams\n", __func__);
    // sync all the streams
    for (int sync_iter = 0; sync_iter < NUM_STREAMS; ++sync_iter) {
        int index = host_mem_ptrs[sync_iter].index;
        // fprintf(stderr, "[M::%s] final sync all index: %d, sync_iter: %d\n", __func__, index, sync_iter);
        if (index != -1) {
            cudaStreamSynchronize(streams[sync_iter]);
            cudaCheck();
            // TODO: check correctness
#ifdef DEBUG_CHECK
            // fprintf(stderr, "[M::%s] correctness check index: %d, batch_size: %d\n", __func__, index, batch_size);
            int64_t read_start = 0;
            input_iter *prev_arr = input_arr + index;
            // range check
            if (is_debug_range() != -1) {
                int32_t* range = (int32_t*)malloc(sizeof(int32_t) * device_mem_ptrs[sync_iter].total_n);
                cudaMemcpy(range, device_mem_ptrs[sync_iter].d_range, sizeof(int32_t)*device_mem_ptrs[sync_iter].total_n, cudaMemcpyDeviceToHost);
                // fprintf(stderr, "[M::%s] Start range check\n", __func__);
                for (int i = 0; i < host_mem_ptrs[sync_iter].size; i++){
                    debug_print_successor_range(range+read_start, prev_arr[i].n);
                    debug_check_range(range + read_start, prev_arr[i].range, prev_arr[i].n);
                    read_start += prev_arr[i].n;
                }
                free(range);
            }
            // score check
            read_start = 0;
            fprintf(stderr, "[M::%s] Start score check of %d reads\n", __func__, host_mem_ptrs[sync_iter].size);
            for (int i = 0; i < host_mem_ptrs[sync_iter].size; i++) {
                // NOTE: this temp buffer needs to be considered in memory alloc
                // int64_t* p = (int64_t*)malloc(sizeof(int64_t)*prev_arr[i].n);
                // p_rel2idx(host_mem_ptrs[sync_iter].p + read_start, p, prev_arr[i].n);
                // debug_check_score(p, host_mem_ptrs[sync_iter].f + read_start, prev_arr[i].p, prev_arr[i].f, prev_arr[i].n);
                debug_check_score_auto_trans(host_mem_ptrs[sync_iter].p + read_start, host_mem_ptrs[sync_iter].f + read_start, prev_arr[i].p, prev_arr[i].f, prev_arr[i].n);
                read_start += prev_arr[i].n;
            }
            fprintf(stderr, "[M::%s] End score check\n", __func__);
#endif
        } else break;
    }

    // free all memory
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
        cudaCheck();

        // free host memory
        cudaFreeHost(host_mem_ptrs[i].ax);
        cudaFreeHost(host_mem_ptrs[i].ay);
        cudaFreeHost(host_mem_ptrs[i].f);
        cudaFreeHost(host_mem_ptrs[i].p);
        cudaFreeHost(host_mem_ptrs[i].start_idx);
        cudaFreeHost(host_mem_ptrs[i].read_end_idx);
        cudaFreeHost(host_mem_ptrs[i].cut_start_idx);
        // free device memory
        cudaFree(device_mem_ptrs[i].d_ax);
        cudaFree(device_mem_ptrs[i].d_ay);
        cudaFree(device_mem_ptrs[i].d_range);
        cudaFree(device_mem_ptrs[i].d_f);
        cudaFree(device_mem_ptrs[i].d_p);
        cudaFree(device_mem_ptrs[i].d_cut);
        cudaFree(device_mem_ptrs[i].d_start_idx);
        cudaFree(device_mem_ptrs[i].d_read_end_idx);
        cudaFree(device_mem_ptrs[i].d_cut_start_idx);
        cudaCheck();
    }
    delete[] streams;
    delete[] events;

    clk_end = time(NULL);
    // clk_end = clock();
    dura =  (double) (clk_end - clk_start);
    fprintf(stderr, "[Dynamic stream chaining] ======CPU Batch Chaining Run Time: %lf secs\n", dura);
    // printf("[Stream chaining] ======Chaining Run Time: %lf secs\n", ((double) (clk_end - clk_start)) / CLOCKS_PER_SEC);
    return dura;

}

void stream_chain_loop(input_iter* input_arr, int total_iter) {
    // NUM_STREAMS must be more than one
    assert(NUM_STREAMS > 1);
    hostMemPtr host_mem_ptrs[NUM_STREAMS];
    deviceMemPtr device_mem_ptrs[NUM_STREAMS];
    double range_selection_dura = 0.0, score_generation_dura = 0.0, cpu_chain_dura = 0.0;
    clock_t clk_start, clk_end;
    clk_start = time(NULL);
    // clk_start = clock();
    
    // set up stream and memory pointers
    cudaStream_t * streams = new cudaStream_t[NUM_STREAMS]; // init streams
    cudaEvent_t * events = new cudaEvent_t[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        cudaCheck();

        // NOTE: ax ay range require dynamic size, others have max size limitation
        // set up host memory pointers
        host_mem_ptrs[i].index = -1;
        cudaMallocHost((void**)&host_mem_ptrs[i].ax, MAX_NUM_ANCHORS * sizeof(int64_t));
        cudaMallocHost((void**)&host_mem_ptrs[i].ay, MAX_NUM_ANCHORS * sizeof(int64_t));
        cudaMallocHost((void**)&host_mem_ptrs[i].f, MAX_NUM_ANCHORS * sizeof(int32_t));
        cudaMallocHost((void**)&host_mem_ptrs[i].p, MAX_NUM_ANCHORS * sizeof(uint16_t));
        cudaMallocHost((void**)&host_mem_ptrs[i].start_idx, MAX_GRID * sizeof(size_t));
        cudaMallocHost((void**)&host_mem_ptrs[i].read_end_idx, MAX_GRID * sizeof(size_t));
        cudaMallocHost((void**)&host_mem_ptrs[i].cut_start_idx, MAX_GRID * sizeof(size_t));
        // set up GPU memory pointers
        cudaMalloc(&device_mem_ptrs[i].d_ax, MAX_NUM_ANCHORS * sizeof(int64_t));
        cudaMalloc(&device_mem_ptrs[i].d_ay, MAX_NUM_ANCHORS * sizeof(int64_t));
        cudaMalloc(&device_mem_ptrs[i].d_range, MAX_NUM_ANCHORS * sizeof(int32_t));
        cudaMalloc(&device_mem_ptrs[i].d_f, MAX_NUM_ANCHORS * sizeof(int32_t));
        cudaMalloc(&device_mem_ptrs[i].d_p, MAX_NUM_ANCHORS * sizeof(uint16_t));    
        cudaMalloc(&device_mem_ptrs[i].d_cut, MAX_NUM_CUT * sizeof(size_t));
        cudaMalloc(&device_mem_ptrs[i].d_start_idx, sizeof(size_t)*MAX_GRID);
        cudaMalloc(&device_mem_ptrs[i].d_read_end_idx, sizeof(size_t)*MAX_GRID);
        cudaMalloc(&device_mem_ptrs[i].d_cut_start_idx, sizeof(size_t)*MAX_GRID);
        cudaCheck();
    }

    int count = total_iter / INPUT_BATCH_SIZE;
    input_iter* curr_arr = input_arr;
    
    // launch streams
    for (int i = 0; i < count; ++i) {
        // fprintf(stderr, "[M::%s] normal round i: %d\n", __func__, i);
        int iter = i;
        if (i >= NUM_STREAMS) {
            // int sync_iter = (i-NUM_STREAMS) % NUM_STREAMS;
            iter = -1;
            while (iter == -1) {
                for (int t = 0; t < NUM_STREAMS; ++t) {
                    if (!cudaEventQuery(events[t])) {
                        iter = t;
                        cudaEventDestroy(events[iter]);
                        cudaEventCreate(&events[iter]);
                        break;
                    }
                }
            }

            // NOTE: streams fill up, sync previous stream
            // fprintf(stderr, "[M::%s] sync stream sync_iter: %d\n", __func__, sync_iter);
            // cudaStreamSynchronize(streams[iter]);
            cudaCheck();
#ifdef DEBUG_CHECK
            int sync_iter = iter;
            // TODO: check correctness
            int index = host_mem_ptrs[sync_iter].index;
            // fprintf(stderr, "[M::%s] correctness check index: %d\n", __func__, index);
            input_iter *prev_arr = input_arr + index*INPUT_BATCH_SIZE;
            // range check
            int32_t* range = (int32_t*)malloc(sizeof(int32_t) * device_mem_ptrs[sync_iter].total_n);
            cudaMemcpy(range, device_mem_ptrs[sync_iter].d_range, sizeof(int32_t)*device_mem_ptrs[sync_iter].total_n, cudaMemcpyDeviceToHost);
            int64_t read_start = 0;
            // fprintf(stderr, "[M::%s] Start range check\n", __func__);
            for (int i = 0; i < INPUT_BATCH_SIZE; i++){
                debug_print_successor_range(range+read_start, prev_arr[i].n);
                debug_check_range(range + read_start, prev_arr[i].range, prev_arr[i].n);
                read_start += prev_arr[i].n;
            }
            free(range);
            // score check
            // fprintf(stderr, "[M::%s] Start score check\n", __func__);
            read_start = 0;
            for (int i = 0; i < INPUT_BATCH_SIZE; i++) {
                int64_t* p = (int64_t*)malloc(sizeof(int64_t)*prev_arr[i].n);
                p_rel2idx(host_mem_ptrs[sync_iter].p + read_start, p, prev_arr[i].n);
                debug_check_score(p, host_mem_ptrs[sync_iter].f + read_start, prev_arr[i].p, prev_arr[i].f, prev_arr[i].n);
                read_start += prev_arr[i].n;
                free(p);
            }
            // fprintf(stderr, "[M::%s] End score check\n", __func__);
#endif
            // TODO: read p, f
            // host_mem_ptrs[sync_iter].f
            // host_mem_ptrs[sync_iter].p
        }

        
        host_mem_ptrs[iter].index = i; // record batch index
        stream_range_selection(curr_arr, host_mem_ptrs + iter, device_mem_ptrs + iter, INPUT_BATCH_SIZE, (void *)(&streams[iter]));
        cudaCheck();
        stream_score_generation(curr_arr, host_mem_ptrs + iter, device_mem_ptrs + iter, INPUT_BATCH_SIZE, (void *)(&streams[iter]), (void *)(&events[iter]));
        cudaCheck();
        curr_arr += INPUT_BATCH_SIZE;

    }

    // launch extra batch shorter than INPUT_BATCH_SIZE
    if (count*INPUT_BATCH_SIZE < total_iter) {
        // fprintf(stderr, "[M::%s] extra round count: %d\n", __func__, count);
        int iter = count;
        if (count >= NUM_STREAMS) {
            // int sync_iter = (count-NUM_STREAMS) % NUM_STREAMS;
            iter = -1;
            while (iter == -1) {
                for (int t = 0; t < NUM_STREAMS; ++t) {
                    if (!cudaEventQuery(events[t])) {
                        iter = t;
                        cudaEventDestroy(events[iter]);
                        cudaEventCreate(&events[iter]);
                        break;
                    }
                }
            }
            // NOTE: streams fill up, sync previous stream
            // fprintf(stderr, "[M::%s] sync stream sync_iter: %d\n", __func__, sync_iter);
            // cudaStreamSynchronize(streams[iter]);
            cudaCheck();
#ifdef DEBUG_CHECK
            int sync_iter = iter;
            // TODO: check correctness
            int index = host_mem_ptrs[sync_iter].index;
            // fprintf(stderr, "[M::%s] correctness check index: %d\n", __func__, index);
            input_iter *prev_arr = input_arr + index*INPUT_BATCH_SIZE;
            // range check
            int32_t* range = (int32_t*)malloc(sizeof(int32_t) * device_mem_ptrs[sync_iter].total_n);
            cudaMemcpy(range, device_mem_ptrs[sync_iter].d_range, sizeof(int32_t)*device_mem_ptrs[sync_iter].total_n, cudaMemcpyDeviceToHost);
            int64_t read_start = 0;
            // fprintf(stderr, "[M::%s] Start range check\n", __func__);
            for (int i = 0; i < INPUT_BATCH_SIZE; i++){
                debug_print_successor_range(range+read_start, prev_arr[i].n);
                debug_check_range(range + read_start, prev_arr[i].range, prev_arr[i].n);
                read_start += prev_arr[i].n;
            }
            free(range);
            // score check
            read_start = 0;
            // fprintf(stderr, "[M::%s] Start score check\n", __func__);
            for (int i = 0; i < INPUT_BATCH_SIZE; i++) {
                int64_t* p = (int64_t*)malloc(sizeof(int64_t)*prev_arr[i].n);
                p_rel2idx(host_mem_ptrs[sync_iter].p + read_start, p, prev_arr[i].n);
                debug_check_score(p, host_mem_ptrs[sync_iter].f + read_start, prev_arr[i].p, prev_arr[i].f, prev_arr[i].n);
                read_start += prev_arr[i].n;
                free(p);
            }
            // fprintf(stderr, "[M::%s] End score check\n", __func__);
#endif
            // TODO: read p, f
            // host_mem_ptrs[sync_iter].f
            // host_mem_ptrs[sync_iter].p
        
        }
        // run the last kernel sets
        int size = total_iter - count*INPUT_BATCH_SIZE;
        host_mem_ptrs[iter].index = count; // record batch index
        stream_range_selection(curr_arr, host_mem_ptrs + iter, device_mem_ptrs + iter, size, (void *)(&streams[iter]));
        cudaCheck();
        stream_score_generation(curr_arr, host_mem_ptrs + iter, device_mem_ptrs + iter, size, (void *)(&streams[iter]), (void *)(&events[iter]));
        cudaCheck();
    }

    // sync all the streams
    for (int sync_iter = 0; sync_iter < NUM_STREAMS; ++sync_iter) {
        int index = host_mem_ptrs[sync_iter].index;
        // fprintf(stderr, "[M::%s] final sync all index: %d, sync_iter: %d\n", __func__, index, sync_iter);
        if (index >= 0) {
            cudaStreamSynchronize(streams[sync_iter]);
            cudaCheck();
            // TODO: check correctness
#ifdef DEBUG_CHECK
            int batch_size = total_iter >= (index+1)*INPUT_BATCH_SIZE ? INPUT_BATCH_SIZE : (index+1)*INPUT_BATCH_SIZE-total_iter;
            // fprintf(stderr, "[M::%s] correctness check index: %d, batch_size: %d\n", __func__, index, batch_size);
            input_iter *prev_arr = input_arr + index*INPUT_BATCH_SIZE;
            // range check
            int32_t* range = (int32_t*)malloc(sizeof(int32_t) * device_mem_ptrs[sync_iter].total_n);
            cudaMemcpy(range, device_mem_ptrs[sync_iter].d_range, sizeof(int32_t)*device_mem_ptrs[sync_iter].total_n, cudaMemcpyDeviceToHost);
            int64_t read_start = 0;
            // fprintf(stderr, "[M::%s] Start range check\n", __func__);
            for (int i = 0; i < batch_size; i++){
                debug_print_successor_range(range+read_start, prev_arr[i].n);
                debug_check_range(range + read_start, prev_arr[i].range, prev_arr[i].n);
                read_start += prev_arr[i].n;
            }
            free(range);
            // score check
            read_start = 0;
            // fprintf(stderr, "[M::%s] Start score check\n", __func__);
            for (int i = 0; i < batch_size; i++) {
                int64_t* p = (int64_t*)malloc(sizeof(int64_t)*prev_arr[i].n);
                p_rel2idx(host_mem_ptrs[sync_iter].p + read_start, p, prev_arr[i].n);
                debug_check_score(p, host_mem_ptrs[sync_iter].f + read_start, prev_arr[i].p, prev_arr[i].f, prev_arr[i].n);
                read_start += prev_arr[i].n;
                free(p);
            }
            // fprintf(stderr, "[M::%s] End score check\n", __func__);
#endif
        } else break;
    }

    // free all memory
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
        cudaCheck();

        // free host memory
        cudaFreeHost(host_mem_ptrs[i].ax);
        cudaFreeHost(host_mem_ptrs[i].ay);
        cudaFreeHost(host_mem_ptrs[i].f);
        cudaFreeHost(host_mem_ptrs[i].p);
        cudaFreeHost(host_mem_ptrs[i].start_idx);
        cudaFreeHost(host_mem_ptrs[i].read_end_idx);
        cudaFreeHost(host_mem_ptrs[i].cut_start_idx);
        // free device memory
        cudaFree(device_mem_ptrs[i].d_ax);
        cudaFree(device_mem_ptrs[i].d_ay);
        cudaFree(device_mem_ptrs[i].d_range);
        cudaFree(device_mem_ptrs[i].d_f);
        cudaFree(device_mem_ptrs[i].d_p);
        cudaFree(device_mem_ptrs[i].d_cut);
        cudaFree(device_mem_ptrs[i].d_start_idx);
        cudaFree(device_mem_ptrs[i].d_read_end_idx);
        cudaFree(device_mem_ptrs[i].d_cut_start_idx);
        cudaCheck();
    }
    delete[] streams;
    delete[] events;

    clk_end = time(NULL);
    // clk_end = clock();
    printf("[Stream chaining] ======Chaining Run Time: %lf secs\n", (double) (clk_end - clk_start));
    // printf("[Stream chaining] ======Chaining Run Time: %lf secs\n", ((double) (clk_end - clk_start)) / CLOCKS_PER_SEC);
    return;
}

/*********************** GPU loops end ************************/

