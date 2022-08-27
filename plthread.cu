#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>

#include "plchain.h"
#include "plthread.h"
#include "debug.h"
#include "kthread.h"

/* 

Thread callback functions with CUDA

*/

void hipkt_for(int n_threads, int (*func)(void*,long,int), void *data, long n) {
    // NOTE: func = hipworker_for, n_threads is input of mm_map_file_frag(), n is number of frags
    // n_threads = 1; // FIXME: comment this to enable multithread 
    fprintf(stderr, "[M: %s] kt_for %ld segs on %d threads\n", __func__, n, n_threads);
    if (n_threads > 1) {
        int i;
		kt_for_t t; // NOTE: multithreads' metadata
		pthread_t *tid;
		t.func = func, t.data = data, t.n_threads = n_threads, t.n = n;
		t.w = (ktf_worker_t*)calloc(n_threads, sizeof(ktf_worker_t));
		tid = (pthread_t*)calloc(n_threads, sizeof(pthread_t));
		for (i = 0; i < n_threads; ++i)
			t.w[i].t = &t, t.w[i].i = i;
		for (i = 0; i < n_threads; ++i) pthread_create(&tid[i], 0, ktf_worker, &t.w[i]);
		for (i = 0; i < n_threads; ++i) pthread_join(tid[i], 0);
		free(tid); free(t.w);
    } else {
        long j;
		for (j = 0; j < n; ++j) func(data, j, 0);
    }
}





