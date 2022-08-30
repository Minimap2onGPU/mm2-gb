#ifndef __PLTHREAD_H__
#define __PLTHREAD_H__ 

#include <assert.h>
#include "minimap.h"
#include "bseq.h"
#include "kseq.h"
#include "hipify.h"
#include "plutils.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
    GPU task types
*/

enum Status {
    EMPTY, // empty slot
    IDLE, // task hasn't started yet
    TASK_ON, // task is running
    TASK_END, // task is done
};

struct task_chain_t {
    int i; // index of the qry sequence 
	int offset; // index where anchors were appended to and where to find f, p 
    int size; // batch size
    Status status; // status of the task
    int n_regs0, n_mini_pos;
	uint32_t hash;
	uint64_t *u, *mini_pos;
	mm128_t *a; // anchors data
    Misc misc; // misc information
    int32_t *f;
    int64_t *p;
    int32_t *t;
};

struct task_align_t {
	// TODO: waiting to be filled
    int i; // index of the qry sequence 
	int offset; // index where anchors were appended to and where to find f, p 
    int size; // batch size
    Status status; // status of the task
    int n_regs0, n_mini_pos;
	uint32_t hash;
	uint64_t *u, *mini_pos;
	mm128_t *a; // anchors data
    Misc misc; // misc information
};

/*
    GPU task function call declarations
*/

int pltask_init(int num_seqs);
const task_chain_t *pltask_chain_get(long i);
const task_align_t *pltask_align_get(long i);

int plchain_append(int max_dist_x, int max_dist_y, const mm_mapopt_t *opt,
    float chn_pen_gap, float chn_pen_skip, int is_cdna,
    int n_seg, int64_t n,  // NOTE: n is number of anchors
    mm128_t *a,            // NOTE: a is ptr to anchors.
    void *km, int n_mini_pos, uint64_t *mini_pos, uint32_t hash, long i); // TODO: make sure this works when n has more than 32 bits

int plchain_check(long i);





#ifdef __cplusplus
}
#endif

#endif // !__PLTHREAD_H__