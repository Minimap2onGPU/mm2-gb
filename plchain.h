#ifndef _PLCHAIN_H_
#define _PLCHAIN_H_

#include <assert.h>
#include "minimap.h"
#include "bseq.h"
#include "kseq.h"

#ifdef __cplusplus
extern "C" {
#endif

// CUDA functions
void range_selection();
void forward_dp();

// debug functions
void debug_print(int32_t *f, int32_t *t, int32_t *v, int64_t *p, int64_t n);
void debug_fprint(int32_t *f, int32_t *t, int32_t *v, int64_t *p, int64_t n);
void debug_chain_fprint(int32_t n_u, uint64_t *u);
void debug_chain_input(mm128_t *a, int64_t n, int max_iter, int max_dist_x, int max_dist_y, int max_skip, int bw, float chn_pen_gap, float chn_pen_skip, int is_cdna, int n_seg);

#ifdef __cplusplus
}
#endif

#endif // _PLCHAIN_H_
