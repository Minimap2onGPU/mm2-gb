#ifndef _PLCHAIN_H_
#define _PLCHAIN_H_

#include <assert.h>
#include "minimap.h"
#include "bseq.h"
#include "kseq.h"

#ifdef __cplusplus
extern "C" {
#endif

// debug functions
void debug_chain_range(int64_t i, int64_t st);
void debug_chain_range_end();
void debug_chain_output(int32_t *f, int32_t *t, int32_t *v, int64_t *p, int64_t n);
void debug_chain_input(mm128_t *a, int64_t n, int max_iter, int max_dist_x, int max_dist_y, int max_skip, int bw, float chn_pen_gap, float chn_pen_skip, int is_cdna, int n_seg);

#ifdef __cplusplus
}
#endif

#endif // _PLCHAIN_H_
