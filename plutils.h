#ifndef _UTILS_H_ 
#define _UTILS_H_ 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <assert.h>
#include "minimap.h"
#include "kalloc.h"

typedef struct {
    int max_iter, max_dist_x, max_dist_y, max_skip, bw, is_cdna, n_seg;
    int min_cnt, min_sc;
    float chn_pen_gap, chn_pen_skip;
} Misc;

typedef struct {
    int64_t *x;
    int64_t *y;
    int32_t *range;
    size_t *cut;
    size_t total_n;
    size_t num_cut;
} a_pass;

typedef struct {
    int index; // read index / batch index 
    int size; // batch size
    // host memory ptrs
    // dynamic size
    int64_t *ax;
    int64_t *ay;
    int32_t *f; // score
    uint16_t *p; // predecessor
    // range selection
    size_t *start_idx;
    size_t *read_end_idx;
    size_t *cut_start_idx;
} hostMemPtr;

typedef struct {
    // device memory ptrs
    // dynamic size
    int64_t *d_ax;
    int64_t *d_ay;
    int32_t *d_range;
    int32_t *d_f; // score
    uint16_t *d_p; // predecessor
    size_t *d_cut; // cut
    // range selection
    size_t *d_start_idx;
    size_t *d_read_end_idx;
    size_t *d_cut_start_idx;
    // number
    size_t total_n;
    size_t num_cut;
} deviceMemPtr;

typedef struct {
    mm128_t *a;
    int64_t n;
    Misc misc;
    int32_t *f;
    int64_t *p;
    int32_t *t;
    int32_t *range;
} input_iter;

#include "mmpriv.h"

#endif // _UTILS_H_ 

