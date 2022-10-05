#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "mmpriv.h" // declare functions in this header
#include "debug.h"
#include "plchain.h"
#include <time.h>

const char *filename = "Log/chain_in_output";
const char *anchors_filename = "Log/anchors_results";
FILE *output = NULL;
FILE *anchors_output = NULL;

void debug_compact_anchors(mm128_t *a, int64_t n) {
    if (anchors_output == NULL) {
        if ((anchors_output = fopen(anchors_filename, "w+")) == NULL) {
            fprintf(stderr, "[Debug]: %s can't be opened\n", anchors_filename);
            exit(0);
        }
    }
    fprintf(anchors_output, "[Debug]: \n");
    for (int i=0; i < n; ++i) {
        fprintf(anchors_output, "Anchor %d, x %ld y %ld\n", i, a[i].x, a[i].y);
    }
}

void debug_print(int32_t *f, int32_t *t, int32_t *v, int64_t *p, int64_t n) {
    printf("[Debug]: \n");
    for (int i=0; i < n; ++i) {
        printf("Anchor %d score: %d predecessor: %ld prepre: %d\n", i, f[i], p[i], t[i]);
    }
}

void debug_chain_output(int32_t *f, int32_t *t, int32_t *v, int64_t *p, int64_t n) {
    // print scores, predecessor, prepre to a file
    if (output == NULL) {
        if ((output = fopen(filename, "w+")) == NULL) {
            fprintf(stderr, "[Debug]: %s can't be opened\n", filename);
            exit(0);
        }
    }
    fprintf(output, "[Debug]: \n");
    for (int i=0; i < n; ++i) {
        fprintf(output, "Anchor %d score: %d predecessor: %ld prepre: %d\n", i, f[i], p[i], t[i]);
    }
}

void debug_chain_result(int32_t n_u, uint64_t *u) {
    // print chains' info to a file
    if (output == NULL) {
        if ((output = fopen(filename, "w+")) == NULL) {
            fprintf(stderr, "[Debug]: %s can't be opened\n", filename);
            exit(0);
        }
    }
    fprintf(output, "[Debug]: \n");
    for (int i=0; i < n_u; ++i) {
        fprintf(output, "Chain %d, score: %d, #anchors: %d\n", i, (int32_t)(u[i]>>32), (int32_t)u[i]);
    }
}

void debug_chain_input(mm128_t *a, int64_t n, int max_iter, int max_dist_x, int max_dist_y, int max_skip,\
                        int bw, float chn_pen_gap, float chn_pen_skip, int is_cdna, int n_seg) {
    if (output == NULL) {
        if ((output = fopen(filename, "w+")) == NULL) {
            fprintf(stderr, "[Debug]: %s can't be opened\n", filename);
            exit(0);
        }
    }
    fprintf(output, "max %d %d %d %d\n", max_dist_x, max_dist_y, max_iter, max_skip);
    fprintf(output, "misc %d %f %f %d %d\n", bw, chn_pen_gap, chn_pen_skip, is_cdna, n_seg);
    fprintf(output, "#anchors %ld\n", n);
    for (int64_t i = 0; i < n; ++i) {
        fprintf(output, "%ld %lu %lu\n", i, a[i].x, a[i].y); // anchor idx, x, y
    }
}

void debug_chain_compute_sc(mm128_t *a, int64_t n, int max_dist_x, int max_iter) {

}

void debug_compare_chain_output(mm128_t *a, mm128_t *a_temp, uint64_t *u, int n_regs) {
    int n_a = 0;
    for (int i = 0; i < n_regs; ++i) n_a += (int32_t)u[i]; // recompute compact a size

    for (int i = 0; i < n_a; i++) {
        assert(a[i].x == a_temp[i].x);
        assert(a[i].y == a_temp[i].y);
    }
}



