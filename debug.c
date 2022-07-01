#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "mmpriv.h" // declare functions in this header

const char *filename = "debug/chain_in_output";
const char *rangename = "debug/range_pred";
FILE *output = NULL;
FILE *range_pred = NULL;

void debug_chain_range(int64_t i, int64_t st) {
    // print scores, predecessor, prepre to a file
    if (range_pred == NULL) {
        if ((range_pred = fopen(rangename, "w+")) == NULL) {
            fprintf(stderr, "[Debug]: %s can't be opened\n", filename);
            exit(0);
        }
    }
    fprintf(range_pred, "%ld,", i - st); // i - st = range 
}

void debug_chain_range_end() {
    // print scores, predecessor, prepre to a file
    if (range_pred == NULL) {
        if ((range_pred = fopen(rangename, "w+")) == NULL) {
            fprintf(stderr, "[Debug]: %s can't be opened\n", filename);
            exit(0);
        }
    }
    fprintf(range_pred, "\n"); // i-st = range 
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

        if (p[i] >= 0 && i - p[i] > 5000){
            fprintf(stderr, "Anchor %d score: %d predecessor: %ld\n",
                    i, f[i], p[i], t[i]);
        }
        fprintf(output, "Anchor %d score: %d predecessor: %ld prepre: %d\n", i, f[i], p[i], t[i]);
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



