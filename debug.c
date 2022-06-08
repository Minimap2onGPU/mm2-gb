#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "mmpriv.h" // declare functions in this header

const char *filename = "debug/output";
FILE *output = NULL;

void debug_print(int32_t *f, int32_t *t, int32_t *v, int64_t *p, int64_t n) {
    printf("[Debug]: \n");
    for (int i=0; i < n; ++i) {
        printf("Anchor %d, score: %d, predecessor: %ld, prepre: %d\n", i, f[i], p[i], t[i]);
    }
}

void debug_fprint(int32_t *f, int32_t *t, int32_t *v, int64_t *p, int64_t n) {
    // print scores, predecessor, prepre to a file
    if (output == NULL) {
        if ((output = fopen(filename, "w+")) == NULL) {
            printf ("[Debug]: %s can't be opened\n", filename);
            exit(0);
        }
    }
    fprintf(output, "[Debug]: \n");
    for (int i=0; i < n; ++i) {
        fprintf(output, "Anchor %d, score: %d, predecessor: %ld, prepre: %d\n", i, f[i], p[i], t[i]);
    }
}

void debug_chain_fprint(int32_t n_u, uint64_t *u) {
    // print chains' info to a file
    if (output == NULL) {
        if ((output = fopen(filename, "w+")) == NULL) {
            printf ("[Debug]: %s can't be opened\n", filename);
            exit(0);
        }
    }
    fprintf(output, "[Debug]: \n");
    for (int i=0; i < n_u; ++i) {
        fprintf(output, "Chain %d, score: %d, #anchors: %d\n", i, u[i]>>32, (int32_t)u[i]);
    }
}




