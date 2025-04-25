#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*
 * Simplified re-implementation of:
 *   static int64_t mg_chain_bk_end(...)
 *   uint64_t *mg_chain_backtrack(...)
 * from minimap2’s lchain.c
 */

/* Walk backward until score drop > max_drop */
static int64_t mg_chain_bk_end(int32_t max_drop,
                               const int32_t *z_x,    // scores
                               const int64_t *z_y,    // indices
                               const int32_t *f,      // score at each i
                               const int64_t *p,      // predecessor at each i
                               int32_t *t,            // tag array
                               int64_t k,             // index in z arrays
                               int64_t n_z)           // size of z arrays (unused here)
{
    int64_t i      = z_y[k];
    int64_t end_i  = -1, max_i = i;
    int32_t max_s  = 0;

    if (i < 0 || t[i] != 0) return i;
    do {
        t[i] = 2;              /* mark in-progress */
        end_i = i = p[i];      /* step to predecessor */

        int32_t s = (i < 0 ? z_x[k] : z_x[k] - f[i]);
        if (s > max_s) {
            max_s = s; max_i = i;
        } else if (max_s - s > max_drop) {
            break;
        }
    } while (i >= 0 && t[i] == 0);

    /* reset tags along this walk */
    for (i = z_y[k]; i >= 0 && i != end_i; i = p[i])
        t[i] = 0;

    return max_i;
}

/* Backtrack all chains passing min_sc and min_cnt */
uint64_t *mg_chain_backtrack(int64_t n,
                             const int32_t *f,
                             const int64_t *p,
                             int32_t *v,
                             int32_t *t,
                             int32_t min_cnt,
                             int32_t min_sc,
                             int32_t max_drop,
                             int32_t *n_u_,
                             int32_t *n_v_)
{
    /* 1) collect qualified anchors */
    int64_t n_z = 0;
    for (int64_t i = 0; i < n; i++)
        if (f[i] >= min_sc) n_z++;

    if (n_z == 0) {
        *n_u_ = *n_v_ = 0;
        return NULL;
    }

    /* pack into z_x, z_y and sort by score descending (naïve bubble for simplicity) */
    int32_t *z_x = malloc(n_z * sizeof(int32_t));
    int64_t  *z_y = malloc(n_z * sizeof(int64_t));
    for (int64_t i = 0, k = 0; i < n; i++) {
        if (f[i] >= min_sc) {
            z_x[k] = f[i];
            z_y[k++] = i;
        }
    }
    /* sort z by z_x descending */
    for (int64_t i = 0; i < n_z-1; i++)
      for (int64_t j = i+1; j < n_z; j++)
        if (z_x[j] > z_x[i]) {
            int32_t tx = z_x[i]; z_x[i] = z_x[j]; z_x[j] = tx;
            int64_t ty = z_y[i]; z_y[i] = z_y[j]; z_y[j] = ty;
        }

    /* prepare outputs */
    int32_t n_u = 0, n_v = 0;
    /* first pass: count valid chains */
    memset(t, 0, n * sizeof(int32_t));
    for (int64_t k = 0; k < n_z; k++) {
        int64_t y = z_y[k];
        if (t[y] == 0) {
            int32_t prev_n_v = n_v;
            int64_t end_i = mg_chain_bk_end(max_drop, z_x, z_y, f, p, t, k, n_z);
            for (int64_t i = y; i != end_i; i = p[i]) {
                t[i] = 1;
                v[n_v++] = (int32_t)i;
            }
            int32_t sc = (end_i < 0 ? z_x[k] : z_x[k] - f[end_i]);
            int32_t len = n_v - prev_n_v;
            if (sc >= min_sc && len >= min_cnt) n_u++;
            else n_v = prev_n_v;
        }
    }

    /* allocate summary array */
    uint64_t *u = malloc(n_u * sizeof(uint64_t));

    /* second pass: fill u and v */
    memset(t, 0, n * sizeof(int32_t));
    n_v = 0; n_u = 0;
    for (int64_t k = 0; k < n_z; k++) {
        int64_t y = z_y[k];
        if (t[y] == 0) {
            int32_t prev_n_v = n_v;
            int64_t end_i = mg_chain_bk_end(max_drop, z_x, z_y, f, p, t, k, n_z);
            for (int64_t i = y; i != end_i; i = p[i]) {
                t[i] = 1;
                v[n_v++] = (int32_t)i;
            }
            int32_t sc = (end_i < 0 ? z_x[k] : z_x[k] - f[end_i]);
            int32_t len = n_v - prev_n_v;
            if (sc >= min_sc && len >= min_cnt) {
                u[n_u++] = ((uint64_t)sc << 32) | (uint32_t)len;
            } else {
                n_v = prev_n_v;
            }
        }
    }

    free(z_x);
    free(z_y);
    *n_u_ = n_u; *n_v_ = n_v;
    return u;
}

/*--------------- Test harness with verification ----------------*/
int main(void)
{
    /* Example data: 5 anchors */
    int64_t n = 5;
    int32_t  f[] = { 5, 4, 3, 2, 6 };     // scores
    int64_t  p[] = { -1, 0, 1, 2, -1 };   // predecessors

    /* Storage for backtracking */
    int32_t *v = calloc(n, sizeof(int32_t));
    int32_t *t = calloc(n, sizeof(int32_t));
    int32_t  min_cnt = 1, min_sc = 1, max_drop = 10;
    int32_t  n_u, n_v;

    /* Run the backtrack */
    uint64_t *u = mg_chain_backtrack(n, f, p, v, t,
                                     min_cnt, min_sc, max_drop,
                                     &n_u, &n_v);

    /* Expected results for this dataset:
     * - Two chains:
     *     Chain 0: anchor 4 alone  => score=6, length=1, indices=[4]
     *     Chain 1: anchor 0 alone  => score=5, length=1, indices=[0]
     *   (Higher scores first.)
     */
    const int32_t  exp_n_u = 2;
    const int32_t  exp_n_v = 2;
    const uint64_t exp_u[] = {
        ((uint64_t)6 << 32) | 1,   /* 6 in high 32b, length=1 */
        ((uint64_t)5 << 32) | 1
    };
    const int32_t  exp_v[] = { 4, 0 };

    /* Verify chain count and total anchors */
    if (n_u != exp_n_u || n_v != exp_n_v) {
        fprintf(stderr,
                "FAIL: Expected n_u=%d, n_v=%d but got n_u=%d, n_v=%d\n",
                exp_n_u, exp_n_v, n_u, n_v);
        return 1;
    }

    /* Verify each chain summary */
    for (int i = 0; i < n_u; i++) {
        if (u[i] != exp_u[i]) {
            fprintf(stderr,
                    "FAIL: Chain %d summary mismatch: expected 0x%016llx, got 0x%016llx\n",
                    i, (unsigned long long)exp_u[i], (unsigned long long)u[i]);
            return 1;
        }
    }

    /* Verify flattened indices */
    for (int i = 0; i < n_v; i++) {
        if (v[i] != exp_v[i]) {
            fprintf(stderr,
                    "FAIL: Chain index %d mismatch: expected %d, got %d\n",
                    i, exp_v[i], v[i]);
            return 1;
        }
    }

    printf("PASS: backtrack produced the expected chains!\n");
    free(u);
    free(v);
    free(t);
    return 0;
}
