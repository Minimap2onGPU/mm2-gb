#ifndef RMI_SEED_H
#define RMI_SEED_H

#include "../mmpriv.h"
#include "../ksort.h"  // 包含 ksort.h 以确保声明正确

#ifdef __cplusplus
extern "C" {
#endif

// Function to filter out minimizers based on their occurrence
void mm_seed_mz_flt(void *km, mm128_v *mv, int32_t q_occ_max, float q_occ_frac);

// Host function to launch the GPU kernel for collecting seeds
mm_seed_t* mm_seed_collect_all(void *km, const mm_idx_t *mi, const mm128_v *mv, int32_t *n_m_);

// Function to select seeds based on their frequency
void mm_seed_select(int32_t n, mm_seed_t *a, int len, int max_occ, int max_max_occ, int dist);

// Function to collect matches from minimizers
mm_seed_t *mm_collect_matches(void *km, int *_n_m, int qlen, int max_occ, int max_max_occ, int dist, const mm_idx_t *mi, const mm128_v *mv, int64_t *n_a, int *rep_len, int *n_mini_pos, uint64_t **mini_pos);

// Declare ks_heapmake_uint64_t and ks_heapdown_uint64_t
void ks_heapmake_uint64_t(size_t n, uint64_t* a);
void ks_heapdown_uint64_t(size_t i, size_t n, uint64_t* a);

#ifdef __cplusplus
}
#endif

#endif // RMI_SEED_H
