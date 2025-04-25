#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <hip/hip_runtime.h>

static const int MAX_CHAIN_LEN = 1024;

// Local-only backtrack to find chain end index
__device__ int64_t mg_chain_bk_end_gpu(int32_t max_drop,
                                       const int32_t *z_x,
                                       const int64_t *z_y,
                                       const int32_t *f,
                                       const int64_t *p,
                                       int64_t k)
{
    int64_t start = z_y[k];
    int64_t end_i = -1, max_i = start;
    int32_t max_s = 0;
    int64_t i = start;

    // Walk backward without touching t[]
    while (i >= 0) {
        int64_t prev = p[i];
        int32_t s = (prev < 0 ? z_x[k] : z_x[k] - f[prev]);
        if (s > max_s) {
            max_s = s;
            max_i = prev;
        } else if (max_s - s > max_drop) {
            break;
        }
        end_i = i = prev;
    }
    return max_i;
}

// Parallel HIP kernel
__global__ void hip_chain_backtrack(int64_t   n_z,
                                    const int32_t * __restrict__ z_x,
                                    const int64_t  * __restrict__ z_y,
                                    const int32_t  * __restrict__ f,
                                    const int64_t  * __restrict__ p,
                                    int32_t        * __restrict__ t,
                                    int32_t min_cnt,
                                    int32_t min_sc,
                                    int32_t max_drop,
                                    int32_t * __restrict__ u_counter,
                                    int32_t * __restrict__ v_counter,
                                    uint64_t * __restrict__ u_out,
                                    int32_t  * __restrict__ v_out)
{
    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_z) return;

    // 1) Try to claim the start anchor
    int64_t y = z_y[k];
    if (y < 0) return;
    if (atomicCAS(&t[y], 0, 2) != 0) return;

    // 2) Find chain end index
    int64_t end_i = mg_chain_bk_end_gpu(max_drop, z_x, z_y, f, p, k);

    // 3) Forward walk: claim each anchor and record it locally
    int32_t local_v[MAX_CHAIN_LEN];
    int    len = 0;
    int64_t curr = y;
    while (curr >= 0 && len < MAX_CHAIN_LEN) {
        // Must have been marked 2 above for the start; for others, go 0→1
        if (curr == y) {
            // Already in state 2
            local_v[len++] = (int32_t)curr;
        } else {
            if (atomicCAS(&t[curr], 0, 1) != 0) {
                // Conflict: undo all previous tags
                for (int i = 0; i < len; ++i) {
                    atomicExch(&t[ local_v[i] ], 0);
                }
                return;
            }
            local_v[len++] = (int32_t)curr;
        }
        if (curr == end_i) break;
        curr = p[curr];
    }

    // 4) Compute chain score
    int32_t sc = (end_i < 0 ? z_x[k] : z_x[k] - f[end_i]);

    // 5) Validate chain length and score
    if (sc < min_sc || len < min_cnt) {
        // Not enough score/count → undo tags
        for (int i = 0; i < len; ++i) {
            atomicExch(&t[ local_v[i] ], 0);
        }
        return;
    }

    // 6) Reserve slots in output arrays
    int32_t u_idx = atomicAdd(u_counter, 1);
    int32_t v_idx = atomicAdd(v_counter, len);

    // 7) Write summary and indices
    u_out[u_idx] = ((uint64_t)sc << 32) | (uint32_t)len;
    for (int i = 0; i < len; ++i) {
        v_out[v_idx + i] = local_v[i];
    }
}


int main() {
    // 1) Host test data (same as CPU harness)
    const int64_t n  = 5;
    int32_t  h_f[]   = { 5, 4, 3, 2, 6 };
    int64_t  h_p[]   = { -1, 0, 1, 2, -1 };
    const int32_t min_cnt  = 1;
    const int32_t min_sc   = 1;
    const int32_t max_drop = 10;

    // 2) Prepare sorted indices z_x, z_y on host
    //    (Here we simply hard-code the sorted order: index 4 (score 6), then 0 (5), 1 (4), 2 (3), 3 (2))
    const int64_t n_z = 5;
    int32_t  h_zx[] = { 6, 5, 4, 3, 2 };
    int64_t  h_zy[] = { 4, 0, 1, 2, 3 };

    // 3) Allocate and initialize device buffers
    int32_t *d_f, *d_zx, *d_t;
    int64_t *d_p, *d_zy;
    uint64_t *d_uout;
    int32_t  *d_vout, *d_ucnt, *d_vcnt;

    hipMalloc(&d_f,    n  * sizeof(int32_t));      // :contentReference[oaicite:4]{index=4}
    hipMalloc(&d_p,    n  * sizeof(int64_t));      //
    hipMalloc(&d_zx,   n_z * sizeof(int32_t));     //
    hipMalloc(&d_zy,   n_z * sizeof(int64_t));     //
    hipMalloc(&d_t,    n  * sizeof(int32_t));      //
    hipMalloc(&d_ucnt, sizeof(int32_t));           //
    hipMalloc(&d_vcnt, sizeof(int32_t));           //
    hipMalloc(&d_uout, n  * sizeof(uint64_t));     //
    hipMalloc(&d_vout, n  * sizeof(int32_t));      //

    // zero out tags and counters
    hipMemset(d_t,    0, n  * sizeof(int32_t));    // :contentReference[oaicite:5]{index=5}
    hipMemset(d_ucnt, 0, sizeof(int32_t));         //
    hipMemset(d_vcnt, 0, sizeof(int32_t));         //

    // 4) Copy inputs to device
    hipMemcpy(d_f,  h_f,  n  * sizeof(int32_t), hipMemcpyHostToDevice); // :contentReference[oaicite:6]{index=6}
    hipMemcpy(d_p,  h_p,  n  * sizeof(int64_t), hipMemcpyHostToDevice); //
    hipMemcpy(d_zx, h_zx, n_z * sizeof(int32_t), hipMemcpyHostToDevice);//
    hipMemcpy(d_zy, h_zy, n_z * sizeof(int64_t), hipMemcpyHostToDevice);//

    // 5) Launch kernel: one thread per sorted anchor
    const int threads = 256;
    int blocks = (n_z + threads - 1) / threads;
    hipLaunchKernelGGL(hip_chain_backtrack,
                       dim3(blocks), dim3(threads),
                       0, 0,
                       n_z, d_zx, d_zy, d_f, d_p, d_t,
                       min_cnt, min_sc, max_drop,
                       d_ucnt, d_vcnt, d_uout, d_vout);  // :contentReference[oaicite:7]{index=7}

    hipDeviceSynchronize();                        // :contentReference[oaicite:8]{index=8}

    // 6) Copy back results
    int32_t h_ucnt = 0, h_vcnt = 0;
    hipMemcpy(&h_ucnt, d_ucnt, sizeof(int32_t), hipMemcpyDeviceToHost);      // :contentReference[oaicite:9]{index=9}
    hipMemcpy(&h_vcnt, d_vcnt, sizeof(int32_t), hipMemcpyDeviceToHost);      //
    uint64_t *h_uout = (uint64_t*)malloc(h_ucnt * sizeof(uint64_t));
    int32_t  *h_vout = (int32_t*) malloc(h_vcnt * sizeof(int32_t));
    hipMemcpy(h_uout, d_uout, h_ucnt * sizeof(uint64_t), hipMemcpyDeviceToHost);  // :contentReference[oaicite:10]{index=10}
    hipMemcpy(h_vout, d_vout, h_vcnt * sizeof(int32_t), hipMemcpyDeviceToHost);   //

    // 7) Verify same expected output as CPU test
    const int32_t exp_ucnt = 2, exp_vcnt = 2;
    const uint64_t exp_u[] = { ((uint64_t)6 << 32) | 1,
                               ((uint64_t)5 << 32) | 1 };
    const int32_t exp_v[] = { 4, 0 };

    bool ok = true;
    if (h_ucnt != exp_ucnt || h_vcnt != exp_vcnt) ok = false;
    for (int i = 0; i < exp_ucnt && ok; i++) {
        if (h_uout[i] != exp_u[i]) ok = false;
    }
    for (int i = 0; i < exp_vcnt && ok; i++) {
        if (h_vout[i] != exp_v[i]) ok = false;
    }

    printf(ok ? "PASS: HIP backtrack matches CPU\n"
              : "FAIL: HIP backtrack mismatch\n");

    // 8) Cleanup
    hipFree(d_f); hipFree(d_p); hipFree(d_zx); hipFree(d_zy);
    hipFree(d_t); hipFree(d_ucnt); hipFree(d_vcnt);
    hipFree(d_uout); hipFree(d_vout);                   // :contentReference[oaicite:11]{index=11}
    free(h_uout); free(h_vout);

    return ok ? 0 : 1;
}