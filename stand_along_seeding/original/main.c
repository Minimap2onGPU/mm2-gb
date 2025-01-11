#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include "kthread.h"
#include "kvec.h"
#include "kalloc.h"
#include "sdust.h"
#include "mmpriv.h"
#include "bseq.h"
#include "khash.h"

static int mm_dust_minier(void *km, int n, mm128_t *a, int l_seq, const char *seq, int sdust_thres)
{
	int n_dreg, j, k, u = 0;
	const uint64_t *dreg;
	sdust_buf_t *sdb;
	if (sdust_thres <= 0) return n;
	sdb = sdust_buf_init(km);
	dreg = sdust_core((const uint8_t*)seq, l_seq, sdust_thres, 64, &n_dreg, sdb);
	for (j = k = 0; j < n; ++j) { // squeeze out minimizers that significantly overlap with LCRs
		int32_t qpos = (uint32_t)a[j].y>>1, span = a[j].x&0xff;
		int32_t s = qpos - (span - 1), e = s + span;
		while (u < n_dreg && (int32_t)dreg[u] <= s) ++u;
		if (u < n_dreg && (int32_t)(dreg[u]>>32) < e) {
			int v, l = 0;
			for (v = u; v < n_dreg && (int32_t)(dreg[v]>>32) < e; ++v) { // iterate over LCRs overlapping this minimizer
				int ss = s > (int32_t)(dreg[v]>>32)? s : dreg[v]>>32;
				int ee = e < (int32_t)dreg[v]? e : (uint32_t)dreg[v];
				l += ee - ss;
			}
			if (l <= span>>1) a[k++] = a[j]; // keep the minimizer if less than half of it falls in masked region
		} else a[k++] = a[j];
	}
	sdust_buf_destroy(sdb);
	return k; // the new size
}

static void collect_minimizers(void *km, const mm_mapopt_t *opt, const mm_idx_t *mi, int n_segs, const int *qlens, const char **seqs, mm128_v *mv)
{
	int i, n, sum = 0;
	mv->n = 0;
	for (i = n = 0; i < n_segs; ++i) {
		size_t j;
		mm_sketch(km, seqs[i], qlens[i], mi->w, mi->k, i, mi->flag&MM_I_HPC, mv);
		for (j = n; j < mv->n; ++j)
			mv->a[j].y += sum << 1;
		if (opt->sdust_thres > 0) // mask low-complexity minimizers
			mv->n = n + mm_dust_minier(km, mv->n - n, mv->a + n, qlens[i], seqs[i], opt->sdust_thres);
		sum += qlens[i], n = mv->n;
	}
}

static mm128_t *collect_seed_hits_heap(void *km, const mm_mapopt_t *opt, int max_occ, const mm_idx_t *mi, const char *qname, const mm128_v *mv, int qlen, int64_t *n_a, int *rep_len,
								  int *n_mini_pos, uint64_t **mini_pos)
{
	int i, n_m, heap_size = 0;
	int64_t j, n_for = 0, n_rev = 0;
	mm_seed_t *m;
	mm128_t *a, *heap;

	m = mm_collect_matches(km, &n_m, qlen, max_occ, opt->max_max_occ, opt->occ_dist, mi, mv, n_a, rep_len, n_mini_pos, mini_pos);

	heap = (mm128_t*)kmalloc(km, n_m * sizeof(mm128_t));
	a = (mm128_t*)kmalloc(km, *n_a * sizeof(mm128_t));

	for (i = 0, heap_size = 0; i < n_m; ++i) {
		if (m[i].n > 0) {
			heap[heap_size].x = m[i].cr[0];
			heap[heap_size].y = (uint64_t)i<<32;
			++heap_size;
		}
	}
	ks_heapmake_heap(heap_size, heap);
	while (heap_size > 0) {
		mm_seed_t *q = &m[heap->y>>32];
		mm128_t *p;
		uint64_t r = heap->x;
		int32_t is_self, rpos = (uint32_t)r >> 1;
		if (!skip_seed(opt->flag, r, q, qname, qlen, mi, &is_self)) {
			if ((r&1) == (q->q_pos&1)) { // forward strand
				p = &a[n_for++];
				p->x = (r&0xffffffff00000000ULL) | rpos;
				p->y = (uint64_t)q->q_span << 32 | q->q_pos >> 1;
			} else { // reverse strand
				p = &a[(*n_a) - (++n_rev)];
				p->x = 1ULL<<63 | (r&0xffffffff00000000ULL) | rpos;
				p->y = (uint64_t)q->q_span << 32 | (qlen - ((q->q_pos>>1) + 1 - q->q_span) - 1);
			}
			p->y |= (uint64_t)q->seg_id << MM_SEED_SEG_SHIFT;
			if (q->is_tandem) p->y |= MM_SEED_TANDEM;
			if (is_self) p->y |= MM_SEED_SELF;
		}
		// update the heap
		if ((uint32_t)heap->y < q->n - 1) {
			++heap[0].y;
			heap[0].x = m[heap[0].y>>32].cr[(uint32_t)heap[0].y];
		} else {
			heap[0] = heap[heap_size - 1];
			--heap_size;
		}
		ks_heapdown_heap(0, heap_size, heap);
	}
	kfree(km, m);
	kfree(km, heap);

	// reverse anchors on the reverse strand, as they are in the descending order
	for (j = 0; j < n_rev>>1; ++j) {
		mm128_t t = a[(*n_a) - 1 - j];
		a[(*n_a) - 1 - j] = a[(*n_a) - (n_rev - j)];
		a[(*n_a) - (n_rev - j)] = t;
	}
	if (*n_a > n_for + n_rev) {
		memmove(a + n_for, a + (*n_a) - n_rev, n_rev * sizeof(mm128_t));
		*n_a = n_for + n_rev;
	}
	return a;
}

static mm128_t *collect_seed_hits(void *km, const mm_mapopt_t *opt, int max_occ, const mm_idx_t *mi, const char *qname, const mm128_v *mv, int qlen, int64_t *n_a, int *rep_len,
								  int *n_mini_pos, uint64_t **mini_pos)
{
	int i, n_m;
	mm_seed_t *m;
	mm128_t *a;
	m = mm_collect_matches(km, &n_m, qlen, max_occ, opt->max_max_occ, opt->occ_dist, mi, mv, n_a, rep_len, n_mini_pos, mini_pos);
	a = (mm128_t*)kmalloc(km, *n_a * sizeof(mm128_t));
	for (i = 0, *n_a = 0; i < n_m; ++i) {
		mm_seed_t *q = &m[i];
		const uint64_t *r = q->cr;
		uint32_t k;
		for (k = 0; k < q->n; ++k) {
			int32_t is_self, rpos = (uint32_t)r[k] >> 1;
			mm128_t *p;
			if (skip_seed(opt->flag, r[k], q, qname, qlen, mi, &is_self)) continue;
			p = &a[(*n_a)++];
			if ((r[k]&1) == (q->q_pos&1)) { // forward strand
				p->x = (r[k]&0xffffffff00000000ULL) | rpos;
				p->y = (uint64_t)q->q_span << 32 | q->q_pos >> 1;
			} else if (!(opt->flag & MM_F_QSTRAND)) { // reverse strand and not in the query-strand mode
				p->x = 1ULL<<63 | (r[k]&0xffffffff00000000ULL) | rpos;
				p->y = (uint64_t)q->q_span << 32 | (qlen - ((q->q_pos>>1) + 1 - q->q_span) - 1);
			} else { // reverse strand; query-strand
				int32_t len = mi->seq[r[k]>>32].len;
				p->x = 1ULL<<63 | (r[k]&0xffffffff00000000ULL) | (len - (rpos + 1 - q->q_span) - 1); // coordinate only accurate for non-HPC seeds
				p->y = (uint64_t)q->q_span << 32 | q->q_pos >> 1;
			}
			p->y |= (uint64_t)q->seg_id << MM_SEED_SEG_SHIFT;
			if (q->is_tandem) p->y |= MM_SEED_TANDEM;
			if (is_self) p->y |= MM_SEED_SELF;
		}
	}
	kfree(km, m);
	radix_sort_128x(a, a + (*n_a));
	return a;
}


void mm_map_seed(const mm_idx_t *mi, const mm_mapopt_t *opt,
                 chain_read_t *read_, mm_tbuf_t *b, void *km) {
    int n_segs = read_->n_seg;
    const int *qlens = read_->qlens;
	const char **seqs = read_->qseqs;
    const char *qname = read_->seq.name;
    int *rep_len = &read_->rep_len;
    int *qlen_sum = &read_->seq.qlen_sum;
    int *n_mini_pos = &read_->n_mini_pos;
    uint64_t **mini_pos = &read_->mini_pos;
    int64_t *n_a = &read_->n;
    mm128_t **a = &read_->a;

    int i;
    mm128_v mv = {0,0,0};
	double *timers = b->timers;
	double t1 = realtime();

    for (i = 0, *qlen_sum = 0; i < n_segs; ++i) *qlen_sum += qlens[i];

    if (*qlen_sum == 0 || n_segs <= 0 || n_segs > MM_MAX_SEG) return;
	if (opt->max_qlen > 0 && *qlen_sum > opt->max_qlen) return;

	collect_minimizers(km, opt, mi, n_segs, qlens, seqs, &mv);
	if (opt->q_occ_frac > 0.0f) mm_seed_mz_flt(km, &mv, opt->mid_occ, opt->q_occ_frac);
	if (opt->flag & MM_F_HEAP_SORT) *a = collect_seed_hits_heap(km, opt, opt->mid_occ, mi, qname, &mv, *qlen_sum, n_a, rep_len, n_mini_pos, mini_pos);
	else *a = collect_seed_hits(km, opt, opt->mid_occ, mi, qname, &mv, *qlen_sum, n_a, rep_len, n_mini_pos, mini_pos);

	if (mm_dbg_flag & MM_DBG_PRINT_SEED) {
		fprintf(stderr, "RS\t%d\n", *rep_len);
		for (i = 0; i < *n_a; ++i)
			fprintf(stderr, "SD\t%s\t%d\t%c\t%d\t%d\t%d\n", mi->seq[(*a)[i].x<<1>>33].name, (int32_t)(*a)[i].x, "+-"[(*a)[i].x>>63], (int32_t)(*a)[i].y, (int32_t)((*a)[i].y>>32&0xff),
					i == 0? 0 : ((int32_t)(*a)[i].y - (int32_t)(*a)[i-1].y) - ((int32_t)(*a)[i].x - (int32_t)(*a)[i-1].x));
	}
	kfree(km, mv.a);
	timers[MM_TIME_SEED] += realtime() - t1;
}
