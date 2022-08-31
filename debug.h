#ifndef __DEBUG_H__
#define __DEBUG_H__
#include "plutils.h"

#ifdef __cplusplus
extern "C" {
#endif

extern char input_filename[];
extern char range_infile[];
extern char binary_file[];
extern char binary_range[];

// chain loop declaration
int debug_chain_loop(int forward_chaining, int use_gpu);
int debug_stream_chain_loop();
int debug_dynamic_stream_chain_loop();

// check loop declaration
int is_debug_range();
void debug_print_successor_range(int32_t* range, int64_t n);
void debug_print_score(int32_t* score, int64_t n);
int debug_check_score(const int64_t* p, const int32_t* f, const int64_t* p_gold, const int32_t* f_gold,
                     int64_t n);
int debug_check_score_auto_trans(const uint16_t *p, const int32_t *f, const int64_t *p_gold,
                      const int32_t *f_gold, int64_t n);
int debug_check_range(const int32_t* range, const int32_t* range_gold, int64_t n);

// file read loop declaration
int debug_read_binary(input_iter *input_struct);
void debug_read_loop();
void debug_print_binary(input_iter *input_struct);
int debug_binary_loop();
void debug_range_binary_print(input_iter *input_struct);
int debug_range_binary_read(input_iter *input_struct);

// debug_data print
void debug_print(int32_t *f, int32_t *t, int32_t *v, int64_t *p, int64_t n);
void debug_chain_output(int32_t *f, int32_t *t, int32_t *v, int64_t *p, int64_t n);
void debug_chain_result(int32_t n_u, uint64_t *u);
void debug_chain_input(mm128_t *a, int64_t n, int max_iter, int max_dist_x, int max_dist_y, int max_skip, int bw, float chn_pen_gap, float chn_pen_skip, int is_cdna, int n_seg);


#ifdef __cplusplus
}
#endif

#endif// __DEBUG_H__
