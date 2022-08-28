#ifndef __PLTHREAD_H__
#define __PLTHREAD_H__ 

#include <assert.h>
#include "minimap.h"
#include "bseq.h"
#include "kseq.h"
#include "hipify.h"
#include "plutils.h"

#ifdef __cplusplus
extern "C" {
#endif


int pltask_init(int num_seqs);
int plchain_append(long i, int64_t n_a);
int plchain_check(long i);





#ifdef __cplusplus
}
#endif

#endif // !__PLTHREAD_H__