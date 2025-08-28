
#ifndef __WB_HIP_H__
#define __WB_HIP_H__

#ifdef WB_USE_HIP
#ifdef __PGI
#define __GNUC__ 4
#endif /* __PGI */
#include <hip.h>
#include <hip/hip_runtime.h>

typedef struct st_wbHIPMemory_t {
  void *mem;
  size_t sz;
} wbHIPMemory_t;

#define _hipMemoryListSize 1024

extern size_t _hipMallocSize;
extern wbHIPMemory_t _hipMemoryList[];
extern int _hipMemoryListIdx;

char *wbRandom_list(size_t sz);

static inline hipError_t wbHIPMalloc(void **devPtr, size_t sz) {
  int idx = _hipMemoryListIdx;

  hipError_t err = hipMalloc(devPtr, sz);

  if (idx == 0) {
    srand(time(NULL));
    memset(_hipMemoryList, 0,
           sizeof(wbHIPMemory_t) * _hipMemoryListSize);
  }

  if (err == hipSuccess) {
#if 0
    char * rands = wbRandom_list(sz);
    // can use curand here, but do not want to invoke a kernel
    err = hipMemcpy(*devPtr, rands, sz, hipMemcpyHostToDevice);
    wbFree(rands);
#else
    err = hipMemset(*devPtr, 0, sz);
#endif
  }

  _hipMallocSize += sz;
  _hipMemoryList[idx].mem = *devPtr;
  _hipMemoryList[idx].sz  = sz;
  _hipMemoryListIdx++;
  return err;
}

static inline hipError_t wbHIPFree(void *mem) {
  int idx = _hipMemoryListIdx;
  if (idx == 0) {
    memset(_hipMemoryList, 0,
           sizeof(wbHIPMemory_t) * _hipMemoryListSize);
  }
  for (int ii = 0; ii < idx; ii++) {
    if (_hipMemoryList[ii].mem != nullptr &&
        _hipMemoryList[ii].mem == mem) {
      hipError_t err = hipFree(mem);
      _hipMallocSize -= _hipMemoryList[ii].sz;
      _hipMemoryList[ii].mem = nullptr;
      return err;
    }
  }
  return hipErrorMemoryAllocation;
}

#define hipMalloc(elem, err) wbHIPMalloc((void **)elem, err)
#define hipFree wbHIPFree

#endif /* WB_USE_HIP */

#endif /* __WB_HIP_H__ */
