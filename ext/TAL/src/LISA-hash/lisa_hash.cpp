#include "lisa_hash.h"
#include <string>

// Define the instance of the template class
lisa_hash<uint64_t, uint64_t>* lh = nullptr;

extern "C" {

void create_lisa_hash(char *inputFile, char* rmi_prefix) {
    lh = new lisa_hash<uint64_t, uint64_t>((string)inputFile, rmi_prefix);
}

// Implementation of the wrapper function
void mm_idx_get_batched_c(uint64_t* minimizers, uint64_t num_minimizers, int64_t* pos, uint64_t** p_ptrs, int* num_hits) {
    if (lh) {
        lh->mm_idx_get_batched(minimizers, num_minimizers, pos, p_ptrs, num_hits);
    } else {
        // Handle the case where lh is not initialized
        fprintf(stderr, "Error: lisa_hash instance (lh) is not initialized.\n");
    }
}

void delete_lisa_hash() {
    delete lh;
}

}
