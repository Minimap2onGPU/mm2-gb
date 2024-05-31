#include <fstream>
#include <wb.h>
#include <hip/hip_runtime.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                               \
    if (err != hipSuccess) {                                             \
      wbLog(ERROR, "HIP error: ", hipGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define THREAD_SIZE 32
#define L0_SIZE 2
#define L1_SIZE 82969

__constant__ double l0_parameters[L0_SIZE];
__constant__ double l1_parameters[L1_SIZE * 3];

enum query_state {
  GUESS_RMI_ROOT,
  GUESS_RMI_LEAF,
  LAST_MILE
};

typedef struct {
  int64_t qid;
  query_state state;
  int64_t key;
  int64_t modelIndex;
  int64_t first;
  int64_t m;
} BatchMetadata;

double *L0_PARAMETERS;
double *L1_PARAMETERS;
int64_t n = 0;
uint64_t *sortedArray;

uint64_t keys_size;
uint64_t p_size;
uint64_t* p;
uint64_t* p_bin;
uint64_t* values_enc;
uint64_t* values_enc_bin;

bool loadRMI() {
  string filename = "./MT-human.fa_map-ont_minimizers_key_value_sorted_keys.rmi_PARAMETERS";
  std::ifstream infile(filename, std::ios::in | std::ios::binary);
  if (!infile.good()) {
    printf("%s file not found\n", filename.c_str());
    exit(0);
  }

  L0_PARAMETERS = (double*) malloc(L0_SIZE * sizeof(double));
  if (L0_PARAMETERS == NULL) {
    return false;
  }

  int64_t l1_size = 0;
  
  infile.read((char *) &L0_PARAMETERS[0], sizeof(double));
  infile.read((char *) &L0_PARAMETERS[1], sizeof(double));
  infile.read((char *) &l1_size, sizeof(int64_t));
  
  printf("L0_PARAMETER0 = %E, L0_PARAMETER1 = %E, L1_SIZE = %d\n", 
            L0_PARAMETERS[0], L0_PARAMETERS[1], L1_SIZE);
  
  L1_PARAMETERS = (double*) malloc(L1_SIZE * 3 * sizeof(double));
  if (L1_PARAMETERS == NULL) {
    return false;
  }
  
  infile.read((char*)L1_PARAMETERS, L1_SIZE * 3 * sizeof(double));
  
  if (!infile.good()) {
    return false;
  }

  infile.close();

  return true;
}

bool loadSortedArray() {
  string filename = "./MT-human.fa_map-ont_minimizers_key_value_sorted_keys.uint64";
  std::ifstream infile(filename, std::ios::in | std::ios::binary);
  if (!infile.good()) {
    printf("%s file not found\n", filename.c_str());
    exit(0);
  }
  
  infile.read((char *) &n, sizeof(uint64_t));
  fprintf(stderr, "n = %ld\n", n);

  sortedArray =(uint64_t*) malloc(n * sizeof(uint64_t));
  
  infile.read((char*)sortedArray, n * sizeof(uint64_t));

  if (!infile.good()) {
    return false;
  }

  infile.close();
    
  return true;
}

void loadBin(){
  std::ifstream f_size("./MT-human.fa_map-ont_minimizers_key_value_sorted_size");
	f_size >> keys_size;
	f_size >> p_size;

  values_enc = (uint64_t*) malloc(keys_size * sizeof(uint64_t));
  p = (uint64_t*) malloc(p_size * sizeof(uint64_t));

  string f1_name = "./MT-human.fa_map-ont_minimizers_key_value_sorted_pos_bin";
  string f2_name = "./MT-human.fa_map-ont_minimizers_key_value_sorted_val_bin";
  std::ifstream instream_f1(f1_name, std::ifstream::binary);
  std::ifstream instream_f2(f2_name, std::ifstream::binary);
  instream_f1.seekg(0);
  instream_f1.read((char*) &values_enc[0], keys_size*sizeof(uint64_t));
  instream_f1.close();

  instream_f2.seekg(0);
  instream_f2.read((char*) &p[0], p_size*sizeof(uint64_t));
  instream_f2.close();	
}

void loadMinimizer(uint64_t* minimizer, int array_size) {
  std::ifstream file("./MT-human.fa_map-ont_minimizers", std::ios::binary);
  file.read(reinterpret_cast<char*>(minimizer), array_size * sizeof(uint64_t));
  file.close();
}

void loadResult(int64_t* lisa_pos, int* num_hits, int array_size) {
  std::ifstream file1("./MT-human.fa_map-ont_lisa_pos", std::ios::binary);
  file1.read(reinterpret_cast<char*>(lisa_pos), array_size * sizeof(uint64_t));
  file1.close();
  
  std::ifstream file2("./MT-human.fa_map-ont_minimizers_num_hits", std::ios::binary);
  file2.read(reinterpret_cast<char*>(num_hits), array_size * sizeof(uint64_t));
  file2.close();
}


__device__ int64_t clamp(double inp, double bound) {
  if (inp < 0.0) return 0;
  return (inp > bound ? bound : (size_t)inp);
}

__device__ int64_t getGuessRoot(uint64_t key) {
  int64_t modelIndex;
  double fpred = fma(l0_parameters[1], key, l0_parameters[0]);
  modelIndex = clamp(fpred, L1_SIZE - 1.0);
  return modelIndex;
}

__device__ int64_t getGuessLeaf(uint64_t key, int64_t modelIndex, int64_t *err, int64_t n) {
  double fpred = fma(l1_parameters[modelIndex * 3 + 1], key, l1_parameters[modelIndex * 3]);
  
  *err = *((uint64_t*) (l1_parameters + (modelIndex * 3 + 2)));
  
  int64_t guess = clamp(fpred, n - 1.0);
  return guess;
}

__device__ void lastMileSearch(uint64_t* sorted_array, uint64_t key, int64_t &first, int64_t &m) {
  int64_t half = m >> 1;
  int64_t middle = first + half;
  int64_t cond = (key >= sorted_array[middle]);
  first = middle * cond + first * (1 - cond);
  m = (m - half) * cond + half * (1 - cond);
}

__device__ void rmiLookup(uint64_t *minimizers, int64_t *pos_array, uint64_t* sorted_array, int len, int i, int64_t n) {
  if (i < len) {
    BatchMetadata bm;
    bm.qid = i;
    bm.state = GUESS_RMI_ROOT;
    bm.key = minimizers[i];

    // GUESS_RMI_ROOT
    int64_t pos;
    bm.modelIndex = getGuessRoot(bm.key);
    bm.state = GUESS_RMI_LEAF;
    
    // GUESS_RMI_LEAF
    int64_t err;
    int64_t guess = getGuessLeaf(bm.key, bm.modelIndex, &err, n);
    bm.first = guess - err;
    if(bm.first < 0) bm.first = 0;
    int64_t last = guess + err + 1;
    if(last > n) last = n;
    bm.m = last - bm.first;
    bm.state = LAST_MILE;
    int64_t middle = bm.m >> 1;

    // LAST_MILE
    while (bm.m > 1) {
      lastMileSearch(sorted_array, bm.key, bm.first, bm.m);
    }
    if (bm.m == 1) {
      pos = bm.first;
      if(sorted_array[pos] != bm.key)
    	  pos = -1;
      pos_array[i] = pos;
    }
  }
}

__global__ void mmIdxGet(uint64_t *minimizers, int64_t *pos, int *num_hits, uint64_t* values_enc, uint64_t* sorted_array, int len, int64_t n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  
  rmiLookup(minimizers, pos, sorted_array, len, i, n);

  if (i < len) {
    int p_i = pos[i];

    if (p_i < 0 || p_i > len) {
      num_hits[i] = 0;
    } else {
      num_hits[i] = (uint32_t) values_enc[p_i];
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  uint64_t *hostMinimizer;
  int64_t *hostLisaPos;
  int *hostNumHits;
  int64_t *hostOutputLisaPos;
  int *hostOutputNumHits;

  uint64_t *deviceMinimizer;
  int64_t *deviceOutputLisaPos;
  int *deviceOutputNumHits;
  uint64_t *deviceValuesEnc;
  uint64_t *deviceSortedArray;

  args = wbArg_read(argc, argv);

  // Import data
  hostMinimizer = (uint64_t *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  free(hostMinimizer);

  hostMinimizer = (uint64_t *)malloc(inputLength * sizeof(uint64_t));
  hostLisaPos = (int64_t *)malloc(inputLength * sizeof(int64_t));
  hostNumHits = (int *)malloc(inputLength * sizeof(int));
  hostOutputLisaPos = (int64_t *)malloc(inputLength * sizeof(int64_t));
  hostOutputNumHits = (int *)malloc(inputLength * sizeof(int));

  loadRMI();
  loadSortedArray();
  loadBin();
  loadMinimizer(hostMinimizer, inputLength);
  loadResult(hostLisaPos, hostNumHits, inputLength);

  wbLog(TRACE, "The input size is ", inputLength);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  hipMalloc((void **) &deviceMinimizer, inputLength * sizeof(uint64_t));
  hipMalloc((void **) &deviceOutputLisaPos, inputLength * sizeof(int64_t));
  hipMalloc((void **) &deviceOutputNumHits, inputLength * sizeof(int));
  hipMalloc((void **) &deviceValuesEnc, inputLength * sizeof(uint64_t));
  hipMalloc((void **) &deviceSortedArray, n * sizeof(uint64_t));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  hipMemcpy(deviceMinimizer, hostMinimizer, inputLength * sizeof(uint64_t), hipMemcpyHostToDevice);
  hipMemcpy(deviceValuesEnc, values_enc, inputLength * sizeof(uint64_t), hipMemcpyHostToDevice);
  hipMemcpy(deviceSortedArray, sortedArray, n * sizeof(uint64_t), hipMemcpyHostToDevice);
  hipMemcpyToSymbol(l0_parameters, L0_PARAMETERS, L0_SIZE * sizeof(double));
  hipMemcpyToSymbol(l1_parameters, L1_PARAMETERS, L1_SIZE * 3 * sizeof(double));

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  dim3 dimGrid(ceil(inputLength * 1.0 / THREAD_SIZE), 1, 1);
  dim3 dimBlock(THREAD_SIZE, 1, 1);

  hipLaunchKernelGGL(mmIdxGet, dimGrid, dimBlock, 0, 0, deviceMinimizer, deviceOutputLisaPos, deviceOutputNumHits, deviceValuesEnc, deviceSortedArray, inputLength, n);
  hipLaunchKernelGGL(mmIdxGet, dimGrid, dimBlock, 0, 0, deviceMinimizer, deviceOutputLisaPos, deviceOutputNumHits, deviceValuesEnc, deviceSortedArray, inputLength, n);
  hipLaunchKernelGGL(mmIdxGet, dimGrid, dimBlock, 0, 0, deviceMinimizer, deviceOutputLisaPos, deviceOutputNumHits, deviceValuesEnc, deviceSortedArray, inputLength, n);
  hipLaunchKernelGGL(mmIdxGet, dimGrid, dimBlock, 0, 0, deviceMinimizer, deviceOutputLisaPos, deviceOutputNumHits, deviceValuesEnc, deviceSortedArray, inputLength, n);
  hipLaunchKernelGGL(mmIdxGet, dimGrid, dimBlock, 0, 0, deviceMinimizer, deviceOutputLisaPos, deviceOutputNumHits, deviceValuesEnc, deviceSortedArray, inputLength, n);

  hipDeviceSynchronize();
  
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start, 0);

  int cycles = 100;
  for (int i = 0; i < cycles; i++) {
    hipLaunchKernelGGL(mmIdxGet, dimGrid, dimBlock, 0, 0, deviceMinimizer, deviceOutputLisaPos, deviceOutputNumHits, deviceValuesEnc, deviceSortedArray, inputLength, n);
  }
  hipDeviceSynchronize();
  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);
  float milliseconds = 0;
  hipEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Average elapsed time: " << milliseconds / cycles << " ms" << std::endl;

  hipEventDestroy(start);
  hipEventDestroy(stop);

  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  hipMemcpy(hostOutputLisaPos, deviceOutputLisaPos, inputLength * sizeof(int64_t), hipMemcpyDeviceToHost);
  hipMemcpy(hostOutputNumHits, deviceOutputNumHits, inputLength * sizeof(int), hipMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  bool isResultCorrect = true;
  for (int i = 0; i < inputLength; i++) {
    if (hostOutputLisaPos[i] != hostLisaPos[i]) {
      isResultCorrect = false;
      break;
    }

    if (hostOutputNumHits[i] != hostNumHits[i]) {
      isResultCorrect = false;
      break;
    }
  }

  std::ofstream outfile1("./test/lisa_pos_l1_parameters_constant_memory_result.dat");
  std::ofstream outfile2("./test/num_hits_l1_parameters_constant_memory_result.dat");
  outfile1 << inputLength << std::endl;
  outfile2 << inputLength << std::endl;
  for (size_t i = 0; i < inputLength; i++) {
      outfile1 << hostOutputLisaPos[i] << std::endl;
      outfile2 << hostOutputNumHits[i] << std::endl;
  }
  outfile1.close();
  outfile2.close();

  if (isResultCorrect) {
    printf("The solution is correct\n");
  } else {
    printf("The solution is NOT correct\n");
  }
  
  hipFree(deviceMinimizer);
  hipFree(deviceOutputLisaPos);
  hipFree(deviceOutputNumHits);
  hipFree(deviceValuesEnc);

  free(hostMinimizer);
  free(hostLisaPos);
  free(hostNumHits);
  free(hostOutputLisaPos);
  free(hostOutputNumHits);
  free(L0_PARAMETERS);
  free(L1_PARAMETERS);
  free(sortedArray);

  return 0;
}
