# Standalone RMI Seeding Lookup
This is a standalone RMI Seeding Lookup process that use minimizer to look up position from the RMI. The lookup process does not include the RMI construction process. Thus the RMI must be constructed with mm2-fast before running the lookup process.  
There are 5 implementations in the folder:
1. `cpu.cu` - CPU implementation
2. `gpu.cu` - GPU implementation
3. `l1ParametersConstantMemory.cu` - GPU implementation with L1 parameters stored in constant memory
4. `sortedArrayConstantMemory.cu` - GPU implementation with sorted array stored in constant memory
5. `allConstantMemory.cu` - GPU implementation with L1 parameters and sorted array stored in constant memory

## Setup
To setup the RMI seeding lookup process, you have to follow the following steps:
1. Clone the [mm2-fast](https://github.com/bwa-mem2/mm2-fast) repository  
```bash
git clone --recursive https://github.com/bwa-mem2/mm2-fast.git mm2-fast
cd mm2-fast
```
2. Download rustup
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
3. Setup rmi parameters
```bash
./build_rmi.sh test/MT-human.fa map-ont  

cp -rf test/MT-human.fa_map-ont_minimizers_key_value_sorted_keys.uint64 test/MT-human.fa_map-ont_minimizers_key_value_sorted_keys.rmi_PARAMETERS
```
4. Build the RMI index using the following command:  
```bash
# Start by building learned hash table index for optimized seeding module 
./build_rmi.sh test/MT-human.fa map-ont  
# Next, compile and run the mapping phase  
make clean && make lhash=1
./minimap2 -ax map-ont test/MT-human.fa test/MT-orang.fa > mm2-fast-lhash_output
```
5. Copy the RMI index to the `rmi-seeding-lookup/test` folder. The RMI index files include:
- `MT-human.fa_map-ont_minimizers_key_value_sorted_keys.rmi_PARAMETERS`
- `MT-human.fa_map-ont_minimizers_key_value_sorted_keys.uint64`
- `MT-human.fa_map-ont_minimizers_key_value_sorted_pos_bin`
- `MT-human.fa_map-ont_minimizers_key_value_sorted_size`
- `MT-human.fa_map-ont_minimizers_key_value_sorted_val_bin`

6. Extract minimizer from the mm2-fast seeding process.  
The code will export minimizers, lisa_pos, and num_hits to the binary and human readable format.  
To extract minimizers, lisa_pos, and num_hits, add the following code to the `seed.c` file in the mm2-fast folder:
```cpp
#include <fstream>
...
// export minimizers in binary format
const char* filename_minimizers = "./test/MT-human.fa_map-ont_minimizers";
std::ofstream file_minimizers(filename_minimizers, std::ios::binary);
file_minimizers.write(reinterpret_cast<const char*>(minimizers), mv->n * sizeof(int64_t));
file_minimizers.close();

// export minimizers in human readable format
const char* filename_minimizers_readable = "./test/mm2-fast_minimizer.dat";
std::ofstream file_minimizers_readable(filename_minimizers_readable);
file_minimizers_readable << mv->n << std::endl;
for (int i = 0; i < mv->n; i++) {
	file_minimizers_readable << minimizers[i] << std::endl;
}
file_minimizers_readable.close();

lh->mm_idx_get_batched(minimizers, mv->n, lisa_pos, cr_batch, t_batch);

// export lisa_pos in binary format
const char* filename_lisa_pos = "./test/MT-human.fa_map-ont_lisa_pos";
std::ofstream file_lisa_pos(filename_lisa_pos, std::ios::binary);
file_lisa_pos.write(reinterpret_cast<const char*>(lisa_pos), mv->n * sizeof(int64_t));
file_lisa_pos.close();

// export lisa_pos in human readable format
const char* filename_lisa_pos_readable = "./test/mm2-fast_lisa_pos.dat";
std::ofstream file_lisa_pos_readable(filename_lisa_pos_readable);
file_lisa_pos_readable << mv->n << std::endl;
for (int i = 0; i < mv->n; i++) {
	file_lisa_pos_readable << lisa_pos[i] << std::endl;
}
file_lisa_pos_readable.close();

// export num_hits in binary format
const char* filename_num_hits = "./test/MT-human.fa_map-ont_minimizers_num_hits";
std::ofstream file_num_hits(filename_num_hits, std::ios::binary);
file_num_hits.write(reinterpret_cast<const char*>(t_batch), mv->n * sizeof(int));
file_num_hits.close();

// export num_hits in human readable format
const char* filename_num_hits_readable = "./test/mm2-fast_num_hits.dat";
std::ofstream file_num_hits_readable(filename_num_hits_readable, std::ios::binary);
file_num_hits_readable << mv->n << std::endl;
for (int i = 0; i < mv->n; i++) {
	file_num_hits_readable << t_batch[i] << std::endl;
}
file_num_hits_readable.close();
...
```
7. Run the mm2-fast to extract minimizers, lisa_pos, and num_hits.  
```bash
./minimap2 -ax map-ont test/MT-human.fa test/MT-orang.fa > mm2-fast_output
```
8. Copy the minimizers, lisa_pos, and num_hits in binary format to the `rmi-seeding-lookup` folder. The files include:
- `MT-human.fa_map-ont_minimizers`
- `MT-human.fa_map-ont_lisa_pos`
- `MT-human.fa_map-ont_minimizers_num_hits`
9. Copy the minimizers, lisa_pos, and num_hits in readable format to the `rmi-seeding-lookup/test` folder. The files include:
- `mm2-fast_minimizer.dat`
- `mm2-fast_lisa_pos.dat`
- `mm2-fast_num_hits.dat`
10. If using constant memory implementation, update `L1_SIZE` and `SORTED_ARRAY_SIZE` manually in the implementation file. They are defined at the beginning of the file.  
The values can be found in the terminal massage after running mm2-fast. The terminal will show the following message:
```
Using LISA_HASH..
Memory allocated 3111 
Num_keys: 3111, num_values = 3111Loading from bin
n = 3111
L0_PARAMETER0 = 1.537038E-320, L0_PARAMETER1 = 9.092290E-320, L1_SIZE = 82969
Loading done.
```
`L1_SIZE` is 82969 in this case, and `SORTED_ARRAY_SIZE` is shown as `n` which is 3111 in this case.

## Build
To build the project, run the following command:
```bash
./build <implementation.cu> <executable_name>
```

## Test
To test the project, run the following command:
```bash
./run_test <executable_name>
```
The test will compare the output of the implementation with the output of the mm2-fast.
The terminal will show the following message if the test is successful:
```
L0_PARAMETER0 = 1.537038E-320, L0_PARAMETER1 = 9.092290E-320, L1_SIZE = 82969
n = 3111
The input size is 3105
Average elapsed time: 0.00661277 ms
The solution is correct
```
After running the test, the human readable output will also be saved in the `test` folder, which can be manually compared with the mm2-fast output.  
The output files include:
- `lisa_pos_<implementation>_result.dat`
- `num_hits_<implementation>_result.dat`