## 1. Cold Start
Last output: 
[M::mm_idx_stat::54.802*1.72] distinct minimizers: 100167746 (38.80% are singletons); average occurrences: 5.519; average spacing: 5.607; total length: 3099922541

## 2. Alinger
call stack to main workload

[M::main.c::428/433](mm_map_file->)mm_map_file_frag
[M::map.c::656]kt_pipeline  // creates 3 threads, one is reader (step0), one is mapper (step1), last one is writer (step2)
[M::kthread.c::154]ktp_worker   //  this function handles lock between threads, makes sure each worker does one job, and feed data between different workers
[M::kthread.c::118]func = worker_pipeline ([map.c::526])
[M::map.c::561] kt_for (create thread for) worker_for ([map.c::414]) // creates threads for mapper, worker_for is kt_for callback
[M::map.c::434/439] mm_map_frag([map.c::232]) // the MAIN workload called for each requery inside a for loop.  START FROM HERE! 
