#!/bin/bash

#SBATCH -o /home/test_output.%j.out
#SBATCH -D /home/AtacWorks/
#SBATCH -J AtacworkJob_CPU
#SBATCH --nodes=8
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=64
##SBATCH --time=8:00:00

hostname
export MASTER_PORT=29500
export LD_LIBRARY_PATH=/home/libxsmm/lib/

source /opt/intel/compilers_and_libraries_2020.2.254/linux/mpi/intel64/bin/mpivars.sh release_mt

source /home/anaconda3/etc/profile.d/conda.sh
conda activate Atac
atacworks=/home/AtacWorks/

export KMP_AFFINITY=compact,1,0,granularity=fine

export OMP_NUM_THREADS=30       # number of cores - 2 (N - 2)

export LD_PRELOAD=/home/libtcmalloc.so 
export LD_PRELOAD=/home/libjemalloc.so

mpiexec.hydra -np 16 -l -genv I_MPI_PIN_DOMAIN=[0x00000000FFFFFFFC,0xFFFFFFFC00000000] -genv CCL_WORKER_AFFINITY=0,32 \
                        -genv CCL_WORKER_COUNT=1 -genv OMP_NUM_THREADS=30 \
                        python $atacworks/scripts/main.py train \
                        --config configs/train_config.yaml \
                        --config_mparams configs/model_structure.yaml \
                        --files_train $atacworks/Mono.50.2400.train.h5 \
                        --val_files $atacworks/Mono.50.2400.val.h5
