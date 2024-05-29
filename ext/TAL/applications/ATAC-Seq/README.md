# Instructions for ubuntu

## Update
```bash
apt-get update
```

## Install software
```bash
apt-get install -y git
apt-get install -y libcurl4-openssl-dev
apt-get install -y hdf5-tools
apt-get install -y rsync
apt-get install -y make
apt-get install -y gcc
apt-get install -y libblas-dev
apt-get install -y python3.7 python3-pip
ln -nsf /usr/bin/python3.7 /usr/bin/python
```
## Anaconda Environment
```bash
conda create --name Atac python=3.7
conda activate Atac
```

## Clone the libxsmm repository and set library path
```bash
cd /home/
git clone https://github.com/libxsmm/libxsmm.git
cd /home/libxsmm
git checkout b3da2b1bed9d27f9d6bae91a683f8cf76fe299b5
make -j                   # Use AVX=2 for AVX2 and AVX=3 for AVX512
cd /home/               
export LD_LIBRARY_PATH=/home/libxsmm/lib/
```

## Clone atacworks repo
```bash
git clone --branch v0.2.0 https://github.com/clara-parabricks/AtacWorks.git
```

## Clone the OpenOmics version
```bash
git clone https://github.com/IntelLabs/Trans-Omics-Acceleration-Library.git
```

## Apply patch
```bash
cd  /home/AtacWorks/
git apply /home/Trans-Omics-Acceleration-Library/applications/ATAC-Seq/AtacWorks_cpu_optimization_patch.patch
```

## Install python packages
```bash
python3.7 -m pip install -r requirements-base.txt
python3.7 -m pip install torch torchvision torchaudio
python3.7 -m pip install -r requirements-macs2.txt
```

## (Optional) Install torch-ccl
```bash
# Install torch-ccl
# git clone --branch v1.1.0 https://github.com/intel/torch-ccl.git && cd torch-ccl
# git submodule sync
# git submodule update --init --recursive
# python3.7 setup.py install
```

## Install 1D convolution module
```bash
cd /home/libxsmm/samples/deeplearning/conv1dopti_layer/Conv1dOpti-extension/ 
python setup.py install
```

## Install AtacWorks folder ans set path
```bash
cd /home/AtacWorks/
python3.7 -m pip install .
atacworks=/home/AtacWorks/
```

## Download data to train
```bash
wget https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/train_data/noisy_data/dsc.1.Mono.50.cutsites.smoothed.200.bw
wget https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/train_data/clean_data/dsc.Mono.2400.cutsites.smoothed.200.bw
wget https://atacworks-paper.s3.us-east-2.amazonaws.com/dsc_atac_blood_cell_denoising_experiments/50_cells/train_data/clean_data/dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak
```

## Download file conversion binaries and set path
```bash
rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bedGraphToBigWig /home/
rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bigWigToBedGraph /home/
export PATH="$PATH:/home/" >> /home/.bashrc         # set the path for bedGraphToBigWig binaries 
```

## Data preprocessing

```python
python $atacworks/scripts/peak2bw.py \
    --input dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak \
    --sizes $atacworks/data/reference/hg19.chrom.sizes \
    --out_dir ./ \
    --skip 1

python $atacworks/scripts/get_intervals.py \
    --sizes $atacworks/data/reference/hg19.auto.sizes \
    --intervalsize 50000 \
    --out_dir ./ \
    --val chr20 \
    --holdout chr10

python $atacworks/scripts/bw2h5.py \
        --noisybw dsc.1.Mono.50.cutsites.smoothed.200.bw \
        --cleanbw dsc.Mono.2400.cutsites.smoothed.200.bw \
        --cleanpeakbw dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak.bw \
        --intervals training_intervals.bed \
        --out_dir ./ \
        --prefix Mono.50.2400.train \
        --pad 5000 \
        --nonzero

python $atacworks/scripts/bw2h5.py \
        --noisybw dsc.1.Mono.50.cutsites.smoothed.200.bw \
        --cleanbw dsc.Mono.2400.cutsites.smoothed.200.bw \
        --cleanpeakbw dsc.Mono.2400.cutsites.smoothed.200.3.narrowPeak.bw \
        --intervals val_intervals.bed \
        --out_dir ./ \
        --prefix Mono.50.2400.val \
        --pad 5000
```

## Set affinity and threads
```bash
export KMP_AFFINITY=compact,1,0,granularity=fine
export LD_PRELOAD=/home/libtcmalloc.so              # Copy these files in the /home folder first
export LD_PRELOAD=/home/libjemalloc.so
export OMP_NUM_THREADS=31                           # (Available cores (N) - 1)
```

## Training run (Single Socket)
```python
# In numactl command, "-C 1-31" is for running on cores 1 to 31. 
# General case for an N core machine is "-C 1-(N-1)".
# Keep batch size in config/train_config.yaml to a multiple of (N-1) for optimum performance

numactl --membind 0 -C 1-31 python $atacworks/scripts/main.py train \
        --config configs/train_config.yaml \
        --config_mparams configs/model_structure.yaml \
        --files_train $atacworks/Mono.50.2400.train.h5 \
        --val_files $atacworks/Mono.50.2400.val.h5                           
```

Option - Another option to use on machines without NUMA --- "taskset -c 1-31 python ..."

## Training run (Multiple Sockets/Nodes)

``` bash
export OMP_NUM_THREADS=30                           # (Available cores (N) - 2)
# 1. change line 23 in configs/train_config.yaml with the following
#        dist-backend: 'gloo' 
# 2. change line 22 in configs/train_config.yaml with the following
#        dist-backend: 'gloo'
# 3. Keep batch size (bs) in config/train_config.yaml to a multiple of (N-2) for optimum performance. 
#    Batch size gets multiplied by number of socket. Hence, if bs=30, no. of sockets = 16 than batch size = 30*16 = 480
# 4. Comment line the following line (79,80) in AtacWorks/claragenomics/dl4atac/utils.py and reinstall AtacWorks using "pip install ." command.
#       if (os.path.islink(latest_symlink)):
#               os.remove(latest_symlink)
# 5. Run the following Slurm batch script that uses MPI commands.

sbatch Batchfile_CPU.slurm

```
