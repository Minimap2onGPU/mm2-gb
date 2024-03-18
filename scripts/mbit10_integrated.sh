#!/bin/bash
make GPU=NV GPUARCH=sm_86 DEBUG=analyze
cuda-gdb --args ./minimap2  -t 1 --gpu-chain --gpu-cfg gfx1030.json data/hg38.mmi data/ONT/random_500MBases_90kto100k.fa 