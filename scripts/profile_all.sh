#!/bin/bash

# Array of LONG_BLOCK_SIZE values
# DATA_SETS=( 1kto5k 1kto10k 1kto50k 1kto300k 200kto300k )
# DATA_SETS=( 1kto5k 1kto10k 1kto50k 1kto300k 200kto300k 1kto20k 1kto30k 1kto70k 1kto200k 10kto50k 10kto100k 50kto100k)
# DATA_SETS=( 1kto300k 50kto300k 100kto300k 150kto300k 200kto300k 250kto300k 1kto200k 20kto200k 50kto200k 70kto200k 100kto200k 130kto200k 150kto200k 170kto200k)
DATA_SETS=( 1kto5k 9kto10k 10kto20k 20kto30k 40kto50k 90kto100k 110kto120k 140kto150k 180kto200k 200kto250k 200kto300k )
# NUM_THREADS=( 32 1 )
# CORE_COMBO=( "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62" "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30" "0,2,4,6,8,10,12,14" "0,2,4,6" "0,2" "1")
CORE_COMBO=( "1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31"  "1,3,5,7,9,11,13,15" "1,3,5,7" "1,3" "1" )
NUM_THREADS=(16 8 4 2 1)
COMBO_IDS=( 0 1 2 3 4 )
make clean
make manual_profile=1

# Iterate over LONG_BLOCK_SIZES array
for DATA_SET in "${DATA_SETS[@]}"
do
    for COMBO_ID in "${COMBO_IDS[@]}"
    do
        echo "Running on data set ${DATA_SET}"
        echo "Using ${NUM_THREADS[COMBO_ID]} threads"
        echo "Using cores ${CORE_COMBO[COMBO_ID]}"

        filename="profile_out/data-${DATA_SET}_profile_${NUM_THREADS[COMBO_ID]}-cores"
        numactl -N 1 -m 1 --physcpubind=${CORE_COMBO[COMBO_ID]} ./minimap2 data/hg38.mmi data/random_500MBases_${DATA_SET}.fa -t ${NUM_THREADS[COMBO_ID]} --max-chain-skip=2147483647 > mm2-fast_output 2> $filename

    done       
done

# # Iterate over LONG_BLOCK_SIZES array
# for DATA_SET in "${DATA_SETS[@]}"
# do
#     for NUM_THREAD in "${NUM_THREADS[@]}"
#     do
#         for COMBO_ID in "${COMBO_IDS[@]}"
#         do
#             echo "Running on data set ${DATA_SET}"
#             echo "Using ${NUM_THREAD} threads"
#             echo "Using cores ${CORE_COMBO[COMBO_ID]}"

#             filename="profile_out/data-${DATA_SET}_profile_${NUM_THREAD}-threads_corecombo-${COMBO_ID}"
#             numactl --physcpubind=${CORE_COMBO[COMBO_ID]} ./minimap2 data/hg38.mmi data/random_500MBases_${DATA_SET}.fa -t ${NUM_THREAD} --max-chain-skip=2147483647 > mm2-fast_output 2> $filename

#         done 
#     done       
# done
