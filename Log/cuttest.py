#!/usr/bin/python3
import sys
import csv

TRACE_BACK_ANCHOR = 10
INV = 1
WARP_SIZE = 64

if len(sys.argv) < 4:
    print("Usage: "+ sys.argv[0] + " <input file> <cut output> <workload balance>")
    exit(1)

## cut long pred range with pred range > 64
fin = open(sys.argv[1], "r")
fout = open(sys.argv[2], "w+")
fwl = open(sys.argv[3], "w+")
reader = csv.reader(fin, delimiter=',')
read_id = 0

import collections
import math

def cut(pre_list):
    for i in range(len(pre_list)):
        if pre_list[i] == 0:
            return i
    return "x"

total_actwork = 0
total_work = 0

for row in reader:
    anchor_id = 0
    cut_list = []
    fout.write("#%d"%(read_id))
    pred_range = collections.deque([0] * TRACE_BACK_ANCHOR)

    work = 0
    actual_work = 0

    long_anchor = False
    for elt in row:
        try:
            elt_int = int(elt)
        except Exception:
            continue
        pred_range.appendleft(elt_int)
        pred_range.pop()

        if anchor_id % 512 == 0:
            cut_idx = cut(list(pred_range))
            if cut_idx == "x":
                cut_list.append("x")
                long_anchor = True
            else:
                if long_anchor:
                    # print("#%d: work %d act work %d" % (read_id, work, actual_work))
                    fwl.write("%d,%d\n" % (work, actual_work))
                    total_actwork += actual_work
                    total_work += work
                
                work = 0
                actual_work = cut_idx
                for i in range(cut_idx):
                    work += pred_range[i+1]
                
                cut_idx = anchor_id - cut_idx
                cut_list.append(cut_idx)

                long_anchor = False

        work += elt_int
        actual_work += int(math.ceil(elt_int/WARP_SIZE))
        anchor_id += 1
    
    for cut_idx in cut_list:
        fout.write("," + str(cut_idx))
        # print("," + str(cut_idx), end="")
    # print("")
    fout.write("\n")

    read_id += 1

print("total work needed for long anchor kernel: %d\ntotal actual work for long anchor kernel (sequencial): %d" % (total_work, total_actwork))