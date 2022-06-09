#!/usr/bin/python
import sys
import csv
import time
if len(sys.argv) != 3:
    print("Usage: " + sys.argv[0] + " <input file> <output file>")
    exit(1)
lessthan64 = 0
morethan512 = 0
zeropred = 0
dict = {}
list = []
total_work = 0
work_lessthan_64 = 0
with open(sys.argv[1]) as fin:
    reader = csv.reader(fin, delimiter=',')
    idx = 0
    for row in reader:
        for elt in row:
            # if elt in dict.keys():
            #     dict[elt] = dict[elt] + 1
            # else:
            #     dict[elt] = 1
            try: 
                elt_int = int(elt)
            except Exception:
                continue
            total_work += elt_int
            if elt_int > 512:
                morethan512 += 1
                print(elt + "," + str(idx)) 
            elif elt_int <= 64:
                lessthan64 += 1
                work_lessthan_64 += elt_int
            if elt_int == 0:
                zeropred += 1
            idx = idx + 1
    print("Total: " + str(idx))
    print(str(lessthan64) + " has less than 64 predecessor")
    print(str(morethan512) + "has more than 512 predecessor")
    print(str(zeropred) + " has 0  predecessor range")
    print(str(work_lessthan_64) + " out of total work " + str(total_work))

# with open(sys.argv[2], 'w') as fout:
#    for key in dict.keys():
#        fout.write(key + "," + str(dict[key]) + "\n")
#    for elt in list:
#        fout.write(elt + ",")
