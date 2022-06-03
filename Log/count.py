#!/usr/bin/python
import sys
import csv
import time
if len(sys.argv) != 3:
    print("Usage: " + sys.argv[0] + " <import file> <output file>")
    exit(1)
dict = {}
list = []
with open(sys.argv[1]) as fin:
    reader = csv.reader(fin, delimiter=',')
    idx = 0
    for row in reader:
        for elt in row:
            if elt in dict.keys():
                dict[elt] = dict[elt] + 1
            else:
                dict[elt] = 1
            if int(elt) > 512:
                print(elt + "," + str(idx))    
            idx = idx + 1

#with open(sys.argv[2], 'w') as fout:
#    for key in dict.keys():
#        fout.write(key + "," + str(dict[key]) + "\n")
#    for elt in list:
#        fout.write(elt + ",")
