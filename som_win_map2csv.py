#!/usr/bin/env python
from minisom import MiniSom
import pickle
import sys

def som_members(win_map):
    for i,j in win_map:
        label=i*8+j
        for e in win_map[i,j]:
            print(label,",".join([str(i) for i in e.tolist()]),sep=',')

with open ('data/ip2.nmap.online.hosts.masscan.csv.optimised.csv', 'r') as fp:
    header = fp.readline().strip()

with open(sys.argv[1], 'rb') as infile:
    win_map = pickle.load(infile)

print('cluster',header,sep=',')

som_members(win_map)
