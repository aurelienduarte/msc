#!/usr/bin/env python
import pandas as pd
import pickle
from random import shuffle
import sys

preamble=sys.argv[1]

with open(preamble+'ip2.nmap.online.hosts.masscan.csv.optimised.test.16.map6', 'rb') as infile:
    tests = pickle.load(infile)

# df = pd.read_csv(sys.argv[1])

print('index','cost','port','progress',sep=',')

cost=10
total={}

search_list=list(range(0,65536))

for index, row in tests.iterrows():
    total[index]=0
    shuffle(search_list)
    ports_total=row.sum(axis=0)
    ports_sum=0
    for port in search_list:
        column_name='port'+str(port)
        total[index]+=cost
        if column_name in row.keys():
            if (row[column_name]==1):
                ports_sum+=1
                print(index,total[index],port,round(ports_sum/ports_total,5),sep=',')
