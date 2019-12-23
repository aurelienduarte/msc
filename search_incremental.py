#!/usr/bin/env python
import pandas as pd
import pickle
import sys

search_list=list(range(0,65536)) #list from 0 to 65535

preamble=sys.argv[1]

with open(preamble+'ip2.nmap.online.hosts.masscan.csv.optimised.test.16.map6', 'rb') as infile:
    tests = pickle.load(infile)

print('index','cost','port','progress',sep=',')

cost=10
total={}

for index, row in tests.iterrows():
    total[index]=0
    ports_total=row.sum(axis=0)
    ports_sum=0
    for port in search_list:
        column_name='port'+str(port)
        total[index]+=cost
        if column_name in row.keys():
            if (row[column_name]==1):
                ports_sum+=1
                print(index,total[index],port,round(ports_sum/ports_total,5),sep=',')
