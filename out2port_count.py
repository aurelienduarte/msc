#!/usr/bin/env python
import sys

list={}
max_ports=65535

f= open(sys.argv[1],"r")
for x in f:
    r=x.replace("\t"," ").strip().split(" ")
    if 'Host:' in r[2]:
        if '/' in r[-1]:
            port=int(r[-1].split('/')[0])
            if port not in list:
                list[port]=0
            list[port]+=1

#header
print('Port,Frequency')

for key, value in sorted(list.items(), key=lambda item: item[1],reverse=True):
    print(key, value,sep=',')
