#!/usr/bin/env python
import sys

list={}
max_ports=65535

f= open(sys.argv[1],"r")
for x in f:
    r=x.replace("\t"," ").strip().split(" ")
    if len(r)>3:
        if 'Host:' in r[2]:
            host=r[3]
            if host not in list:
                list[host]=0
            list[host]+=1

#header
print('host,port_count')

for key, value in sorted(list.items(), key=lambda item: item[1],reverse=True):
    print(key, value,sep=',')
