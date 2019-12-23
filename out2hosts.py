#!/usr/bin/env python
import sys

list={}
max_ports=65535

f= open(sys.argv[1],"r")
for x in f:
    r=x.replace("\t"," ").strip().split(" ")
    if 'Host:' in r[2]:
        host=r[3]
        if host not in list:
            list[host]=0

for host in list:
    print(host)
