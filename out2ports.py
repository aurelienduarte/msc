#!/usr/bin/env python
import sys
import pickle

port_list={}

f= open(sys.argv[1],"r")
#f= open('data/ip2.nmap.online.hosts.masscan',"r")
for x in f:
    r=x.replace("\t"," ").strip().split(" ")
    if len(r)>3:
        if 'Host:' in r[2]:
            host=r[3]
            if host not in port_list:
                port_list[host]=set() #avoid duplicate ports
            if '/' in r[-1]:
                port=int(r[-1].split('/')[0])
                port_list[host].add(port)

output=[]
for host in port_list:
    if (len(port_list[host])>1): #remove single ports
        if (len(port_list[host])<1001):
            s=list(port_list[host])
            s.sort() #sort port_list
            output.append(s)

output.sort()
output = [output[i] for i in range(len(output)) if i == 0 or output[i] != output[i-1]] #remove duplicates

for example in output:
    print((str(example)[1:-1]).replace(' ','').replace(',',"\n"))
