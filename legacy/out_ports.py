#!/usr/bin/env python

list={}
port_list={}
max_ports=1000

f= open("data/random_ip.out","r")
for x in f:
    r=x.strip().split(' ')
    if 'Host:' in r[0]:
        host=r[1]
        if host not in list:
            list[host]=[0]*(max_ports+1)
        port=int(r[3].split('/')[0])
        list[host][port]=1


f= open("data/random_ip.out","r")
for x in f:
    r=x.strip().split(' ')
    if 'Host:' in r[0]:
        host=r[1]
        if sum(list[host])<50:
            port=int(r[3].split('/')[0])
            if port not in port_list:
                port_list[port]=1
            else:
                port_list[port]=port_list[port]+1

for port in port_list:
    print(port_list[port],port)