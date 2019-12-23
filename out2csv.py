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
                list[host]=[0]*(max_ports+1)
            if '/' in r[-1]:
                port=int(r[-1].split('/')[0])
                list[host][port]=1

#header
# print('host', end = '')
output=''
for i in range(max_ports+1):
    output+=',port'+str(i)
print(output.strip(','))

for host in list:
    output=''
    # print(sum(list[host]))
    # if sum(list[host])<50:
    # print(host, end = '')
    for item in list[host]:
        output+=','+str(item)
    print(output.strip(','))
