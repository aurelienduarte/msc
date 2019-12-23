#!/usr/bin/env python
import sys

list={}
max_ports=65535

f= open(sys.argv[1],"r")
ports=''
for x in f:
    r=x.strip().split(' ')
    if 'Host:' in r[0]:
        host=r[1]
        if host not in list:
            list[host]=[]
        port=int(r[-1].split('/')[0])
        list[host].append(port)

# #header
# # print('host', end = '')
# output=''
# for i in range(max_ports+1):
#     output+=',port'+str(i)
# print(output.strip(','))

for host in list:
    output=''
    # print(sum(list[host]))
    # if sum(list[host])<50:
    # print(host, end = '')
    for item in list[host]:
        output+=','+str(item)
    print('nmap -sT -Pn -n -p',output.strip(','),'--open --append-output -oG random_ip.nmap.out',host)
