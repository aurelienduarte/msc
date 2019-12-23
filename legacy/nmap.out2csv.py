#!/usr/bin/env python
import sys

list_port={}
services=[]
max_ports=1000
open_only=False

undetected_filer=['tcpwrapped','unknown','']
if "-open_only" in sys.argv: #will only report in CSV that the port was open
    # undetected_filer=['unknown','']    
    undetected_filer=[]
    open_only=True  

f= open("data/random_ip.nmap.out","r")
for x in f:
    r=x.strip().split(' ')
    if 'Host:' in r[0]:
        host=r[1]
        if host not in list_port:
            list_port[host]=[0]*(max_ports+1)
        if 'Ports: ' in x:
            data_array=x.split('Ports: ')
            ports=data_array[1].split(', ')
            for port in ports:
                port_array=port.split('/')
                if len(port_array)==8:
                    port_number=int(port_array[0].strip())
                    service=port_array[4].strip()
                    if open_only:
                        if port_array[1]=='open':
                            list_port[host][port_number]=1
                    else:
                        if port_array[1]=='open' and ('?' not in service):
                            if service not in undetected_filer:
                                if service not in services:
                                    services.append(service)
                                list_port[host][port_number]=services.index(service)+2
                            else:
                                list_port[host][port_number]=1

# print(list_port)

print('host', end = '')
for i in range(max_ports+1):
    print(',port'+str(i), end = '')
print('')

for host in list_port:
    # print(sum(list[host]))
    if sum(list_port[host])<50:
        print(host, end = '')
        for item in list_port[host]:
                print(','+str(item), end = '')
        print('')
