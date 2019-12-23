#!/usr/bin/env python

import pandas as pd

url = 'https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.csv'
data = pd.read_csv(url,dtype={'Port Number':object})

# print(data)

tcp_filter=data['Transport Protocol']!="udp" #using udp mismatch to catch tcp and blank entries

for port in (data.where(tcp_filter)['Port Number'].unique()):
        if '-' in str(port):
                ports=port.split('-')
                for port in range(int(ports[0]),int(ports[1])):
                        print(port)
        else:
                print(port)

# for row in cr:
#     print(row)
