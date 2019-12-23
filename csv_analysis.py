#!/usr/bin/env python
import sys
import logging
import pandas as pd

#Configure logging
logging.basicConfig(level=logging.INFO,
format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ])

logger = logging.getLogger()

logger.info('Loading data')
df = pd.read_csv(sys.argv[1])
# print(df)
# logger.info('Pre-Sanitisation')
# logger.info("\n"+str(df.describe()))

sum_attributes=df.sum(axis=0).sort_values(ascending=False)
logger.info(sum_attributes)

sum_samples=df.sum(axis=1).sort_values(ascending=False)
logger.info(sum_samples)

i=0
for key, value in sorted(sum_samples.items(), key=lambda item: item[1]):
    if value<1000:
        i+=1
    print(key, value,sep=',')

# tcp_filter=data['Transport Protocol']!="udp" #using udp mismatch to catch tcp and blank entries

# for port in (data.where(tcp_filter)['Port Number'].unique()):
#         if '-' in str(port):
#                 ports=port.split('-')
#                 for port in range(int(ports[0]),int(ports[1])):
#                         print(port)
#         else:
#                 print(port)

# # for row in cr:
# #     print(row)
