#!/usr/bin/env python
import pandas as pd
from apyori import apriori
import pyfpgrowth
import pickle

import logging

#Configure logging
logging.basicConfig(level=logging.INFO,
format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ])

logger = logging.getLogger()

transactions = []
logger.info('Loading data')
with open ('data/ip2.nmap.online.hosts.masscan.tsv', 'r') as fp:
    example = fp.readline()
    while example:
        transactions.append([int(i) for i in example.rstrip().split(',')] )
        example = fp.readline()

association_rules = apriori(transactions, min_support=0.002, min_length=2)
association_results = list(association_rules)
print(len(association_results))

print()

for rule in association_results:
    print(rule)

