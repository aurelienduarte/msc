#!/usr/bin/env python
import sys
import pandas as pd
import logging

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
logger.info('Pre-Optimisation')
logger.info(df.describe())

#remove duplicate samples as per "Handbook of Statistical Analysis and Data Mining Applications (Second Edition) 2018, Pages 727-740 - A Data Preparation Cookbook.pdf"
df = df.drop_duplicates()
logger.info('Duplicate rows removed')
logger.info(df.describe())

#remove examples with single ports
sum_samples=df.sum(axis=1).sort_values(ascending=False)
logger.info(sum_samples)

row_sum_one=[]
for index, value in sum_samples.items():
    if (value==1):
        row_sum_one.append(index)

if (len(row_sum_one)>0):
    logger.info('Delete single attribute row count: %d' % len(row_sum_one))
    df=df.drop(row_sum_one)


#delete samples with more than 1000 open ports
row_sum_limit=[]
for index, value in sum_samples.items():
    if (value>999):
        row_sum_limit.append(index)

if (len(row_sum_limit)>0):
    logger.info('Delete limit (999) attribute row count: %d' % len(row_sum_limit))
    df=df.drop(row_sum_limit)


#remove attributes with 0 variance
df = df.loc[:,df.apply(pd.Series.nunique) != 1]
logger.info('Zero variance attributes removed')
logger.info(df.describe())

df.to_csv(sys.argv[1]+'.optimised.csv', sep=',', encoding='utf-8', index=False)
