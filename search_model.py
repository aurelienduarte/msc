#!/usr/bin/env python
import pandas as pd
import pickle
from random import shuffle
import logging
import time
import sys

closed_ports=verts = [0]*65536 #closed ports list from 0 to 65535

#Configure logging
logging.basicConfig(level=logging.INFO,
format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ])

logger = logging.getLogger()

preamble=sys.argv[1]

# print(preamble)

logger.info('Loading data')

with open(preamble+'ip2.nmap.online.hosts.masscan.csv.optimised.test.16.map6', 'rb') as infile:
    tests = pickle.load(infile)

with open(preamble+'ip2.nmap.online.hosts.masscan.csv.optimised.classifier.16.map6', 'rb') as infile:
    classifier = pickle.load(infile)

with open(preamble+'ip2.nmap.online.hosts.masscan.csv.optimised.train.16.map6', 'rb') as infile:
    train = pickle.load(infile)

with open(preamble+'ip2.nmap.online.hosts.masscan.csv.optimised.clusters.16.map6', 'rb') as infile:
    clusters = pickle.load(infile)

train['cluster']=clusters.argmax(axis=1)

# df = pd.read_csv(sys.argv[1])

print('index','cost','port','progress','cluster',sep=',')

cost=10
total={}

frequency_all=train.groupby('cluster').sum().sum(axis=0).sort_values(ascending=False)
frequency=train.groupby('cluster').sum()

freq={}

closed_ports_df=train.drop(columns=['cluster']).iloc[[0]].replace(1,0)

cluster_freq_exhausted=False
freq_exhausted=False

def update_current_cluster(cluster):
    global freq
    if cluster not in freq.keys():
        logger.info('Creating new freq: '+str(cluster))
        freq[cluster]=frequency.loc[[cluster]].sum(axis=0).sort_values(ascending=False).keys().array
        i=0
        f={}
        for a in freq[cluster]:
            f[i]=a
            i+=1
        freq[cluster]=f

def get_next_most_frequent(cluster,ports_detected_open):
    global cluster_freq_exhausted
    global freq_exhausted
    global freq

    update_current_cluster(cluster)
    if ports_detected_open==0:
        for i in frequency_all.keys():
            port=int(i.replace('port',''))
            if port not in tested:
                return port
        return None

# def ports_to_dataframe(ports):
#     start_time = time.time()
#     closed_ports=closed_ports_df.copy()
#     row_keys=closed_ports.keys()
#     # print(row_keys)
#     for key in row_keys:
#         port=int(key[4:])
#         closed_ports[key]=ports[port]
#     return closed_ports
#     # for index, port in enumerate(ports,start=0):
#     #     column_name='port'+str(index)
#     #     if column_name in row_keys:
#     #         closed_ports[column_name]=port
#     # return closed_ports

for index, row in tests.iterrows():
    search_list=list(range(0,65536))
    shuffle(search_list)
    search_list=set(search_list)
    tested=set()
    most_frequent=int(frequency_all.keys()[0].replace('port',''))
    cluster_pred=-1
    total[index]=0
    ports_total=row.sum(axis=0)
    ports_sum=0
    ports=closed_ports.copy()
    #
    df_closed_ports=closed_ports_df.copy()
    df_closed_ports_keys=set(df_closed_ports.keys())
    ports_detected_open=0
    cluster_freq_exhausted=False
    freq_exhausted=False
    row_keys=set(row.keys())
    while len(tested)!=65536:
        # current_list=
        tested_len=len(tested)
        if (tested_len % 1000 == 1):
            logger.info('index: '+str(index)+', testing most_frequent: '+str(most_frequent)+', cluster: '+str(cluster_pred)+', tested count: '+str(tested_len)+', ports_detected_open: '+str(ports_detected_open))
        
        if ports_sum==0:
            tested.add(most_frequent)
            column_name='port'+str(most_frequent)
            total[index]+=cost
            try:
                if (row[column_name]==1):
                    ports_sum+=1
                    port=int(column_name.replace('port',''))
                    ports[port]=1
                    if column_name in df_closed_ports_keys:
                        df_closed_ports[column_name]=1
                    progress=ports_sum/ports_total
                    print(index,total[index],port,progress,cluster_pred,sep=',')
            except:
                continue
            cluster_pred=classifier.predict(df_closed_ports).argmax(axis=1)[0]
            most_frequent=get_next_most_frequent(cluster_pred,ports_detected_open)

        if ports_sum==0:
            cluster_pred=-1
        
        if not freq_exhausted:
            if (cluster_pred!=-1):
                if not cluster_freq_exhausted:
                    update_current_cluster(cluster_pred)
                    for i in freq[cluster_pred].keys():
                        port=int(freq[cluster_pred][i][4:])
                        if port not in tested:
                            # del freq[cluster_pred][i]
                            tested.add(port)
                            column_name='port'+str(port)
                            total[index]+=cost
                            try:
                                if (row[column_name]==1):
                                    ports_sum+=1
                                    progress=ports_sum/ports_total
                                    port=int(column_name.replace('port',''))
                                    ports[port]=1
                                    if column_name in df_closed_ports_keys:
                                        df_closed_ports[column_name]=1
                                    print(index,total[index],port,progress,cluster_pred,sep=',')
                                    cluster_pred=classifier.predict(df_closed_ports).argmax(axis=1)[0]
                                    break
                            except:
                                continue
                    freq_exhausted=True

        if (freq_exhausted):
            # logger.info('freq_exhausted=True in while')
            for port in search_list:
                if port not in tested:
                    tested.add(port)
                    column_name='port'+str(port)
                    total[index]+=cost
                    try:
                        if (row[column_name]==1):
                            ports_sum+=1
                            progress=ports_sum/ports_total
                            port=int(column_name.replace('port',''))
                            ports[port]=1
                            if column_name in df_closed_ports_keys:
                                df_closed_ports[column_name]=1
                            print(index,total[index],port,progress,cluster_pred,sep=',')
                            cluster_pred=classifier.predict(df_closed_ports).argmax(axis=1)[0]
                            freq_exhausted=False
                            # logger.info('set freq_exhausted=False in while')
                            break
                    except:
                        continue
