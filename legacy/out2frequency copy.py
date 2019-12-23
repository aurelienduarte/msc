#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

plt.style.use('seaborn-whitegrid')

# list=[]
x_a=[]
y_a=[]
data=np.zeros((256,256))
max_ports=65535
ports={}

print('x','y','label',sep=',')
f= open(sys.argv[1],"r")
for x in f:
    r=x.strip().split(' ')
    # print(r)
    if 'Host:' in r[0]:
        port=int(r[-1].split('/')[0])
        if port not in ports:
            ports[port]=0
        ports[port]+=1
        # print(x,y,port,sep=',')
            
for key, value in sorted(ports.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
