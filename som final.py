from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns


df = pd.read_csv('data/ip2.nmap.online.hosts.masscan.csv.optimised.csv')

som16 = MiniSom(16, 16, 39514) # initialization of 16x16 SOM
som16.train_batch(df.values, 100) # trains the SOM with 100 iterations

with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.som16', 'wb') as outfile:
    pickle.dump(som16, outfile)

with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.som16', 'rb') as infile:
    som16 = pickle.load(infile)


som10 = MiniSom(10, 10, 39514) # initialization of 10x10 SOM
som10.train_batch(df.values, 100) # trains the SOM with 100 iterations

with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.som10', 'wb') as outfile:
    pickle.dump(som10, outfile)

with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.som10', 'rb') as infile:
    som10 = pickle.load(infile)

som8 = MiniSom(8, 8, 39514) # initialization of 8x8 SOM
som8.train_batch(df.values, 100) # trains the SOM with 100 iterations

with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.som8', 'wb') as outfile:
    pickle.dump(som8, outfile)

with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.som8', 'rb') as infile:
    som8 = pickle.load(infile)

som6 = MiniSom(6, 6, 39514) # initialization of 6x6 SOM
som6.train_batch(df.values, 100) # trains the SOM with 100 iterations

with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.som6', 'wb') as outfile:
    pickle.dump(som6, outfile)

with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.som6', 'rb') as infile:
    som6 = pickle.load(infile)


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(som16.distance_map())
plt.show()

sns.heatmap(som10.distance_map())
plt.show()

win_map16 = som16.win_map(df.values)
len(win_map16), type(win_map16)

def som_member_count(win_map):
    for i,j in win_map:
        label=str(i)+'_'+str(j)
        print(label,len(win_map[i,j]),sep=',')

def som_members(win_map):
    for i,j in win_map:
        label=str(i)+'_'+str(j)
        for e in win_map[i,j]:
            print(label,",".join([str(i) for i in e.tolist()]),sep=',')

som_member_count(win_map16)
som_members(win_map16)

win_map10 = som10.win_map(df.values)
len(win_map10), type(win_map10)

som_member_count(win_map10)
som_members(win_map10)

win_map8 = som8.win_map(df.values)
len(win_map8), type(win_map8)

som_member_count(win_map8)
som_members(win_map8)

sns.heatmap(som8.distance_map())
plt.show()

win_map6 = som6.win_map(df.values)
len(win_map6), type(win_map6)

som_member_count(win_map6)
som_members(win_map6)

sns.heatmap(som6.distance_map())
plt.show()

with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.som_win_map16', 'wb') as outfile:
    pickle.dump(win_map16, outfile)


with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.som_win_map10', 'wb') as outfile:
    pickle.dump(win_map10, outfile)


with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.som_win_map8', 'wb') as outfile:
    pickle.dump(win_map8, outfile)


with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.som_win_map6', 'wb') as outfile:
    pickle.dump(win_map6, outfile)


