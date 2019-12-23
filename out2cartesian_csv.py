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
ports=[]

print('x','y','label',sep=',')
f= open(sys.argv[1],"r")
for x in f:
    r=x.strip().split(' ')
    if 'Host:' in r[0]:
        port=int(r[-1].split('/')[0])
        # if not port in list:
        # list.append(port)
        x=port%256
        y=port//256
        x_a.append(x)
        y_a.append(y)
        data[x][y]+=1
        ports.append(port)
        # print(x,y,port,sep=',')
            
# heatmap, xedges, yedges = np.histogram2d(x_a, y_a, bins=256)
# # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# # Plot heatmap
# plt.clf()
# plt.title('Ports heatmap')
# plt.ylabel('y')
# plt.xlabel('x')
# plt.imshow(heatmap, cmap='hot', interpolation='nearest')
# plt.show()

plt.hist2d(x_a, y_a, bins=(256,256), normed=False, cmap='hot')
plt.show()

# ex_dic = {
#     'x': x_a,
#     'y': y_a
# }
# columns = ['x', 'y']

# df = pd.DataFrame(ex_dic, columns=columns)

# ax = sns.heatmap(data)
# plt.show()

# n, bins, patches = plt.hist(x=ports, bins=1024, color='#0504aa',
#                             alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Port')
# plt.ylabel('Frequency')
# plt.title('TCP Port Histogram')
# plt.text(23, 45, r'$\mu=15, b=3$')
# maxfreq = n.max()
# # Set a clean upper y-axis limit.
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# hist, xedges, yedges = np.histogram2d(x_a, y_a, bins=256)

# # Add title and labels to plot.
# plt.title('3D histogram of 2D normally distributed data points')
# plt.xlabel('x axis')
# plt.ylabel('y axis')

# # Construct arrays for the anchor positions of the bars.
# # Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
# # ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
# # with indexing='ij'.
# xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
# xpos = xpos.flatten('F')
# ypos = ypos.flatten('F')
# zpos = np.zeros_like(xpos)

# # Construct arrays with the dimensions for the 16 bars.
# dx = 0.5 * np.ones_like(zpos)
# dy = dx.copy()
# dz = hist.flatten()

# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

# # Show the plot.
# plt.show()
