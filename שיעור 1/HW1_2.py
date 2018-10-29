import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def culc_med(n,p):
    d = (1 - (1/2) ** (1/n)) ** (1/p)
    return d

n = [100, 5000, 100000]
colors = ['r', 'b', 'g']
p = [3, 5, 10, 20, 50, 100]
d = {}

for i,color in zip(n,colors):
    d[i] = []
    for j in p:
        d[i].append(culc_med(i,j))
    plt.plot(p, d[i], color=color, marker='o', linestyle='dashed',linewidth = 2, markersize = 12, label=i)

plt.xlabel('p value')
plt.ylabel('d(p,n) values')
plt.legend()
plt.show()



