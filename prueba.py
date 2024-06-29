from multiprocessing import cpu_count
import numpy as np

print(int(cpu_count()))

param = np.arange(0,10,1)
param1 = np.arange(30,40,1)

grid = np.meshgrid(param,param1)
print(np.vstack(list(map(np.ravel, grid))).T)
print(grid)
