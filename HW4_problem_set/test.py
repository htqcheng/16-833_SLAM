import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D([0, 1, 2], [1, 1, 1], [3,3,3])
plt.savefig("3dtest.png")
plt.show()