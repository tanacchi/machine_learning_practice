# page 9
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)   # [-10. -9.7979798 ... 9.7979798   10.]
y = np.sin(x)                   # [ 0.54402111  0.36459873 ... -0.36459873 -0.54402111]

plt.plot(x, y, marker="x")
plt.show()
