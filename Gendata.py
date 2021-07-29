import numpy as np
a = np.random.rand(100,1,1,1)
print(a)
np.save('b.npy',a)

b = np.load("b.npy")
print(b)
print(len(b))