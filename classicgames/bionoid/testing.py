





import numpy as np


A = np.random.randint(0, 2, size=5, dtype=bool)
B = np.random.randint(0, 2, size=(4, 5), dtype=bool)

print(A)
print(B)
print(A == B)
print((A == B).sum(axis=1, dtype=int))
