import numpy as np
from dinosaur.princals import PRINCALS

data = np.array([
    [1, 1, 1, 2, 2],  # Crowd
    [2, 3, 2, 2, 2],  # Modern community, Neighborhood
    [1, 2, 2, 1, 1],  # Public
    [4, 4, 4, 2, 3],  # Primary Group
    [4, 1, 4, 2, 3],  # Mob
    [3, 3, 3, 1, 2],  # Secondary Group
    [2, 1, 2, 2, 2]   # Audience
])

var_types = ['ordinal', 'ordinal', 'ordinal', 'nominal', 'nominal']


model = PRINCALS(X=data, n_components=2, var_types=var_types)
Z, W = model.fit(max_iter=100, tol=1e-6)

print("Object Scores (Z):")
print(Z)

print("\nComponent Loadings (W):")
for i, w in enumerate(W):
    print(f"W{i+1}:")
    print(w)

print("A: ")
print(model.A)
print("---------")
print("Y ")
for i in model.Y:
    print(i)