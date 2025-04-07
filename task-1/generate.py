import numpy as np
import json

np.random.seed(0)

N = int(100000 * 1.5)
D = 512 * 2
K = 10

A = np.random.randn(N, D)
X = np.random.randn(D)

A_filename = f'data/A_{N}x{D}.txt'
X_filename = f'data/X_{D}.txt'
config_filename = f'data/test_config_{N}x{D}.json'

np.savetxt(A_filename, A)
np.savetxt(X_filename, X)

config = {'n': N, 'd': D, 'a_file': A_filename, 'x_file': X_filename, 'k': K}

with open(config_filename, 'w') as f:
  json.dump(config, f, indent=2)

print('Data files and configuration created.')
print(config_filename)
