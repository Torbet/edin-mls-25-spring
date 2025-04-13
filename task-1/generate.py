import numpy as np
import json

np.random.seed(0)


def generate(i=1, ratio=64):
  D = 2**i
  N = D * ratio
  K = 10

  A = np.random.randn(N, D)
  X = np.random.randn(D)

  A_filename = f'data/A_{i}.txt'
  X_filename = f'data/X_{i}.txt'
  config_filename = f'data/{i}.json'

  np.savetxt(A_filename, A)
  np.savetxt(X_filename, X)

  config = {'n': N, 'd': D, 'a_file': A_filename, 'x_file': X_filename, 'k': K}

  with open(config_filename, 'w') as f:
    json.dump(config, f, indent=2)

  print('Data files and configuration created.')
  print(config_filename)


def generate_independent(N, D):
  K = 10
  A = np.random.randn(N, D)
  X = np.random.randn(D)

  A_filename = f'data/A_{N}_{D}.txt'
  X_filename = f'data/X_{N}_{D}.txt'
  config_filename = f'data/{N}_{D}.json'

  np.savetxt(A_filename, A)
  np.savetxt(X_filename, X)

  config = {'n': N, 'd': D, 'a_file': A_filename, 'x_file': X_filename, 'k': K}

  with open(config_filename, 'w') as f:
    json.dump(config, f, indent=2)

  print('Data files and configuration created.')
  print(config_filename)


if __name__ == '__main__':
  for n in [4000, 40000, 400000]:
    for d in [512]:
      generate_independent(n, d)
