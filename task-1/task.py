import torch
import cupy as cp
import triton
import numpy as np
import time
import json
import csv
import pandas as pd
from test import testdata_kmeans, testdata_knn, testdata_ann

np.random.seed(0)
cp.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


import cupy as cp
import numpy as np
import torch

# --- Batched Distance Functions using CuPy (GPU) ---


def distance_cosine(X, Y, batch_size=1024):
  # X: (n, D), Y: (m, D)
  n = X.shape[0]
  result = cp.empty((n, Y.shape[0]), dtype=cp.float32)
  # Pre-compute normalization for Y (shared for all batches)
  Y_norm = cp.linalg.norm(Y, axis=1, keepdims=True)
  Y_normalized = Y / (Y_norm + 1e-8)
  for i in range(0, n, batch_size):
    X_batch = X[i : i + batch_size]
    X_norm = cp.linalg.norm(X_batch, axis=1, keepdims=True)
    X_normalized = X_batch / (X_norm + 1e-8)
    dot_product = cp.dot(X_normalized, Y_normalized.T)
    result[i : i + batch_size] = 1 - dot_product
  return result


def distance_l2(X, Y, batch_size=1024):
  # Computes Euclidean (L2) distances: sqrt(sum((x-y)^2, axis=2))
  n = X.shape[0]
  result = cp.empty((n, Y.shape[0]), dtype=cp.float32)
  for i in range(0, n, batch_size):
    X_batch = X[i : i + batch_size]
    result[i : i + batch_size] = cp.sqrt(cp.sum((X_batch[:, None] - Y) ** 2, axis=2))
  return result


def distance_dot(X, Y, batch_size=1024):
  # Computes dot product; note that if you want a distance you may need to adjust the value
  n = X.shape[0]
  result = cp.empty((n, Y.shape[0]), dtype=cp.float32)
  for i in range(0, n, batch_size):
    X_batch = X[i : i + batch_size]
    result[i : i + batch_size] = cp.dot(X_batch, Y.T)
  return result


def distance_manhattan(X, Y, batch_size=1024):
  # Computes Manhattan (L1) distance: sum(abs(x-y), axis=2)
  n = X.shape[0]
  result = cp.empty((n, Y.shape[0]), dtype=cp.float32)
  for i in range(0, n, batch_size):
    X_batch = X[i : i + batch_size]
    result[i : i + batch_size] = cp.sum(cp.abs(X_batch[:, None] - Y), axis=2)
  return result


# --- Batched Distance Functions using NumPy (CPU) ---


def distance_cosine_np(X, Y, batch_size=1024):
  n = X.shape[0]
  result = np.empty((n, Y.shape[0]), dtype=np.float32)
  Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
  Y_normalized = X / (Y_norm + 1e-8)  # This seems an inadvertent error, so see below:
  # --- Correction: We need to normalize Y, not X ---
  Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
  Y_normalized = Y / (Y_norm + 1e-8)
  for i in range(0, n, batch_size):
    X_batch = X[i : i + batch_size]
    X_norm = np.linalg.norm(X_batch, axis=1, keepdims=True)
    X_normalized = X_batch / (X_norm + 1e-8)
    dot_product = np.dot(X_normalized, Y_normalized.T)
    result[i : i + batch_size] = 1 - dot_product
  return result


def distance_l2_np(X, Y, batch_size=1024):
  n = X.shape[0]
  result = np.empty((n, Y.shape[0]), dtype=np.float32)
  for i in range(0, n, batch_size):
    X_batch = X[i : i + batch_size]
    result[i : i + batch_size] = np.sqrt(np.sum((X_batch[:, None] - Y) ** 2, axis=2))
  return result


def distance_dot_np(X, Y, batch_size=1024):
  n = X.shape[0]
  result = np.empty((n, Y.shape[0]), dtype=np.float32)
  for i in range(0, n, batch_size):
    X_batch = X[i : i + batch_size]
    result[i : i + batch_size] = np.dot(X_batch, Y.T)
  return result


def distance_manhattan_np(X, Y, batch_size=1024):
  n = X.shape[0]
  result = np.empty((n, Y.shape[0]), dtype=np.float32)
  for i in range(0, n, batch_size):
    X_batch = X[i : i + batch_size]
    result[i : i + batch_size] = np.sum(np.abs(X_batch[:, None] - Y), axis=2)
  return result


# --- Batched Distance Functions using Torch ---


def distance_cosine_torch(X, Y, batch_size=1024):
  # Assumes X, Y are torch tensors.
  n = X.shape[0]
  device = X.device
  m = Y.shape[0]
  result = torch.empty((n, m), device=device, dtype=torch.float32)
  Y_norm = torch.norm(Y, dim=1, keepdim=True)
  Y_normalized = Y / (Y_norm + 1e-8)
  for i in range(0, n, batch_size):
    X_batch = X[i : i + batch_size]
    X_norm = torch.norm(X_batch, dim=1, keepdim=True)
    X_normalized = X_batch / (X_norm + 1e-8)
    dot_product = torch.mm(X_normalized, Y_normalized.T)
    result[i : i + batch_size] = 1 - dot_product
  return result


def distance_l2_torch(X, Y, batch_size=1024):
  n = X.shape[0]
  m = Y.shape[0]
  device = X.device
  result = torch.empty((n, m), device=device, dtype=torch.float32)
  for i in range(0, n, batch_size):
    X_batch = X[i : i + batch_size]
    result[i : i + batch_size] = torch.sqrt(torch.sum((X_batch[:, None] - Y) ** 2, dim=2))
  return result


def distance_dot_torch(X, Y, batch_size=1024):
  n = X.shape[0]
  m = Y.shape[0]
  device = X.device
  result = torch.empty((n, m), device=device, dtype=torch.float32)
  for i in range(0, n, batch_size):
    X_batch = X[i : i + batch_size]
    result[i : i + batch_size] = torch.mm(X_batch, Y.T)
  return result


def distance_manhattan_torch(X, Y, batch_size=1024):
  n = X.shape[0]
  m = Y.shape[0]
  device = X.device
  result = torch.empty((n, m), device=device, dtype=torch.float32)
  for i in range(0, n, batch_size):
    X_batch = X[i : i + batch_size]
    result[i : i + batch_size] = torch.sum(torch.abs(X_batch[:, None] - Y), dim=2)
  return result


# --- Updating the dictionary to use batched versions ---

dists = {
  'l2': {'cpu': distance_l2_np, 'gpu': distance_l2, 'torch': distance_l2_torch},
  'cosine': {'cpu': distance_cosine_np, 'gpu': distance_cosine, 'torch': distance_cosine_torch},
  'dot': {'cpu': distance_dot_np, 'gpu': distance_dot, 'torch': distance_dot_torch},
  'manhattan': {'cpu': distance_manhattan_np, 'gpu': distance_manhattan, 'torch': distance_manhattan_torch},
}


topk_kernel = cp.RawKernel(
  r"""
extern "C" __global__
void topk_small(const float* distances, int* indices, int N, int K) {
    __shared__ float dist_shared[1024];
    __shared__ int index_shared[1024];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // load into shared memory
    if (i < N) {
        dist_shared[tid] = distances[i];
        index_shared[tid] = i;
    } else {
        dist_shared[tid] = 1e10;  // max distance
        index_shared[tid] = -1;
    }

    __syncthreads();

    // simple bitonic sort (only works well when blockDim.x = power of 2)
    for (int k = 2; k <= blockDim.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (dist_shared[tid] > dist_shared[ixj]) {
                        float tmpd = dist_shared[tid];
                        dist_shared[tid] = dist_shared[ixj];
                        dist_shared[ixj] = tmpd;

                        int tmpi = index_shared[tid];
                        index_shared[tid] = index_shared[ixj];
                        index_shared[ixj] = tmpi;
                    }
                } else {
                    if (dist_shared[tid] < dist_shared[ixj]) {
                        float tmpd = dist_shared[tid];
                        dist_shared[tid] = dist_shared[ixj];
                        dist_shared[ixj] = tmpd;

                        int tmpi = index_shared[tid];
                        index_shared[tid] = index_shared[ixj];
                        index_shared[ixj] = tmpi;
                    }
                }
            }
            __syncthreads();
        }
    }

    // only first K threads write output
    if (tid < K) {
        indices[blockIdx.x * K + tid] = index_shared[tid];
    }
}
""",
  'topk_small',
)


def kernel_knn(N, D, A, X, K, distance='l2'):
  distances = dists[distance]['gpu'](A, X).flatten()

  threads = 1024
  blocks = int(np.ceil(N / threads))

  output = cp.empty((blocks * K,), dtype=cp.int32)

  topk_kernel((blocks,), (threads,), (distances, output, N, K))

  cp.cuda.Stream.null.synchronize()

  partial_indices = output.reshape(blocks, K)
  candidate_distances = distances[partial_indices].flatten()
  candidate_indices = partial_indices.flatten()

  if candidate_distances.shape[0] > K:
    k_candidates = cp.argpartition(candidate_distances, K)[:K]
    sorted_candidates = k_candidates[cp.argsort(candidate_distances[k_candidates])]
    final_indices = candidate_indices[sorted_candidates]
  else:
    final_indices = candidate_indices

  return final_indices


def cp_knn(N, D, A, X, K, distance='l2'):
  X = X.reshape(1, -1)

  distances = dists[distance]['gpu'](A, X).flatten()
  k = cp.argpartition(distances, K)[:K]
  indices = k[cp.argsort(distances[k])]

  return indices


def np_knn(N, D, A, X, K, distance='l2'):
  X = X.reshape(1, -1)

  distances = dists[distance]['cpu'](A, X).flatten()
  k = np.argpartition(distances, K)[:K]
  indices = k[np.argsort(distances[k])]

  return indices


def torch_knn(N, D, A, X, K, distance='l2'):
  X = X.reshape(1, -1)

  distances = dists[distance]['torch'](A, X).flatten()
  _, indices = torch.topk(distances, K, largest=False)

  return indices


def our_kmeans(N, D, A, K, distance='l2', batch_size=1024):
  """
  Implements K-Means clustering entirely in CuPy with batching.

  Parameters:
    N (int): Number of data points.
    D (int): Dimension of each data point.
    A: Array-like (N x D) of data points.
    K (int): Number of clusters.
    distance (str): Currently only "l2" is supported.
    batch_size (int): Size of batches to compute distances.

  Returns:
    assignments (cp.ndarray): Cluster ID (0 <= id < K) for each data point.
    centroids (cp.ndarray): Final cluster centroids (K x D).
  """
  max_iter = 100
  tol = 1e-4

  init_indices = np.random.choice(N, K, replace=False)
  centroids = A[init_indices]

  for it in range(max_iter):
    assignments = cp.empty((N,), dtype=cp.int32)

    for i in range(0, N, batch_size):
      A_batch = A[i : i + batch_size]

      dists = cp.sum((A_batch[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
      assignments[i : i + batch_size] = cp.argmin(dists, axis=1)

    new_centroids = []
    for k in range(K):
      mask = assignments == k
      if cp.sum(mask) == 0:
        new_centroids.append(centroids[k])
      else:
        new_centroids.append(cp.mean(A[mask], axis=0))
    new_centroids = cp.stack(new_centroids, axis=0)

    shift = cp.linalg.norm(new_centroids - centroids)
    centroids = new_centroids
    if shift < tol:
      break

  return assignments, centroids


def our_ann(N, D, A, X, K, distance='l2', assignments=None, centroids=None):
  """
  1. Cluster A into K clusters using KMeans.
  2. For the query X, find the nearest K1 cluster centers.
  3. For each selected cluster, retrieve up to K2 candidate neighbors.
  4. Merge candidates and use cp_knn (or cp.argsort when needed) to select the final top K neighbors.

  Parameters:
    N (int): Number of data points.
    D (int): Dimensionality of each data point.
    A (cp.ndarray): Dataset (N x D).
    X (cp.ndarray): Query vector.
    K (int): Used for both the number of clusters in KMeans and the number of final neighbors.
    distance (str): Distance metric key ('l2', 'dot', etc.).
    assignments (cp.ndarray, optional): Pre-computed cluster labels for A.
    centroids (cp.ndarray, optional): Pre-computed cluster centers.

  Returns:
    cp.ndarray: Global indices (into A) of the final top K nearest neighbors.
  """

  if assignments is None or centroids is None:
    assignments, centroids = our_kmeans(N, D, A, K, distance)

  K1 = min(5, centroids.shape[0])
  K2 = 10

  cluster_center_indices = kernel_knn(centroids.shape[0], D, centroids, X, K1, distance)

  candidate_indices_list = []

  for cid in cluster_center_indices:
    cluster_mask = assignments == cid
    member_indices = cp.nonzero(cluster_mask)[0]
    if member_indices.size == 0:
      continue

    A_subset = A[member_indices]
    M = A_subset.shape[0]

    current_K2 = min(K2, M)
    if M <= current_K2:
      dists_subset = dists[distance]['gpu'](A_subset, X).flatten()
      candidate_local_indices = cp.argsort(dists_subset)
    else:
      candidate_local_indices = kernel_knn(M, D, A_subset, X, current_K2, distance)

    candidates_global = member_indices[candidate_local_indices]
    candidate_indices_list.append(candidates_global)

  if len(candidate_indices_list) == 0:
    return cp.empty((0,), dtype=cp.int32)

  candidate_indices = cp.concatenate(candidate_indices_list)
  candidates = A[candidate_indices]

  if candidates.shape[0] <= K:
    final_order = cp.argsort(dists[distance]['gpu'](candidates, X).flatten())
  else:
    final_order = kernel_knn(candidates.shape[0], D, candidates, X, K, distance)
  final_global_indices = candidate_indices[final_order]

  return final_global_indices


def timeit(func, *args, **kwargs):
  its = 10
  times = []
  for i in range(its):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    times.append(end - start)
  t = np.mean(times)
  return t, result


def test_kmeans(distance='l2'):
  N, D, A, K = testdata_kmeans('')
  start = time.time()
  kmeans_result = our_kmeans(N, D, A, K)
  end = time.time()
  print('K Means Time taken:', end - start)


def test_knn(n, d, distance='l2'):
  print('Testing KNN with n:', n, 'd:', d, 'distance:', distance)
  N, D, A, X, K = testdata_knn(f'data/{n}_{d}.json')

  results = {}

  print()

  print('N:', N, 'D:', D, 'K:', K)

  print()

  a, x = np.asarray(A), np.asarray(X)
  t, result = timeit(np_knn, N, D, a, x, K, distance)
  results['numpy'] = t
  print('KNN Time taken (Numpy CPU):', t)
  print('KNN Result (Numpy CPU):', result)

  print()

  cp.get_default_memory_pool().free_all_blocks()
  cp.get_default_pinned_memory_pool().free_all_blocks()
  cp.cuda.Stream.null.synchronize()

  a, x = cp.asarray(A), cp.asarray(X)
  t, knn_result = timeit(cp_knn, N, D, a, x, K, distance)
  results['cupy'] = t
  print('KNN Time taken (Cupy GPU):', t)
  print('KNN Result (Cupy GPU):', result)

  print()

  cp.get_default_memory_pool().free_all_blocks()
  cp.get_default_pinned_memory_pool().free_all_blocks()
  cp.cuda.Stream.null.synchronize()

  a, x = cp.asarray(A, dtype=cp.float32), cp.asarray(X, dtype=cp.float32).reshape(1, -1)
  t, result = timeit(kernel_knn, N, D, a, x, K, distance)
  results['kernel'] = t
  print('KNN Time taken (Kernel GPU):', t)
  print('KNN Result (Kernel GPU):', result)

  print()

  cp.get_default_memory_pool().free_all_blocks()
  cp.get_default_pinned_memory_pool().free_all_blocks()
  cp.cuda.Stream.null.synchronize()

  a, x = cp.asarray(A, dtype=cp.float32), cp.asarray(X, dtype=cp.float32).reshape(1, -1)
  num_clusters = 100 if N > 100 else N
  assignments, centroids = our_kmeans(N, D, a, num_clusters, distance=distance)
  t, ann_result = timeit(our_ann, N, D, a, x, K, distance, assignments=assignments, centroids=centroids)
  results['ann'] = t
  print('ANN Time taken:', t)
  print('ANN Result:', result)

  print()

  cp.get_default_memory_pool().free_all_blocks()
  cp.get_default_pinned_memory_pool().free_all_blocks()
  cp.cuda.Stream.null.synchronize()

  ann_result = cp.asnumpy(ann_result)
  knn_result = cp.asnumpy(knn_result)
  rr = recall_rate(ann_result, knn_result)
  print('Recall rate:', rr)
  results['recall_rate'] = rr

  print()

  cp.get_default_memory_pool().free_all_blocks()
  cp.get_default_pinned_memory_pool().free_all_blocks()
  cp.cuda.Stream.null.synchronize()

  torch.cuda.empty_cache()
  torch.cuda.synchronize()

  a, x = torch.tensor(A).to('cpu'), torch.tensor(X).to('cpu')
  t, result = timeit(torch_knn, N, D, a, x, K, distance)
  results['torch_cpu'] = t
  print('KNN Time taken (Torch CPU):', t)
  print('KNN Result (torch CPU):', result.tolist())

  print()

  torch.cuda.empty_cache()
  torch.cuda.synchronize()

  a, x = torch.tensor(A).to('cuda'), torch.tensor(X).to('cuda')
  t, result = timeit(torch_knn, N, D, a, x, K, distance)
  results['torch_gpu'] = t
  print('KNN Time taken (Torch GPU):', t)
  print('KNN Result (torch GPU):', result.tolist())

  print()

  return results


def test_ann():
  N, D, A, X, K = testdata_ann('')
  start = time.time()
  ann_result = our_ann(N, D, A, X, K)
  end = time.time()
  print('ANN Time taken:', end - start)


def test_recall_rate(i):
  N, D, A, X, K = testdata_ann(f'data/{i}.json')
  A, X = cp.asarray(A), cp.asarray(X)
  ann_result = our_ann(N, D, A, X, K)
  knn_result = cp_knn(N, D, A, X, K)

  ann_result = cp.asnumpy(ann_result)
  knn_result = cp.asnumpy(knn_result)
  print('Recall rate:', recall_rate(ann_result, knn_result))


def recall_rate(list1, list2):
  """
  Calculate the recall rate of two lists
  list1[K]: The top K nearest vectors ID
  list2[K]: The top K nearest vectors ID
  """
  return len(set(sorted(list1)) & set(sorted(list2))) / len(list1)


if __name__ == '__main__':
  # distance = 'dot'

  for distance in ['dot']:
    results = {}
    for n in [4000, 40000, 400000]:
      for d in [2, 512, 1024]:
        r = test_knn(n, d, distance)
        results[f'{n}_{d}'] = r

    print('Results:', results)
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = 'i'
    df = df.reset_index()
    df.to_csv(f'results_independent_{distance}.csv', index=True, header=True)
    print(f'Results saved to results_independent_{distance}.csv')
