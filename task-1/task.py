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

# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass


# normalized (cp.linalg.norm)
def distance_cosine(X, Y):
  X_norm = cp.linalg.norm(X, axis=1, keepdims=True)
  Y_norm = cp.linalg.norm(Y, axis=1, keepdims=True)
  X_normalized = X / (X_norm + 1e-8)
  Y_normalized = Y / (Y_norm + 1e-8)
  dot_product = cp.dot(X_normalized, Y_normalized.T)
  return 1 - dot_product


def distance_cosine_np(X, Y):
  X_norm = np.linalg.norm(X, axis=1, keepdims=True)
  Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
  X_normalized = X / (X_norm + 1e-8)
  Y_normalized = Y / (Y_norm + 1e-8)
  dot_product = np.dot(X_normalized, Y_normalized.T)
  return 1 - dot_product


def distance_cosine_torch(X, Y):
  X_norm = torch.norm(X, dim=1, keepdim=True)
  Y_norm = torch.norm(Y, dim=1, keepdim=True)
  X_normalized = X / (X_norm + 1e-8)
  Y_normalized = Y / (Y_norm + 1e-8)
  dot_product = torch.mm(X_normalized, Y_normalized.T)
  return 1 - dot_product


def distance_l2(X, Y):
  return cp.sqrt(cp.sum((X[:, None] - Y) ** 2, axis=2))


def distance_l2_batch(A, X, batch_size=1024):
  N, D = A.shape
  result = cp.empty((N, 1), dtype=cp.float32)
  for i in range(0, N, batch_size):
    A_batch = A[i : i + batch_size]
    result[i : i + batch_size] = cp.sqrt(cp.sum((A_batch[:, None] - X) ** 2, axis=2))
  return result


def distance_l2_np(X, Y):
  return np.sqrt(np.sum((X[:, None] - Y) ** 2, axis=2))


def distance_l2_torch(X, Y):
  return torch.sqrt(torch.sum((X[:, None] - Y) ** 2, dim=2))


def distance_dot(X, Y):
  return cp.dot(X, Y.T)


def distance_dot_np(X, Y):
  return np.dot(X, Y.T)


def distance_dot_torch(X, Y):
  return torch.mm(X, Y.T)


def distance_manhattan(X, Y):
  return cp.sum(cp.abs(X[:, None] - Y), axis=2)


def distance_manhattan_np(X, Y):
  return np.sum(np.abs(X[:, None] - Y), axis=2)


def distance_manhattan_torch(X, Y):
  return torch.sum(torch.abs(X[:, None] - Y), dim=2)


dists = {
  'l2': {'cpu': distance_l2_np, 'gpu': distance_l2, 'torch': distance_l2_torch},
  'l2_batch': {'cpu': distance_l2_np, 'gpu': distance_l2_batch, 'torch': distance_l2_torch},
  'cosine': {'cpu': distance_cosine_np, 'gpu': distance_cosine, 'torch': distance_cosine_torch},
  'dot': {'cpu': distance_dot_np, 'gpu': distance_dot, 'torch': distance_dot_torch},
  'manhattan': {'cpu': distance_manhattan_np, 'gpu': distance_manhattan, 'torch': distance_manhattan_torch},
}


# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

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


# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass


def our_kmeans(N, D, A, K, distance='l2'):
  """
  Implements K-Means clustering entirely in CuPy.

  Parameters:
    N (int): Number of data points.
    D (int): Dimension of each data point.
    A: Array-like (N x D) of data points.
    K (int): Number of clusters.
    distance (str): Currently only "l2" is supported.

  Returns:
    assignments (cp.ndarray): Cluster ID (0 <= id < K) for each data point.
    centroids (cp.ndarray): Final cluster centroids (K x D).
  """
  max_iter = 100
  tol = 1e-4

  A_gpu = cp.asarray(A, dtype=cp.float32)

  init_indices = np.random.choice(N, K, replace=False)
  centroids = A_gpu[init_indices]

  for it in range(max_iter):
    distances = cp.sum((A_gpu[:, None, :] - centroids[None, :, :]) ** 2, axis=2)

    assignments = cp.argmin(distances, axis=1)

    new_centroids = []
    for k in range(K):
      mask = assignments == k
      if cp.sum(mask) == 0:
        new_centroids.append(centroids[k])
      else:
        new_centroids.append(cp.mean(A_gpu[mask], axis=0))
    new_centroids = cp.stack(new_centroids, axis=0)

    shift = cp.linalg.norm(new_centroids - centroids)
    centroids = new_centroids
    if shift < tol:
      break

  return assignments, centroids


# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here


def our_ann(N, D, A, X, K, distance='l2', assignments=None, centroids=None):
  """
  Approximate Nearest Neighbor (ANN) search using K-Means.

  Steps:
    1. Cluster the database A into a fixed number of clusters.
    2. For a query vector X, compute its distance to each centroid.
    3. Select a small subset (K1) of clusters with the closest centroids.
    4. Restrict the search to all points in the selected clusters.
    5. Compute exact distances from X to all candidate points and return the top K neighbors.

  Parameters:
    N (int): Number of data points in A.
    D (int): Dimension of each data point.
    A: Array-like (N x D) of data points.
    X: Array-like (D,) query vector.
    K (int): Number of nearest neighbors to return.
    distance (str): Currently supports "l2".

  Returns:
    cp.ndarray: Indices of the top K nearest data points (using CuPy array).
  """

  num_clusters = 100 if N > 100 else N

  if assignments is None and centroids is None:
    assignments, centroids = our_kmeans(N, D, A, num_clusters, distance=distance)

  X_gpu = cp.asarray(X, dtype=cp.float32)

  centroid_dists = cp.linalg.norm(centroids - X_gpu, axis=1)

  K1 = max(1, num_clusters // 10)
  top_centroid_indices = cp.argpartition(centroid_dists, K1)[:K1]

  candidate_mask = cp.isin(assignments, top_centroid_indices)
  candidate_indices = cp.nonzero(candidate_mask)[0]

  if candidate_indices.size == 0:
    return cp.array([], dtype=cp.int32)

  A_gpu = cp.asarray(A, dtype=cp.float32)
  candidates = A_gpu[candidate_indices]

  candidate_dists = cp.linalg.norm(candidates - X_gpu, axis=1)

  if candidate_dists.shape[0] > K:
    topk_local = cp.argpartition(candidate_dists, K)[:K]
    sorted_local = topk_local[cp.argsort(candidate_dists[topk_local])]
  else:
    sorted_local = cp.argsort(candidate_dists)

  final_indices = candidate_indices[sorted_local]
  return final_indices


# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------


# Example


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


def test_knn(i=1, distance='l2'):
  N, D, A, X, K = testdata_knn(f'data/{i}.json')

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
  t, result = timeit(cp_knn, N, D, a, x, K, distance)
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
  assignments, centroids = our_kmeans(N, D, A, num_clusters, distance=distance)
  t, result = timeit(our_ann, N, D, a, x, K, distance, assignments=assignments, centroids=centroids)
  results['ann'] = t
  print('ANN Time taken:', t)
  print('ANN Result:', result)

  print()

  cp.get_default_memory_pool().free_all_blocks()
  cp.get_default_pinned_memory_pool().free_all_blocks()
  cp.cuda.Stream.null.synchronize()

  test_recall_rate(i)

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
  # to numpy
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
  distance = 'l2'
  results = {}
  for i in range(1, 11):
    results[i] = test_knn(i)

  print('Results:', results)
  df = pd.DataFrame.from_dict(results, orient='index')
  df.to_csv(f'results_{distance}.csv', index=True, header=True)
  print(f'Results saved to results_{distance}.csv')
