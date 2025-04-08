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


def distance_cosine(X, Y):
  X_norm = cp.linalg.norm(X, axis=1, keepdims=True) + 1e-8
  Y_norm = cp.linalg.norm(Y, axis=1, keepdims=True) + 1e-8
  dot_product = cp.dot(X, Y.T)
  cosine_similarity = dot_product / (X_norm * Y_norm.T)
  return 1 - cosine_similarity


def distance_cosine_np(X, Y):
  X_norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
  Y_norm = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8
  dot_product = np.dot(X, Y.T)
  cosine_similarity = dot_product / (X_norm * Y_norm.T)
  return 1 - cosine_similarity


def distance_cosine_torch(X, Y):
  X_norm = torch.norm(X, dim=1, keepdim=True) + 1e-8
  Y_norm = torch.norm(Y, dim=1, keepdim=True) + 1e-8
  dot_product = torch.mm(X, Y.T)
  cosine_similarity = dot_product / (X_norm * Y_norm.T)
  return 1 - cosine_similarity


def distance_l2(X, Y):
  X_sq = cp.sum(X**2, axis=1, keepdims=True)
  Y_sq = cp.sum(Y**2, axis=1, keepdims=True)
  XY = cp.dot(X, Y.T)
  return cp.sqrt(X_sq + Y_sq.T - 2 * XY)


def distance_l2_np(X, Y):
  X_sq = np.sum(X**2, axis=1, keepdims=True)
  Y_sq = np.sum(Y**2, axis=1, keepdims=True)
  XY = np.dot(X, Y.T)
  return np.sqrt(X_sq + Y_sq.T - 2 * XY)


def distance_l2_torch(X, Y):
  X_sq = torch.sum(X**2, dim=1, keepdim=True)
  Y_sq = torch.sum(Y**2, dim=1, keepdim=True)
  XY = torch.mm(X, Y.T)
  return torch.sqrt(X_sq + Y_sq.T - 2 * XY)


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


def our_knn_topk_kernel(N, D, A, X, K, distance='l2'):
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


def our_knn(N, D, A, X, K, distance='l2'):
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


def our_kmeans(N, D, A, K, distance=distance_l2):
  A = cp.asarray(A)
  centroids = A[cp.random.choice(N, K, replace=False)]

  for _ in range(100):  # max iters
    distances = distance(A, centroids)
    labels = cp.argmin(distances, axis=1)

    # new_centroids = cp.array([A[labels == k].mean(axis=0) if cp.any(labels == k) else centroids[k] for k in range(K)])
    new_centroids = cp.array([cp.mean(A[labels == k], axis=0) for k in range(K)])

    if cp.allclose(centroids, new_centroids):
      break

    centroids = new_centroids

  return labels


# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here


def our_ann(N, D, A, X, K, distance=distance_l2):
  A = cp.asarray(A)
  X = cp.asarray(X).reshape(1, -1)

  K1 = min(20, N // 50 + 1)  # Number of clusters
  K2 = min(50, N // 10 + 1)  # Number of candidates per cluster

  # 1. Use KMeans to cluster the data into K1 clusters
  cluster_labels = our_kmeans(N, D, A, K1, distance)

  centroids = cp.zeros((K1, D))
  for k in range(K1):
    if cp.any(cluster_labels == k):
      centroids[k] = cp.mean(A[cluster_labels == k], axis=0)

  # 2. Find the nearest K1 cluster centers to the query point
  distances_to_centroids = distance(centroids, X).flatten()
  nearest_clusters = cp.argsort(distances_to_centroids)[:K1]

  # 3. For each of the K1 clusters, find K2 nearest neighbors
  candidate_indices = []
  for cluster_idx in nearest_clusters:
    cluster_points_indices = cp.where(cluster_labels == cluster_idx)[0]

    if len(cluster_points_indices) > 0:
      cluster_points = A[cluster_points_indices]
      distances = distance(cluster_points, X).flatten()

      k2_actual = min(K2, len(cluster_points_indices))
      nearest_indices = cp.argsort(distances)[:k2_actual]
      candidate_indices.append(cluster_points_indices[nearest_indices])

  # 4. Merge candidates from all clusters and find overall top K
  if candidate_indices:
    all_candidates = cp.concatenate(candidate_indices)
    all_candidates_points = A[all_candidates]

    distances = distance(all_candidates_points, X).flatten()

    k_actual = min(K, len(all_candidates))
    top_k_indices = cp.argsort(distances)[:k_actual]

    return all_candidates[top_k_indices]

  return our_knn(N, D, A, X, K)


# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------


# Example


def test_kmeans(distance='l2'):
  N, D, A, K = testdata_kmeans('')
  start = time.time()
  kmeans_result = our_kmeans(N, D, A, K)
  end = time.time()
  print('K Means Time taken:', end - start)


def test_knn(i=1, distance='l2'):
  N, D, A, X, K = testdata_knn(f'data/knn_{i}.json')

  results = {}

  print()

  print('N:', N, 'D:', D, 'K:', K)

  print()

  its = 10

  a, x = np.asarray(A), np.asarray(X)
  times = []
  for i in range(its):
    start = time.time()
    result = np_knn(N, D, a, x, K, distance)
    end = time.time()
    times.append(end - start)
  t = np.mean(times)
  results['numpy'] = t
  print('KNN Time taken (Numpy CPU):', t)
  print('KNN Result (Numpy CPU):', result)

  print()

  a, x = cp.asarray(A), cp.asarray(X)
  times = []
  for i in range(its):
    start = time.time()
    result = our_knn(N, D, a, x, K, distance)
    end = time.time()
    times.append(end - start)
  t = np.mean(times)
  results['cupy'] = t
  print('KNN Time taken (Cupy GPU):', t)
  print('KNN Result (Cupy GPU):', result)

  print()

  a = cp.asarray(A, dtype=cp.float32)
  x = cp.asarray(X, dtype=cp.float32).reshape(1, -1)
  times = []
  for i in range(its):
    start = time.time()
    result = our_knn_topk_kernel(N, D, a, x, K, distance)
    end = time.time()
    times.append(end - start)
  t = np.mean(times)
  results['cupy_kernel'] = t
  print('KNN Time taken (Cupy Kernel GPU):', t)
  print('KNN Result (Cupy Kernel GPU):', result)

  print()

  a, x = torch.tensor(A).to('cpu'), torch.tensor(X).to('cpu')
  times = []
  for i in range(its):
    start = time.time()
    result = torch_knn(N, D, a, x, K, distance)
    end = time.time()
    times.append(end - start)
  t = np.mean(times)
  results['torch_cpu'] = t
  print('KNN Time taken (Torch CPU):', t)
  print('KNN Result (torch CPU):', result.tolist())

  print()

  a, x = torch.tensor(A).to('cuda'), torch.tensor(X).to('cuda')
  times = []
  for i in range(its):
    start = time.time()
    result = torch_knn(N, D, a, x, K, distance)
    end = time.time()
    times.append(end - start)
  t = np.mean(times)
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


def test_recall_rate():
  N, D, A, X, K = testdata_ann('')
  ann_result = our_ann(N, D, A, X, K)
  knn_result = our_knn(N, D, A, X, K)
  print('Recall rate:', recall_rate(ann_result, knn_result))


def recall_rate(list1, list2):
  """
  Calculate the recall rate of two lists
  list1[K]: The top K nearest vectors ID
  list2[K]: The top K nearest vectors ID
  """
  return len(set(list1) & set(list2)) / len(list1)


if __name__ == '__main__':
  distance = 'manhattan'
  results = {}
  for i in range(1, 11):
    results[i] = test_knn(i)

  print('Results:', results)
  df = pd.DataFrame.from_dict(results, orient='index')
  df.to_csv(f'results_{distance}.csv', index=True, header=True)
  print(f'Results saved to results_{distance}.csv')
