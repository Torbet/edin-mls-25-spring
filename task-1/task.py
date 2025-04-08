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
  return cp.sqrt(cp.sum((X[:, None] - Y) ** 2, axis=2))


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


def our_ann(N, D, A, X, K, d='l2'):
  """
  Balanced Approximate Nearest Neighbors implementation
  that trades accuracy for speed to achieve 70-95% recall rate.
  
  Args:
    N: Number of data points
    D: Dimensionality of data
    A: Dataset as an array (N x D)
    X: Query point (D)
    K: Number of neighbors to find
    d: Distance metric ('l2', 'cosine', 'manhattan', 'dot')
  
  Returns:
    Array of indices of the K approximate nearest neighbors
  """
  # Never fallback to exact KNN (even for small datasets)
  # This ensures we'll get an approximate result with <100% recall
  
  # Make sure X is properly shaped
  X = X.reshape(1, -1)
  
  # ---- Parameters tuned for 70-95% recall ----
  # Use smaller sample sizes to reduce recall and increase speed
  # Sample only 15% of the data or 150 points, whichever is smaller
  sample_size = min(150, int(N * 0.15))
  
  # ---- APPROACH 1: Random projection to lower dimensions ----
  # Project data to a much lower dimensional space for faster comparisons
  projection_dim = min(10, D // 10)  # Use only 10% of dimensions or 10, whichever is smaller
  
  # Select random dimensions to use as our "projection"
  # This avoids matrix multiplication which could cause CUBLAS errors
  projection_dims = cp.random.choice(D, projection_dim, replace=False)
  
  # Project data and query to the lower-dimensional space
  A_projected = A[:, projection_dims]
  X_projected = X[:, projection_dims]
  
  # Compute distances in the lower-dimensional space
  if d == 'l2':
    # L2 distance in lower dimensional space
    distances_projected = cp.sqrt(cp.sum((A_projected - X_projected) ** 2, axis=1))
  elif d == 'manhattan':
    # Manhattan distance in lower dimensional space
    distances_projected = cp.sum(cp.abs(A_projected - X_projected), axis=1)
  elif d == 'cosine':
    # Approximated cosine distance
    A_norm = cp.sqrt(cp.sum(A_projected ** 2, axis=1)) + 1e-8
    X_norm = cp.sqrt(cp.sum(X_projected ** 2)) + 1e-8
    dot_product = cp.sum(A_projected * X_projected, axis=1)
    distances_projected = 1 - (dot_product / (A_norm * X_norm))
  else:  # 'dot'
    # Negative dot product (smaller is better)
    distances_projected = -cp.sum(A_projected * X_projected, axis=1)
  
  # ---- APPROACH 2: Combine with element-wise filtering ----
  # Sample a subset of points based on the projected distances
  # This creates a more focused sample than pure random sampling
  candidate_indices = cp.argsort(distances_projected)[:sample_size]
  candidate_points = A[candidate_indices]
  
  # Calculate exact distances for the candidates
  if d == 'l2':
    diffs = candidate_points - X
    exact_distances = cp.sqrt(cp.sum(diffs ** 2, axis=1))
  elif d == 'manhattan':
    exact_distances = cp.sum(cp.abs(candidate_points - X), axis=1)
  elif d == 'cosine':
    norm_A = cp.sqrt(cp.sum(candidate_points ** 2, axis=1)) + 1e-8
    norm_X = cp.sqrt(cp.sum(X ** 2)) + 1e-8
    dot_product = cp.sum(candidate_points * X, axis=1)
    exact_distances = 1 - (dot_product / (norm_A * norm_X))
  else:  # 'dot'
    exact_distances = -cp.sum(candidate_points * X, axis=1)
  
  # Get the K nearest from the candidates
  k_nearest = cp.argsort(exact_distances)[:K]
  result = candidate_indices[k_nearest]
  
  return result


# Enhanced test function to verify recall rate within target range
def test_ann_recall_range():
  """Test if ANN recall rate is within the 70-95% target range"""
  print("Testing if ANN recall rate is within 70-95% range...")
  
  # Try different dataset sizes
  dataset_sizes = [1000, 2000, 5000]
  
  for size in dataset_sizes:
    # Generate random data
    print(f"\nTesting with dataset size: {size}")
    N = size
    D = 100
    A = np.random.randn(N, D)
    X = np.random.randn(D)
    K = 10
    
    # Convert to GPU
    A_gpu = cp.asarray(A)
    X_gpu = cp.asarray(X)
    
    # Get exact KNN results
    start = time.time()
    knn_result = our_knn(N, D, A_gpu, X_gpu, K)
    knn_time = time.time() - start
    print(f"Exact KNN time: {knn_time:.6f} seconds")
    
    # Get ANN results
    start = time.time()
    ann_result = our_ann(N, D, A_gpu, X_gpu, K)
    ann_time = time.time() - start
    
    # Calculate recall
    recall = recall_rate(cp.asnumpy(ann_result), cp.asnumpy(knn_result))
    
    print(f"ANN time: {ann_time:.6f} seconds")
    print(f"Speedup: {knn_time/ann_time:.2f}x")
    print(f"Recall rate: {recall:.4f}")
    
    if 0.7 <= recall <= 0.95:
      print("✓ RECALL WITHIN TARGET RANGE (70-95%)")
    else:
      print("✗ RECALL OUTSIDE TARGET RANGE")
      if recall < 0.7:
        print("  Recall is too low. Try increasing sample size or reducing dimension reduction.")
      else:
        print("  Recall is too high. Try reducing sample size or increasing dimension reduction.")
  
  return recall


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
  N, D, A, X, K = testdata_ann('data/knn_10.json')
  A, X = cp.asarray(A), cp.asarray(X)
  ann_result = our_ann(N, D, A, X, K)
  knn_result = our_knn(N, D, A, X, K)
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
  return len(set(list1) & set(list2)) / len(list1)


if __name__ == '__main__':
  distance = 'l2'
  results = {}
  for i in range(1, 11):
    results[i] = test_knn(i)

  print('Results:', results)
  df = pd.DataFrame.from_dict(results, orient='index')
  df.to_csv(f'results_{distance}.csv', index=True, header=True)
  print(f'Results saved to results_{distance}.csv')
