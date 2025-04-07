import torch
import cupy as cp
import triton
import numpy as np
import time
import json
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


def distance_dot(X, Y):
  return cp.dot(X, Y.T)


def distance_manhattan(X, Y):
  return cp.sum(cp.abs(X[:, None] - Y), axis=2)


# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here


def our_knn(N, D, A, X, K, distance=distance_l2):
  X = X.reshape(1, -1)

  distances = distance(A, X).flatten()
  k = cp.argpartition(distances, K)[:K]
  indices = k[cp.argsort(distances[k])]

  return indices


def np_knn(N, D, A, X, K, distance=distance_l2_np):
  X = X.reshape(1, -1)

  distances = distance(A, X).flatten()
  indices = np.argsort(distances)[:K]

  return indices


def torch_knn(N, D, A, X, K):
  X = X.reshape(1, -1)

  distances = torch.cdist(A, X).flatten()
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

  for _ in range(100):
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


def test_kmeans():
  N, D, A, K = testdata_kmeans('')
  start = time.time()
  kmeans_result = our_kmeans(N, D, A, K)
  end = time.time()
  print('K Means Time taken:', end - start)


def test_knn():
  N, D, A, X, K = testdata_knn('data/test_config_200000x1024.json')

  print('N:', N, 'D:', D, 'K:', K)
  print('A:', A.shape, 'X:', X.shape)

  a, x = np.asarray(A), np.asarray(X)
  start = time.time()
  result = np_knn(N, D, a, x, K)
  end = time.time()
  print('KNN Time taken (Numpy CPU):', end - start)
  print('KNN Result (Numpy CPU):', result)

  a, x = cp.asarray(A), cp.asarray(X)
  start = time.time()
  result = our_knn(N, D, a, x, K)
  end = time.time()
  print('KNN Time taken (Cupy GPU):', end - start)
  print('KNN Result (Cupy GPU):', result)

  a, x = torch.tensor(A).to('cpu'), torch.tensor(X).to('cpu')
  start = time.time()
  result = torch_knn(N, D, a, x, K)
  end = time.time()
  print('KNN Time taken (torch CPU):', end - start)
  print('KNN Result (torch CPU):', result.tolist())

  a, x = torch.tensor(A).to('cuda'), torch.tensor(X).to('cuda')
  start = time.time()
  result = torch_knn(N, D, a, x, K)
  end = time.time()
  print('KNN Time taken (torch GPU):', end - start)
  print('KNN Result (torch GPU):', result.tolist())


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
  test_knn()
