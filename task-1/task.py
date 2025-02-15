import torch
import cupy as cp
import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
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


def distance_l2(X, Y):
  X_sq = cp.sum(X**2, axis=1, keepdims=True)
  Y_sq = cp.sum(Y**2, axis=1, keepdims=True)
  XY = cp.dot(X, Y.T)
  return cp.sqrt(X_sq + Y_sq.T - 2 * XY)


def distance_dot(X, Y):
  return cp.dot(X, Y.T)


def distance_manhattan(X, Y):
  return cp.sum(cp.abs(X[:, None] - Y), axis=2)


# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here


def our_knn(N, D, A, X, K):
  A = cp.asarray(A)
  X = cp.asarray(X).reshape(1, -1)

  distances = distance_l2(A, X).flatten()
  indices = cp.argsort(distances)[:K]

  return indices


# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass


def our_kmeans(N, D, A, K):
  pass


# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here


def our_ann(N, D, A, X, K):
  pass


# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------


# Example
def test_kmeans():
  N, D, A, K = testdata_kmeans('')
  kmeans_result = our_kmeans(N, D, A, K)
  print(kmeans_result)


def test_knn():
  N, D, A, X, K = testdata_knn('')
  knn_result = our_knn(N, D, A, X, K)
  print(knn_result)


def test_ann():
  N, D, A, X, K = testdata_ann('')
  ann_result = our_ann(N, D, A, X, K)
  print(ann_result)


def recall_rate(list1, list2):
  """
  Calculate the recall rate of two lists
  list1[K]: The top K nearest vectors ID
  list2[K]: The top K nearest vectors ID
  """
  return len(set(list1) & set(list2)) / len(list1)


if __name__ == '__main__':
  test_kmeans()
  test_knn()
  test_ann()
