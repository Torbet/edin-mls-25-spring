import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import uuid
import time
import threading
import queue
import os
from torch.cuda.amp import autocast
from request_queue import RequestQueue
import argparse


app = FastAPI()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Start the RAG service with customizable max batch size.')
parser.add_argument('--max_batch_size', type=int, default=20, help='Maximum batch size for request processing')
args = parser.parse_args()

# Assign the parsed batch size
MAX_BATCH_SIZE = args.max_batch_size
MAX_WAIT_TIME = 0.5

request_queue = RequestQueue()
response_queues = {}

# Example documents in memory
documents = [
  'Cats are small furry carnivores that are often kept as pets.',
  'Dogs are domesticated mammals, not natural wild animals.',
  'Hummingbirds can hover in mid-air by rapidly flapping their wings.',
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load embedding model
print('Loading embedding model name')
EMBED_MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'
print('Loading embed_tokenizer')
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
print('Loading embedding model...')
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)
print('Embedding model loaded.')

# Basic Chat LLM
chat_pipeline = pipeline('text-generation', model='facebook/opt-125m', device=0 if torch.cuda.is_available() else -1)


def get_embedding_batch(texts: list[str]) -> np.ndarray:
  # Tokenize the texts and send to the appropriate device (GPU or CPU)
  inputs = embed_tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to('cuda')

  # Use torch.no_grad to avoid tracking gradients - saves memory and computation
  with torch.no_grad():
    with torch.amp.autocast('cuda'):
      outputs = embed_model(**inputs)

  # Use more efficient mean calculation along dimension 1
  embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

  # Ensure embeddings are normalized for more accurate dot product similarity
  # Normalize across the embedding dimension (axis=1)
  norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
  normalized_embeddings = embeddings / norms

  return normalized_embeddings


# Precompute document embeddings and store on GPU
doc_embeddings = get_embedding_batch(documents)
# Convert to contiguous tensor and keep on GPU
doc_embeddings_tensor = torch.tensor(doc_embeddings, dtype=torch.float16).to('cuda')


import torch


def retrieve_top_k_batch(query_embs: np.ndarray, k_list: list[int]) -> list[list[str]]:
  # Convert query embeddings to the same dtype as document embeddings and move to device (CUDA)
  query_embs_tensor = torch.tensor(query_embs, dtype=torch.float16).to('cuda')

  # Compute the cosine similarity matrix: sims shape [num_queries, num_documents]
  sims = torch.matmul(query_embs_tensor, doc_embeddings_tensor.T)  # Efficient batch-wise similarity calculation

  # Initialize an empty list for the batch results
  batch_results = []

  # Process all queries at once
  top_k_indices = torch.topk(sims, k=max(k_list), dim=1).indices  # Get top-k indices for all queries in one go

  # For each query in the batch, retrieve the top-k documents
  for i, k in enumerate(k_list):
    # Retrieve the top-k indices for the current query, limited by k
    batch_results.append([documents[idx] for idx in top_k_indices[i, :k].cpu().numpy()])

  return batch_results


# Define request model
class QueryRequest(BaseModel):
  query: str
  k: int = 2
  _id: str = None


def batch_worker():
  while True:
    # Get a batch of requests in a blocking manner (non-async)
    batch = request_queue.get_batch(MAX_BATCH_SIZE, MAX_WAIT_TIME)

    if not batch:
      # If there is no batch, sleep for a short time before checking again
      time.sleep(0.01)  # Using time.sleep() since we're not in an async function
      continue

    try:
      # Extract the queries, ks, and ids from the batch
      queries = [req.query for req in batch]
      ks = [req.k for req in batch]
      ids = [req._id for req in batch]

      # Get the embeddings for the queries and retrieve the top-k documents
      query_embs = get_embedding_batch(queries)
      retrieved_docs_batch = retrieve_top_k_batch(query_embs, ks)

      # Create prompts for the chat model
      prompts = [f'Question: {query}\nContext:\n{chr(10).join(docs)}\nAnswer:' for query, docs in zip(queries, retrieved_docs_batch)]

      # Generate responses
      generations = chat_pipeline(prompts, max_length=50, do_sample=True)

      results = [g[0]['generated_text'] for g in generations]

      for req_id, result in zip(ids, results):
        response_queues[req_id].put(result)

    except Exception as e:
      print(f'[Batch Worker ERROR] Chat generation failed: {e}')


# FastAPI route to handle requests
@app.post('/rag')
def predict(payload: QueryRequest):
  payload._id = f'req_{uuid.uuid4()}'

  resp_q = queue.Queue()
  response_queues[payload._id] = resp_q

  # Add request to the queue
  request_queue.add_request(payload)

  try:
    # Wait for the result in the response queue with a timeout
    result = resp_q.get(timeout=30.0)
  except queue.Empty:
    print(f'Timeout for request {payload._id}')
    raise HTTPException(status_code=504, detail='Timeout waiting for batch response')
  except Exception as e:
    print(f'Error in response queue: {e}')
    raise HTTPException(status_code=500, detail='Server error')
  finally:
    del response_queues[payload._id]

  return {'query': payload.query, 'result': result}


# Launch background worker when the app starts
@app.on_event('startup')
def start_batch_worker():
  # Start the batch worker in a new thread
  threading.Thread(target=batch_worker, daemon=True).start()


if __name__ == '__main__':
  port = int(os.environ.get('PORT', 8000))
  print(f'Starting server on port {port}...')
  uvicorn.run(app, host='0.0.0.0', port=port)
