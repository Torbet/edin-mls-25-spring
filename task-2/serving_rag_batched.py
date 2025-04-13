# ---------------------------- Imports ----------------------------
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
from torch.cuda.amp import autocast  # For mixed precision operations on GPU
from request_queue import RequestQueue  # Custom queue for handling requests
import argparse  # For parsing command-line arguments

# ---------------------------- FastAPI App & Command-line Arguments ----------------------------
app = FastAPI()

# Parse command-line arguments to customize the batch size for processing requests
parser = argparse.ArgumentParser(description='Start the RAG service with customizable max batch size.')
parser.add_argument('--max_batch_size', type=int, default=20, help='Maximum batch size for request processing')
args = parser.parse_args()

# Assign the parsed batch size value to the MAX_BATCH_SIZE variable
MAX_BATCH_SIZE = args.max_batch_size
MAX_WAIT_TIME = 0.5  # Maximum wait time in seconds before processing a batch of requests

request_queue = RequestQueue()  # Queue to handle incoming requests
response_queues = {}  # Dictionary to map request IDs to response queues

# ---------------------------- Example Documents ----------------------------
# In-memory sample documents for document retrieval
documents = [
  'Cats are small furry carnivores that are often kept as pets.',
  'Dogs are domesticated mammals, not natural wild animals.',
  'Hummingbirds can hover in mid-air by rapidly flapping their wings.',
]

# Set the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------- Load Embedding Model ----------------------------
print('Loading embedding model...')
EMBED_MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)  # Load tokenizer
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)  # Load model and move to device (GPU/CPU)
print('Embedding model loaded.')

# ---------------------------- Initialize Chat Pipeline ----------------------------
# Basic chat pipeline (text generation model) using Hugging Face's pipeline
chat_pipeline = pipeline('text-generation', model='facebook/opt-125m', device=0 if torch.cuda.is_available() else -1)

# ---------------------------- Function to Get Embeddings ----------------------------
def get_embedding_batch(texts: list[str]) -> np.ndarray:
  """
  Tokenizes input texts and generates normalized embeddings using the preloaded model.
  Uses mixed precision (autocast) for performance when on GPU.
  """
  inputs = embed_tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to('cuda')  # Tokenize and send to GPU

  # No gradient tracking for faster inference
  with torch.no_grad():
    with torch.amp.autocast('cuda'):  # Use automatic mixed precision on GPU
      outputs = embed_model(**inputs)

  # Average token embeddings to create sentence-level embeddings
  embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

  # Normalize embeddings (cosine similarity works better with normalized vectors)
  norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
  normalized_embeddings = embeddings / norms

  return normalized_embeddings

# ---------------------------- Precompute Document Embeddings ----------------------------
# Precompute embeddings for documents and store them on the GPU
doc_embeddings = get_embedding_batch(documents)
doc_embeddings_tensor = torch.tensor(doc_embeddings, dtype=torch.float16).to('cuda')  # Store on GPU as half-precision

# ---------------------------- Function to Retrieve Top-K Documents ----------------------------
def retrieve_top_k_batch(query_embs: np.ndarray, k_list: list[int]) -> list[list[str]]:
  """
  Retrieves top-k most similar documents based on cosine similarity for each query embedding.
  """
  query_embs_tensor = torch.tensor(query_embs, dtype=torch.float16).to('cuda')  # Move query embeddings to GPU
  sims = torch.matmul(query_embs_tensor, doc_embeddings_tensor.T)  # Compute similarity between query and document embeddings

  batch_results = []
  top_k_indices = torch.topk(sims, k=max(k_list), dim=1).indices  # Get top-k indices for all queries in one go

  # Retrieve the top-k documents for each query
  for i, k in enumerate(k_list):
    batch_results.append([documents[idx] for idx in top_k_indices[i, :k].cpu().numpy()])

  return batch_results

# ---------------------------- Request Model ----------------------------
# Define the request payload model using Pydantic for data validation
class QueryRequest(BaseModel):
  query: str  # The query string
  k: int = 2  # Number of documents to retrieve (default is 2)
  _id: str = None  # Unique identifier for the request

# ---------------------------- Batch Worker ----------------------------
# Function to process requests in batches
def batch_worker():
  while True:
    batch = request_queue.get_batch(MAX_BATCH_SIZE, MAX_WAIT_TIME)  # Get a batch of requests

    if not batch:
      time.sleep(0.01)  # If no requests, sleep for a short time
      continue

    try:
      queries = [req.query for req in batch]
      ks = [req.k for req in batch]
      ids = [req._id for req in batch]

      # Generate embeddings for the queries and retrieve the top-k documents
      query_embs = get_embedding_batch(queries)
      retrieved_docs_batch = retrieve_top_k_batch(query_embs, ks)

      # Generate chat model prompts from queries and retrieved documents
      prompts = [f'Question: {query}\nContext:\n{chr(10).join(docs)}\nAnswer:' for query, docs in zip(queries, retrieved_docs_batch)]

      # Get the generated responses from the chat model
      generations = chat_pipeline(prompts, max_length=50, do_sample=True)
      results = [g[0]['generated_text'] for g in generations]

      # Store the results in the corresponding response queues
      for req_id, result in zip(ids, results):
        response_queues[req_id].put(result)

    except Exception as e:
      print(f'[Batch Worker ERROR] Chat generation failed: {e}')

# ---------------------------- FastAPI Route ----------------------------
@app.post('/rag')
def predict(payload: QueryRequest):
  """
  FastAPI route to handle requests. It adds the request to the queue and waits for a response.
  """
  payload._id = f'req_{uuid.uuid4()}'  # Generate a unique ID for each request

  resp_q = queue.Queue()  # Create a response queue for the request
  response_queues[payload._id] = resp_q  # Store the response queue in a global dictionary

  request_queue.add_request(payload)  # Add the request to the request queue

  try:
    # Wait for the response in the queue with a timeout
    result = resp_q.get(timeout=30.0)
  except queue.Empty:
    print(f'Timeout for request {payload._id}')
    raise HTTPException(status_code=504, detail='Timeout waiting for batch response')
  except Exception as e:
    print(f'Error in response queue: {e}')
    raise HTTPException(status_code=500, detail='Server error')
  finally:
    del response_queues[payload._id]  # Remove the response queue after processing

  return {'query': payload.query, 'result': result}

# ---------------------------- Launch Background Worker ----------------------------
@app.on_event('startup')
def start_batch_worker():
  """
  Launch the batch worker when the FastAPI app starts.
  The worker processes requests in a separate thread.
  """
  threading.Thread(target=batch_worker, daemon=True).start()

# ---------------------------- Start Server ----------------------------
if __name__ == '__main__':
  port = int(os.environ.get('PORT', 8000))
  print(f'Starting server on port {port}...')
  uvicorn.run(app, host='0.0.0.0', port=port)  # Run the FastAPI app with Uvicorn
