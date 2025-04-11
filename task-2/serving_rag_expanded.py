import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import uuid
import asyncio
import time
from request_queue import RequestQueue
import os
from torch.cuda.amp import autocast

app = FastAPI()

MAX_BATCH_SIZE = 10
MAX_WAIT_TIME = 0.5  # seconds

request_queue = RequestQueue()
response_queues = {}

# Example documents in memory
documents = [
    'Cats are small furry carnivores that are often kept as pets.',
    'Dogs are domesticated mammals, not natural wild animals.',
    'Hummingbirds can hover in mid-air by rapidly flapping their wings.',
]

# 1. Load embedding model
EMBED_MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

# Check if CUDA is available and move the model to GPU if so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_model = embed_model.to(device)

# Basic Chat LLM
chat_pipeline = pipeline('text-generation', model='facebook/opt-125m', device=0 if torch.cuda.is_available() else -1)

def get_embedding_batch(texts: list[str]) -> np.ndarray:
    # Tokenize the texts and send to the appropriate device (GPU or CPU)
    inputs = embed_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    # Move tokenized inputs to the same device as the model (GPU or CPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Use mixed precision for faster computations on supported hardware (FP16) if on GPU
    with torch.no_grad(), autocast(device.type):  # Automatically use "cuda" or "cpu" for autocast
        # Get the model outputs
        outputs = embed_model(**inputs)
        
    # Perform average pooling on the token embeddings to get sentence-level embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Transfer embeddings to CPU for further processing
    return embeddings

# Precompute document embeddings
doc_embeddings = np.vstack([get_embedding_batch([doc]) for doc in documents])

def retrieve_top_k_batch(query_embs: np.ndarray, k_list: list[int]) -> list[list[str]]:
    batch_results = []
    
    # Convert the query embeddings to a PyTorch tensor for optimized operations
    query_embs_tensor = torch.tensor(query_embs).to(device)  # Move to GPU if available
    doc_embeddings_tensor = torch.tensor(doc_embeddings).to(device)  # Document embeddings on GPU
    
    # Compute similarities for all queries at once (matrix multiplication)
    sims = torch.matmul(query_embs_tensor, doc_embeddings_tensor.T)  # Shape: (batch_size, num_documents)
    
    # For each query, retrieve top-k documents
    for i, k in enumerate(k_list):
        top_k_indices = torch.argsort(sims[i], descending=True)[:k].cpu().numpy()  # Get top-k indices for the i-th query
        batch_results.append([documents[idx] for idx in top_k_indices])  # Append top-k docs to the results list
    
    return batch_results

def rag_pipeline(query: str, k: int = 2) -> str:
    # Step 1: Input embedding
    query_emb = get_embedding_batch([query])

    # Step 2: Retrieval
    retrieved_docs = retrieve_top_k_batch(query_emb, [k])

    # Construct the prompt from query + retrieved docs
    context = '\n'.join(retrieved_docs[0])
    prompt = f'Question: {query}\nContext:\n{context}\nAnswer:'

    # Step 3: LLM Output
    generated = chat_pipeline(prompt, max_length=50, do_sample=True)[0]['generated_text']
    return generated

# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2
    _id: str = None

async def batch_worker():

    while True:
        # Await the asynchronous call to get_batch
        batch = await request_queue.get_batch(MAX_BATCH_SIZE, MAX_WAIT_TIME)
        if not batch:
            # If there is no batch, sleep for a short time before checking again
            await asyncio.sleep(0.01)
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
            prompts = [
                f"Question: {query}\nContext:\n{chr(10).join(docs)}\nAnswer:"
                for query, docs in zip(queries, retrieved_docs_batch)
            ]

            # Generate responses
            generations = chat_pipeline(prompts, max_length=50, do_sample=True)

            results = [g[0]["generated_text"] for g in generations]

            for req_id, result in zip(ids, results):
                await response_queues[req_id].put(result)

        except Exception as e:
            print(f"[Batch Worker ERROR] Chat generation failed: {e}")
            
# FastAPI route to handle requests
@app.post("/rag")
async def predict(payload: QueryRequest):
    payload._id = f"req_{uuid.uuid4()}"
    
    resp_q = asyncio.Queue()
    response_queues[payload._id] = resp_q

    # Add request to the queue
    await request_queue.add_request(payload)

    try:
        result = await asyncio.wait_for(resp_q.get(), timeout=30.0)
    except asyncio.TimeoutError:
        print(f"Timeout for request {payload._id}")
        raise HTTPException(status_code=504, detail="Timeout waiting for batch response")
    except Exception as e:
        print(f"Error in response queue: {e}")
        raise HTTPException(status_code=500, detail="Server error")
    finally:
        del response_queues[payload._id]

    return {"query": payload.query, "result": result}

# Launch background worker when the app starts
@app.on_event("startup")
async def start_batch_worker():
    # Correct the usage of asyncio.create_task here to start the batch worker correctly
    asyncio.create_task(batch_worker())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
