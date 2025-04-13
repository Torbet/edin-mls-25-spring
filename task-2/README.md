# Task 2
A FastAPI-based Retrieval-Augmented Generation (RAG) service that combines document retrieval with text generation.

## Overview:
The following files implement a FastAPI-based Retrieval-Augmented Generation (RAG) service, which combines document retrieval with text generation.

Three implementations are provided: base, batched, and autoscaling.

- **Base Implementation**: This is the simplest version of the service, which performs document retrieval and text generation without any batching optimisations.

- **Batched Implementation**: This implementation batches incoming requests together to optimize the use of batching when processing with the LLM (Large Language Model). This reduces overhead and improves throughput.

- **Autoscaling and Load Balancing**: Building on the batched implementation, this version takes things further by dynamically distributing incoming requests across multiple servers. This helps balance the load efficiently and ensures better scalability under high traffic.

## Enviromental Setup
### Step 1: Create a Conda Environment

Create a conda environment with Python 3.10 and activate it.

   **TIP**: Check [this example](https://github.com/ServerlessLLM/ServerlessLLM/blob/main/docs/stable/getting_started/slurm_setup.md) for how to use SLURM to create a conda environment.

```bash
 conda create -n rag python=3.10 -y
 conda activate rag
```
### Step 2: Clone the Repository and Install Dependencies
Clone the repository and install the dependencies from ```requirements.txt```.

```bash
git clone https://github.com/Torbet/edin-mls-25-spring.git
cd edin-mls-25-spring/task-2
pip install -r requirements.txt
```
This will set up the environment with all the required packages for the project.

**Note:**  
If you encounter issues while downloading model checkpoints on a GPU machine, try the following workaround:  

1. Manually download the model on the host machine:  

```bash
conda activate rag
huggingface-cli download <model_name>
```

## Running the Server

To run the examples, first open an interactive session on the teaching cluster, it is advised to use the most powerfull GPU available on the cluster as startup times and processing greatly depends on the hardware.

The following will use the RTX A6000 GPU
```bash
srun --gres=gpu:a6000:1 --pty bash
```
alternatively you can call the below to get any availble GPU
```bash
srun --gres=gpu:gpu:1 --pty bash
```
After loading in you may have to reactivate your conda enviroment and also cd back into ```edin-mls-25-spring/task-2```

The servers can be ran with the following command:

```bash
python serving_rag.py &
#or
python serving_rag_batched.py &
#or
python serving_rag_scaling.py &
```
To test the service connections you can do it by:

```bash
curl -X POST "http://localhost:8000/rag" -H "Content-Type: application/json" -d '{"query": "Which animals can hover in the air?"}'
```

## Testing

To verify the server's performance under various loads, a custom asynchronous testing script is provided. The test simulates client requests to the server with configurable parameters.

### Script Options

The test runner supports the following command-line arguments:

```bash
python your_script.py --timing_mode <mode> --target <endpoint> --rps <rate> --num_requests <count>
```

| Argument         | Description                                                                                                 |
|------------------|-------------------------------------------------------------------------------------------------------------|
| `--timing_mode`  | **Request timing mode**:                                                                                     |
|                  | - `ideal`: Sends requests at evenly spaced intervals.                                                       |
|                  | - `poisson`: Sends requests with randomized intervals (exponential distribution).                           |
| `--target`       | **Target endpoint**: Chooses which server or load balancer to hit. Options are defined in the `ENDPOINTS`. |
| `--rps`          | **Requests per second**: How many requests to send each second.                                             |
| `--num_requests` | **Total number of requests**: Total requests to send during the test.                                       |



## Benchmarking:

The benchmarking tasks were run using the Teaching Cluster with an RTX A6000 GPU.

To run the benchmarking tests, you can use the following bash scripts:

- `run_benchmarks_base.sh`
- `run_benchmarks_batched.sh`
- `run_benchmarks_scaling.sh`

### Running the Benchmarking Tests:

To execute the benchmarking scripts, use the following command:

```bash
srun --gres=gpu:a6000:1 --pty ". ./BENCHMARKING_script"
```

The scripts will output results in a csv under ```test-results``` diretory

## Plotting Results
After gathering results all plotting happens by running the command:
```bash
python generate_plots.py
```

The plots include the following:
  - Latency vs RPS (Multiple variations)
  - Throuput vs RPS (Multiple variations)
  - Max Batch Size of 10 vs 20
    






