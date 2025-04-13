#!/bin/bash
set -e

echo "[INFO] Starting load balancer..."
python load_balancer.py & 
LB_PID=$!
trap "echo '[INFO] Cleaning up...'; kill $LB_PID $AUTO_PID 2>/dev/null; exit" INT TERM

echo "[INFO] Waiting 30 seconds for load balancer to initialize..."
sleep 30

echo "[INFO] Starting autoscaling backend..."
python autoscaler.py & 
AUTO_PID=$!

echo "[INFO] Waiting 2 minutes for autoscaling backend to initialize..."
sleep 120

echo "[INFO] Running benchmark script..."
python tests/general_benchmark.py --timing_mode ideal --target lb_server --total_requests 1000 --output_file scalingIdeal
python tests/general_benchmark.py --timing_mode poisson --target lb_server --total_requests 1000 --output_file scalingPoisson

echo "[INFO] Cleaning up processes..."
kill $LB_PID $AUTO_PID 2>/dev/null
wait $LB_PID $AUTO_PID 2>/dev/null

echo "[INFO] Done."
