#!/bin/bash

# Activate the Conda environment
echo "Activating Conda environment..."
conda activate PhotoRealisticAI

# Function to clean up processes on exit
cleanup() {
    echo "Cleaning up..."
    # Terminate both processes
    kill -SIGTERM $backend_pid $ng_serve_pid
    exit 0
}

# Trap interrupt signal (Ctrl+C) to call the cleanup function
trap cleanup INT

# Start the backend API
echo "Starting backend API..."
python backend/generate_images_api.py &
# Capture the process ID (PID) of the backend API command
backend_pid=$!

sleep 10

# Start the frontend server in the background
echo "Starting frontend server..."
cd frontend/photorealistic-ai-frontend
ng serve &
# Capture the process ID (PID) of the ng serve command
ng_serve_pid=$!

# Wait for the backend to start (you may need to adjust the sleep duration)
sleep 10

# Wait for both processes to finish
wait $backend_pid
wait $ng_serve_pid
