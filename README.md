# GPU-Accelerated AI Classroom Attendance System

This repository contains a high-performance, GPU-accelerated application designed to automate classroom attendance using face recognition feature matching. The core matching algorithm uses a custom CUDA kernel to parallelize the distance computations between extracted facial embeddings from a classroom image and a pre-registered student database.

## System Requirements
- NVIDIA GPU (Compute Capability 3.0 or higher)
- CUDA Toolkit installed (nvcc compiler)
- Linux environment or compatible terminal

## File Structure
- `attendance_gpu.cu`: Core C++/CUDA source code containing CLI handling and parallel matching kernel.
- `Makefile`: Build automation file.
- `students.csv`: Mock database containing student IDs and pre-computed facial embedding vectors.

## Compilation Instructions
To clean previous builds and compile the project, run:
```bash
make clean
make
