# GPU Accelerated Image Processing using CUDA

## Overview
This project demonstrates GPU-based image processing using CUDA C++. The application performs grayscale conversion on images using parallel CUDA kernels.

## Features
- CUDA GPU acceleration
- CPU vs GPU performance comparison
- Parallel image processing
- Execution timing measurements

## Technologies Used
- CUDA C++
- NVIDIA GPU
- OpenCV
- Linux / Ubuntu

## Build Instructions

### Prerequisites
- NVIDIA CUDA Toolkit
- OpenCV
- g++

### Compile
```bash
nvcc image_processing.cu -o image_processing `pkg-config --cflags --libs opencv4`
```

### Run
```bash
./image_processing input/sample.jpg
```

## Output
Processed grayscale image will be saved in the output folder.

## Author
Madhu
