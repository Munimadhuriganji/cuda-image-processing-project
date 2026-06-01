#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

__global__ void grayscaleKernel(unsigned char* input, unsigned char* output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (y * width + x) * channels;

        unsigned char blue = input[idx];
        unsigned char green = input[idx + 1];
        unsigned char red = input[idx + 2];

        output[y * width + x] = (red + green + blue) / 3;
    }
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "Usage: ./image_processing <image_path>" << endl;
        return -1;
    }

    Mat image = imread(argv[1], IMREAD_COLOR);

    if (image.empty())
    {
        cout << "Could not open image" << endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();

    int inputSize = width * height * channels * sizeof(unsigned char);
    int outputSize = width * height * sizeof(unsigned char);

    unsigned char *d_input, *d_output;

    cudaMalloc((void**)&d_input, inputSize);
    cudaMalloc((void**)&d_output, outputSize);

    cudaMemcpy(d_input, image.data, inputSize, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y);

    auto start = chrono::high_resolution_clock::now();

    grayscaleKernel<<<blocks, threads>>>(d_input, d_output, width, height, channels);

    cudaDeviceSynchronize();

    auto stop = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> duration = stop - start;

    cout << "GPU Execution Time: " << duration.count() << " ms" << endl;

    Mat outputImage(height, width, CV_8UC1);

    cudaMemcpy(outputImage.data, d_output, outputSize, cudaMemcpyDeviceToHost);

    imwrite("output/grayscale.jpg", outputImage);

    cout << "Output image saved successfully." << endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
