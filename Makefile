all:
	nvcc image_processing.cu -o image_processing `pkg-config --cflags --libs opencv4`

run:
	./image_processing input/sample.jpg
