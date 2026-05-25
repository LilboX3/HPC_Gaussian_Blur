#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if _WIN32
#include <CL/cl.h>
#elif __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "cppTga/tga.h"

#define _USE_MATH_DEFINES

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::string cl_errorstring(cl_int err)
{
	switch (err)
	{
	case CL_SUCCESS:									return std::string("Success");
	case CL_DEVICE_NOT_FOUND:							return std::string("Device not found");
	case CL_DEVICE_NOT_AVAILABLE:						return std::string("Device not available");
	case CL_COMPILER_NOT_AVAILABLE:						return std::string("Compiler not available");
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:				return std::string("Memory object allocation failure");
	case CL_OUT_OF_RESOURCES:							return std::string("Out of resources");
	case CL_OUT_OF_HOST_MEMORY:							return std::string("Out of host memory");
	case CL_PROFILING_INFO_NOT_AVAILABLE:				return std::string("Profiling information not available");
	case CL_MEM_COPY_OVERLAP:							return std::string("Memory copy overlap");
	case CL_IMAGE_FORMAT_MISMATCH:						return std::string("Image format mismatch");
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:					return std::string("Image format not supported");
	case CL_BUILD_PROGRAM_FAILURE:						return std::string("Program build failure");
	case CL_MAP_FAILURE:								return std::string("Map failure");
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:				return std::string("Misaligned sub buffer offset");
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	return std::string("Exec status error for events in wait list");
	case CL_INVALID_VALUE:                    			return std::string("Invalid value");
	case CL_INVALID_DEVICE_TYPE:              			return std::string("Invalid device type");
	case CL_INVALID_PLATFORM:                 			return std::string("Invalid platform");
	case CL_INVALID_DEVICE:                   			return std::string("Invalid device");
	case CL_INVALID_CONTEXT:                  			return std::string("Invalid context");
	case CL_INVALID_QUEUE_PROPERTIES:         			return std::string("Invalid queue properties");
	case CL_INVALID_COMMAND_QUEUE:            			return std::string("Invalid command queue");
	case CL_INVALID_HOST_PTR:                 			return std::string("Invalid host pointer");
	case CL_INVALID_MEM_OBJECT:               			return std::string("Invalid memory object");
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  			return std::string("Invalid image format descriptor");
	case CL_INVALID_IMAGE_SIZE:               			return std::string("Invalid image size");
	case CL_INVALID_SAMPLER:                  			return std::string("Invalid sampler");
	case CL_INVALID_BINARY:                   			return std::string("Invalid binary");
	case CL_INVALID_BUILD_OPTIONS:            			return std::string("Invalid build options");
	case CL_INVALID_PROGRAM:                  			return std::string("Invalid program");
	case CL_INVALID_PROGRAM_EXECUTABLE:       			return std::string("Invalid program executable");
	case CL_INVALID_KERNEL_NAME:              			return std::string("Invalid kernel name");
	case CL_INVALID_KERNEL_DEFINITION:        			return std::string("Invalid kernel definition");
	case CL_INVALID_KERNEL:                   			return std::string("Invalid kernel");
	case CL_INVALID_ARG_INDEX:                			return std::string("Invalid argument index");
	case CL_INVALID_ARG_VALUE:                			return std::string("Invalid argument value");
	case CL_INVALID_ARG_SIZE:                 			return std::string("Invalid argument size");
	case CL_INVALID_KERNEL_ARGS:             			return std::string("Invalid kernel arguments");
	case CL_INVALID_WORK_DIMENSION:          			return std::string("Invalid work dimension");
	case CL_INVALID_WORK_GROUP_SIZE:          			return std::string("Invalid work group size");
	case CL_INVALID_WORK_ITEM_SIZE:           			return std::string("Invalid work item size");
	case CL_INVALID_GLOBAL_OFFSET:            			return std::string("Invalid global offset");
	case CL_INVALID_EVENT_WAIT_LIST:          			return std::string("Invalid event wait list");
	case CL_INVALID_EVENT:                    			return std::string("Invalid event");
	case CL_INVALID_OPERATION:                			return std::string("Invalid operation");
	case CL_INVALID_GL_OBJECT:                			return std::string("Invalid OpenGL object");
	case CL_INVALID_BUFFER_SIZE:              			return std::string("Invalid buffer size");
	case CL_INVALID_MIP_LEVEL:                			return std::string("Invalid mip-map level");
	case CL_INVALID_GLOBAL_WORK_SIZE:         			return std::string("Invalid gloal work size");
	case CL_INVALID_PROPERTY:                 			return std::string("Invalid property");
	default:                                  			return std::string("Unknown error code");
	}
}

void checkStatus(cl_int err)
{
	if (err != CL_SUCCESS) {
		printf("OpenCL Error: %s \n", cl_errorstring(err).c_str());
		exit(EXIT_FAILURE);
	}
}

void printCompilerError(cl_program program, cl_device_id device)
{
	cl_int status;
	size_t logSize;
	char* log;

	// get log size
	status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	checkStatus(status);

	// allocate space for log
	log = static_cast<char*>(malloc(logSize));
	if (!log)
	{
		exit(EXIT_FAILURE);
	}

	// read the log
	status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
	checkStatus(status);

	// print the log
	printf("Build Error: %s\n", log);
	free(log);
}

tga::TGAImage loadTgaImage(const std::string& fileName)
{
	tga::TGAImage image;
	if (!tga::LoadTGA(&image, fileName.c_str()))
	{
		printf("Error: Could not load input image '%s'.\n", fileName.c_str());
		exit(EXIT_FAILURE);
	}

	return image;
}

void saveTgaImage(const std::string& fileName, const tga::TGAImage& image)
{
	if (!tga::saveTGA(image, fileName.c_str()))
	{
		printf("Error: Could not save output image '%s'.\n", fileName.c_str());
		exit(EXIT_FAILURE);
	}
}

std::vector<float> createGaussianFilter(unsigned int smooth_kernel_size, double sigma)
{
	if (smooth_kernel_size == 0 || smooth_kernel_size % 2 == 0)
	{
		printf("Error: Filter size must be odd and greater than 0.\n");
		exit(EXIT_FAILURE);
	}

	if (sigma <= 0.0)
	{
		printf("Error: Sigma must be greater than 0.\n");
		exit(EXIT_FAILURE);
	}

	std::vector<float> filter(smooth_kernel_size);
	double sum = 0.0;
	unsigned int i;
	const int radius = static_cast<int>(smooth_kernel_size / 2);

	for (i = 0; i < smooth_kernel_size; i++) {
		double x = static_cast<double>(static_cast<int>(i) - radius);
		double gauss = std::exp(-(x * x) / (2.0 * sigma * sigma));
		filter[i] = static_cast<float>(gauss);
		sum += gauss;
	}

	for (i = 0; i < smooth_kernel_size; i++) {
		filter[i] = static_cast<float>(filter[i] / sum);
	}

	return filter;
}

tga::TGAImage resizeImageNearestNeighbor(const tga::TGAImage& sourceImage, unsigned int targetWidth, unsigned int targetHeight)
{
	const unsigned int channels = sourceImage.bpp / 8;
	tga::TGAImage resizedImage;
	resizedImage.width = targetWidth;
	resizedImage.height = targetHeight;
	resizedImage.bpp = sourceImage.bpp;
	resizedImage.type = sourceImage.type;
	resizedImage.imageData.resize(static_cast<size_t>(targetWidth) * targetHeight * channels);

	for (unsigned int y = 0; y < targetHeight; ++y)
	{
		const unsigned int sourceY = std::min(
			static_cast<unsigned int>((static_cast<unsigned long long>(y) * sourceImage.height) / targetHeight),
			sourceImage.height - 1);

		for (unsigned int x = 0; x < targetWidth; ++x)
		{
			const unsigned int sourceX = std::min(
				static_cast<unsigned int>((static_cast<unsigned long long>(x) * sourceImage.width) / targetWidth),
				sourceImage.width - 1);
			const size_t destinationIndex = (static_cast<size_t>(y) * targetWidth + x) * channels;
			const size_t sourceIndex = (static_cast<size_t>(sourceY) * sourceImage.width + sourceX) * channels;

			for (unsigned int channel = 0; channel < channels; ++channel)
			{
				resizedImage.imageData[destinationIndex + channel] = sourceImage.imageData[sourceIndex + channel];
			}
		}
	}

	return resizedImage;
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		printf("Usage: %s <input.tga> [output.tga] [filter_size] [sigma]\n", argv[0]);
		return EXIT_FAILURE;
	}

	const std::string inputFileName = argv[1];
	const std::string outputFileName = argc > 2 ? argv[2] : "blurred_output.tga";
	const unsigned int filterSize = argc > 3 ? static_cast<unsigned int>(std::strtoul(argv[3], NULL, 10)) : 9;
	const double sigma = argc > 4 ? std::strtod(argv[4], NULL) : 1.0;

	tga::TGAImage inputImage = loadTgaImage(inputFileName);
	std::vector<float> gaussianFilter = createGaussianFilter(filterSize, sigma);
	const unsigned int channels = inputImage.bpp / 8;

	printf("Input image: %s (%ux%u, %u channels)\n", inputFileName.c_str(), inputImage.width, inputImage.height, channels);
	printf("Output image: %s\n", outputFileName.c_str());
	printf("Gaussian filter: 1x%u, sigma %.3f\n", filterSize, sigma);

	// used for checking error status of api calls
	cl_int status;

	// retrieve the number of platforms
	cl_uint numPlatforms = 0;
	checkStatus(clGetPlatformIDs(0, NULL, &numPlatforms));

	if (numPlatforms == 0)
	{
		printf("Error: No OpenCL platform available!\n");
		exit(EXIT_FAILURE);
	}

	// select the platform
	cl_platform_id platform;
	checkStatus(clGetPlatformIDs(1, &platform, NULL));

	// retrieve the number of devices
	cl_uint numDevices = 0;
	checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));

	if (numDevices == 0)
	{
		printf("Error: No OpenCL device available for platform!\n");
		exit(EXIT_FAILURE);
	}

	// select the device
	cl_device_id device;
	checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

	// create context
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	checkStatus(status);

	// create command queue
	cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &status);
	checkStatus(status);

	// read the kernel source
	const char* kernelFileName = "kernel.cl";
	std::ifstream ifs(kernelFileName);
	if (!ifs.good())
	{
		printf("Error: Could not open kernel with file name %s!\n", kernelFileName);
		exit(EXIT_FAILURE);
	}

	std::string programSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	const char* programSourceArray = programSource.c_str();
	size_t programSize = programSource.length();

	// create the program
	cl_program program = clCreateProgramWithSource(context, 1, static_cast<const char**>(&programSourceArray), &programSize, &status);
	checkStatus(status);

	// build the program
	status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		printCompilerError(program, device);
		exit(EXIT_FAILURE);
	}

	// create the vector addition kernel
	cl_kernel kernel = clCreateKernel(program, "gaussian_blur", &status);
	checkStatus(status);

	// output device capabilities
	size_t maxWorkGroupSize;
	checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL));
	printf("Device Capabilities: Max work items in single group: %zu\n", maxWorkGroupSize);

	size_t kernelMaxWorkGroupSize;
	checkStatus(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernelMaxWorkGroupSize, NULL));
	printf("Kernel Capabilities: Max work items in single group: %zu\n", kernelMaxWorkGroupSize);

	cl_uint maxWorkItemDimensions;
	checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWorkItemDimensions, NULL));
	printf("Device Capabilities: Max work item dimensions: %u\n", maxWorkItemDimensions);

	size_t* maxWorkItemSizes = static_cast<size_t*>(malloc(maxWorkItemDimensions * sizeof(size_t)));
	checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemDimensions * sizeof(size_t), maxWorkItemSizes, NULL));
	printf("Device Capabilities: Max work items in group per dimension:");
	for (cl_uint i = 0; i < maxWorkItemDimensions; ++i)
		printf(" %u:%zu", i, maxWorkItemSizes[i]);
	printf("\n");

	cl_ulong localMemorySize;
	checkStatus(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemorySize, NULL));
	printf("Device Capabilities: Local memory size: %llu bytes\n", static_cast<unsigned long long>(localMemorySize));

	const size_t effectiveMaxWorkGroupSize = std::min(maxWorkGroupSize, kernelMaxWorkGroupSize);
	const size_t localPixelsCapacity = static_cast<size_t>(localMemorySize / (channels * sizeof(cl_uchar)));
	const size_t rowWorkGroupLimit = std::min(effectiveMaxWorkGroupSize, std::min(maxWorkItemSizes[0], localPixelsCapacity));
	const size_t columnWorkGroupLimit = std::min(effectiveMaxWorkGroupSize, std::min(maxWorkItemSizes[1], localPixelsCapacity));
	printf("Derived limits: row work-group <= %zu, column work-group <= %zu\n", rowWorkGroupLimit, columnWorkGroupLimit);
	free(maxWorkItemSizes);

	if (rowWorkGroupLimit == 0 || columnWorkGroupLimit == 0)
	{
		printf("Error: Local memory is too small to cache even a single pixel per work-group.\n");
		exit(EXIT_FAILURE);
	}

	if (inputImage.width > rowWorkGroupLimit || inputImage.height > columnWorkGroupLimit)
	{
		const double rowScale = static_cast<double>(rowWorkGroupLimit) / inputImage.width;
		const double columnScale = static_cast<double>(columnWorkGroupLimit) / inputImage.height;
		const double scale = std::min(rowScale, columnScale);

		if (scale <= 0.0)
		{
			printf("Error: Could not resize image to match device capabilities.\n");
			exit(EXIT_FAILURE);
		}

		const unsigned int resizedWidth = std::max(1u, static_cast<unsigned int>(std::floor(inputImage.width * scale)));
		const unsigned int resizedHeight = std::max(1u, static_cast<unsigned int>(std::floor(inputImage.height * scale)));
		printf("Resizing input from %ux%u to %ux%u to satisfy row/column work-group limits.\n",
			inputImage.width, inputImage.height, resizedWidth, resizedHeight);

		inputImage = resizeImageNearestNeighbor(inputImage, resizedWidth, resizedHeight);
	}

	if (inputImage.width > rowWorkGroupLimit || inputImage.height > columnWorkGroupLimit)
	{
		printf("Error: Resulting work-group is still too big for this system.\n");
		exit(EXIT_FAILURE);
	}

	tga::TGAImage outputImage = inputImage;
	const cl_uint width = inputImage.width;
	const cl_uint height = inputImage.height;
	const cl_uint channelCount = channels;
	const cl_uint kernelSize = filterSize;
	const size_t imageByteSize = inputImage.imageData.size() * sizeof(unsigned char);
	const size_t filterByteSize = gaussianFilter.size() * sizeof(float);
	const size_t horizontalLocalBytes = static_cast<size_t>(width) * channelCount * sizeof(cl_uchar);
	const size_t verticalLocalBytes = static_cast<size_t>(height) * channelCount * sizeof(cl_uchar);

	printf("Processing image size: %ux%u\n", width, height);
	printf("Horizontal pass: local work-group = {%u, 1}, local memory = %zu bytes\n", width, horizontalLocalBytes);
	printf("Vertical pass: local work-group = {1, %u}, local memory = %zu bytes\n", height, verticalLocalBytes);

	// allocate two input and one output buffer
	cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, imageByteSize, NULL, &status);
	checkStatus(status);
	cl_mem intermediateBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, imageByteSize, NULL, &status);
	checkStatus(status);
	cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imageByteSize, NULL, &status);
	checkStatus(status);
	cl_mem filterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, filterByteSize, NULL, &status);
	checkStatus(status);

	// write data from the input to the buffers
	checkStatus(clEnqueueWriteBuffer(commandQueue, inputBuffer, CL_TRUE, 0, imageByteSize, inputImage.imageData.data(), 0, NULL, NULL));
	checkStatus(clEnqueueWriteBuffer(commandQueue, filterBuffer, CL_TRUE, 0, filterByteSize, gaussianFilter.data(), 0, NULL, NULL));

	size_t globalWorkSize[2];
	globalWorkSize[0] = static_cast<size_t>(width);
	globalWorkSize[1] = static_cast<size_t>(height);

	size_t horizontalLocalWorkSize[2];
	horizontalLocalWorkSize[0] = static_cast<size_t>(width);
	horizontalLocalWorkSize[1] = 1;

	size_t verticalLocalWorkSize[2];
	verticalLocalWorkSize[0] = 1;
	verticalLocalWorkSize[1] = static_cast<size_t>(height);

	cl_uint isHorizontal = 1;

	// set the kernel arguments
	checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer));
	checkStatus(clSetKernelArg(kernel, 1, sizeof(cl_mem), &intermediateBuffer));
	checkStatus(clSetKernelArg(kernel, 2, sizeof(cl_mem), &filterBuffer));
	checkStatus(clSetKernelArg(kernel, 3, sizeof(cl_uint), &width));
	checkStatus(clSetKernelArg(kernel, 4, sizeof(cl_uint), &height));
	checkStatus(clSetKernelArg(kernel, 5, sizeof(cl_uint), &channelCount));
	checkStatus(clSetKernelArg(kernel, 6, sizeof(cl_uint), &kernelSize));
	checkStatus(clSetKernelArg(kernel, 7, sizeof(cl_uint), &isHorizontal));
	checkStatus(clSetKernelArg(kernel, 8, horizontalLocalBytes, NULL));

	cl_event horizontalPassEvent;
	checkStatus(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, horizontalLocalWorkSize, 0, NULL, &horizontalPassEvent));

	isHorizontal = 0;
	checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &intermediateBuffer));
	checkStatus(clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer));
	checkStatus(clSetKernelArg(kernel, 2, sizeof(cl_mem), &filterBuffer));
	checkStatus(clSetKernelArg(kernel, 3, sizeof(cl_uint), &width));
	checkStatus(clSetKernelArg(kernel, 4, sizeof(cl_uint), &height));
	checkStatus(clSetKernelArg(kernel, 5, sizeof(cl_uint), &channelCount));
	checkStatus(clSetKernelArg(kernel, 6, sizeof(cl_uint), &kernelSize));
	checkStatus(clSetKernelArg(kernel, 7, sizeof(cl_uint), &isHorizontal));
	checkStatus(clSetKernelArg(kernel, 8, verticalLocalBytes, NULL));

	cl_event verticalPassEvent;
	checkStatus(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, verticalLocalWorkSize, 1, &horizontalPassEvent, &verticalPassEvent));

	// read the device output buffer to the host output array
	checkStatus(clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, imageByteSize, outputImage.imageData.data(), 1, &verticalPassEvent, NULL));
	saveTgaImage(outputFileName, outputImage);

	checkStatus(clReleaseEvent(verticalPassEvent));
	checkStatus(clReleaseEvent(horizontalPassEvent));
	checkStatus(clReleaseKernel(kernel));
	checkStatus(clReleaseProgram(program));
	checkStatus(clReleaseMemObject(filterBuffer));
	checkStatus(clReleaseMemObject(outputBuffer));
	checkStatus(clReleaseMemObject(intermediateBuffer));
	checkStatus(clReleaseMemObject(inputBuffer));
	checkStatus(clReleaseCommandQueue(commandQueue));
	checkStatus(clReleaseContext(context));

	exit(EXIT_SUCCESS);
}
