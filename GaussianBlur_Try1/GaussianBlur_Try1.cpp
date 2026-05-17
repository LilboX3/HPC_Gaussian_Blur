/*
 * HPC Project SS26 – Simple Gaussian Blur (single-pass OpenCL kernel)
 *
 * Usage:
 *   GaussianBlur_Try1 <input.tga> <output.tga> [kernel_size] [sigma]
 *
 *   kernel_size  odd integer 1..9, default 5
 *   sigma        positive float,   default 1.0
 */

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define _USE_MATH_DEFINES

#ifdef _WIN32
#include <CL/cl.h>
#else
#include <OpenCL/opencl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string>

#include "cppTga/tga.h"

/* -------------------------------------------------------------------------
 * OpenCL error helpers
 * ---------------------------------------------------------------------- */

static std::string cl_errorstring(cl_int err)
{
    switch (err)
    {
    case CL_SUCCESS:                                    return "Success";
    case CL_DEVICE_NOT_FOUND:                           return "Device not found";
    case CL_DEVICE_NOT_AVAILABLE:                       return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:                     return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:              return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:                           return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:                         return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:               return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:                           return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:                      return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:                 return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:                      return "Program build failure";
    case CL_MAP_FAILURE:                                return "Map failure";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:               return "Misaligned sub-buffer offset";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:  return "Exec status error for events in wait list";
    case CL_INVALID_VALUE:                              return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:                        return "Invalid device type";
    case CL_INVALID_PLATFORM:                           return "Invalid platform";
    case CL_INVALID_DEVICE:                             return "Invalid device";
    case CL_INVALID_CONTEXT:                            return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:                   return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:                      return "Invalid command queue";
    case CL_INVALID_HOST_PTR:                           return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:                         return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:            return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:                         return "Invalid image size";
    case CL_INVALID_SAMPLER:                            return "Invalid sampler";
    case CL_INVALID_BINARY:                             return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:                      return "Invalid build options";
    case CL_INVALID_PROGRAM:                            return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:                 return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:                        return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:                  return "Invalid kernel definition";
    case CL_INVALID_KERNEL:                             return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:                          return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:                          return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:                           return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:                        return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:                     return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:                    return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:                     return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:                      return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:                    return "Invalid event wait list";
    case CL_INVALID_EVENT:                              return "Invalid event";
    case CL_INVALID_OPERATION:                          return "Invalid operation";
    case CL_INVALID_GL_OBJECT:                          return "Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:                        return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:                          return "Invalid mip-map level";
    case CL_INVALID_GLOBAL_WORK_SIZE:                   return "Invalid global work size";
    case CL_INVALID_PROPERTY:                           return "Invalid property";
    default:                                            return "Unknown error code";
    }
}

static void checkStatus(cl_int err)
{
    if (err != CL_SUCCESS)
    {
        printf("OpenCL Error: %s\n", cl_errorstring(err).c_str());
        exit(EXIT_FAILURE);
    }
}

static void printCompilerError(cl_program program, cl_device_id device)
{
    cl_int status;
    size_t logSize = 0;

    status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    checkStatus(status);

    char* log = static_cast<char*>(malloc(logSize));
    if (!log) { exit(EXIT_FAILURE); }

    status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
    checkStatus(status);

    printf("Build error:\n%s\n", log);
    free(log);
}

/* -------------------------------------------------------------------------
 * Gaussian kernel generation
 * Based on gauss_filter_kernel_generator.cpp (provided in the project).
 * Produces a normalized (kernelSize x kernelSize) kernel stored row-major.
 * ---------------------------------------------------------------------- */

static void generateGaussianKernel(float* kernel, int kernelSize, float sigma)
{
    int half = kernelSize / 2;
    double sum = 0.0;

    for (int i = 0; i < kernelSize; i++)
    {
        for (int j = 0; j < kernelSize; j++)
        {
            double x = i - half;
            double y = j - half;
            double val = (1.0 / (2.0 * M_PI * sigma * sigma)) *
                         exp(-(x * x + y * y) / (2.0 * sigma * sigma));
            kernel[i * kernelSize + j] = (float)val;
            sum += val;
        }
    }

    /* Normalize so the filter preserves overall image brightness */
    for (int i = 0; i < kernelSize * kernelSize; i++)
        kernel[i] = (float)(kernel[i] / sum);
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(int argc, char** argv)
{
    /* --- Parse command-line arguments ---------------------------------- */
    if (argc < 3)
    {
        printf("Usage: %s <input.tga> <output.tga> [kernel_size (odd 1-9, default 5)] [sigma (default 1.0)]\n",
               argv[0]);
        exit(EXIT_FAILURE);
    }

    const char* inputFile  = argv[1];
    const char* outputFile = argv[2];
    int   kernelSize = (argc >= 4) ? atoi(argv[3]) : 5;
    float sigma      = (argc >= 5) ? (float)atof(argv[4]) : 1.0f;

    if (kernelSize < 1 || kernelSize > 9 || (kernelSize % 2) == 0)
    {
        printf("Error: kernel_size must be an odd integer between 1 and 9 (got %d).\n", kernelSize);
        exit(EXIT_FAILURE);
    }
    if (sigma <= 0.0f)
    {
        printf("Error: sigma must be positive (got %f).\n", sigma);
        exit(EXIT_FAILURE);
    }

    /* --- Load input image ---------------------------------------------- */
    tga::TGAImage image;
    if (!tga::LoadTGA(&image, inputFile))
    {
        printf("Error: could not load image '%s'.\n", inputFile);
        exit(EXIT_FAILURE);
    }

    int width          = (int)image.width;
    int height         = (int)image.height;
    int bytesPerPixel  = (int)(image.bpp / 8);   /* 3 = RGB, 4 = RGBA */

    printf("Image loaded: %dx%d, %u bpp\n", width, height, image.bpp);

    /* --- Build the Gaussian filter kernel ------------------------------ */
    float* gaussKernel = static_cast<float*>(malloc(kernelSize * kernelSize * sizeof(float)));
    if (!gaussKernel) { printf("Error: malloc failed.\n"); exit(EXIT_FAILURE); }

    generateGaussianKernel(gaussKernel, kernelSize, sigma);

    printf("Using %dx%d Gaussian kernel (sigma=%.2f).\n", kernelSize, kernelSize, sigma);

    /* --- OpenCL setup -------------------------------------------------- */
    cl_int status;

    /* Platform */
    cl_uint numPlatforms = 0;
    checkStatus(clGetPlatformIDs(0, NULL, &numPlatforms));
    if (numPlatforms == 0)
    {
        printf("Error: no OpenCL platform available.\n");
        exit(EXIT_FAILURE);
    }
    cl_platform_id platform;
    checkStatus(clGetPlatformIDs(1, &platform, NULL));

    /* Device – prefer GPU, fall back to any device */
    cl_device_id device;
    cl_uint numDevices = 0;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (status == CL_SUCCESS && numDevices > 0)
    {
        checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
    }
    else
    {
        checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));
        if (numDevices == 0)
        {
            printf("Error: no OpenCL device available.\n");
            exit(EXIT_FAILURE);
        }
        checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));
    }

    /* --- Query and print device capabilities --------------------------- */
    size_t maxWorkGroupSize = 0;
    checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                sizeof(size_t), &maxWorkGroupSize, NULL));

    cl_uint maxWorkItemDimensions = 0;
    checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                sizeof(cl_uint), &maxWorkItemDimensions, NULL));

    size_t* maxWorkItemSizes = static_cast<size_t*>(malloc(maxWorkItemDimensions * sizeof(size_t)));
    if (!maxWorkItemSizes) { printf("Error: malloc failed.\n"); exit(EXIT_FAILURE); }
    checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                maxWorkItemDimensions * sizeof(size_t), maxWorkItemSizes, NULL));

    printf("Device capabilities:\n");
    printf("  Max work-group size     : %zu\n", maxWorkGroupSize);
    printf("  Max work-item dimensions: %u\n",  maxWorkItemDimensions);
    printf("  Max work-items per dim  :");
    for (cl_uint d = 0; d < maxWorkItemDimensions; d++)
        printf(" [%u]=%zu", d, maxWorkItemSizes[d]);
    printf("\n");

    /* Validate that the device supports a 2D NDRange */
    if (maxWorkItemDimensions < 2)
    {
        printf("Error: device does not support 2D NDRange (only %u dimension(s)).\n",
               maxWorkItemDimensions);
        free(maxWorkItemSizes);
        free(gaussKernel);
        exit(EXIT_FAILURE);
    }

    free(maxWorkItemSizes);

    /* --- Create context and command queue ------------------------------ */
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    checkStatus(status);

    cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &status);
    checkStatus(status);

    /* --- Create device buffers ----------------------------------------- */
    size_t imageDataSize  = (size_t)(width * height * bytesPerPixel);
    size_t filterDataSize = (size_t)(kernelSize * kernelSize) * sizeof(float);

    cl_mem inputBuffer = clCreateBuffer(context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        imageDataSize,
                                        image.imageData.data(),
                                        &status);
    checkStatus(status);

    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imageDataSize, NULL, &status);
    checkStatus(status);

    cl_mem filterBuffer = clCreateBuffer(context,
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         filterDataSize,
                                         gaussKernel,
                                         &status);
    checkStatus(status);

    /* --- Load, compile, and build the OpenCL program ------------------- */
    const char* kernelFileName = "kernel.cl";
    std::ifstream ifs(kernelFileName);
    if (!ifs.good())
    {
        printf("Error: could not open kernel file '%s'.\n", kernelFileName);
        exit(EXIT_FAILURE);
    }
    std::string programSource((std::istreambuf_iterator<char>(ifs)),
                               std::istreambuf_iterator<char>());
    const char* programSourcePtr = programSource.c_str();
    size_t      programSourceLen = programSource.length();

    cl_program program = clCreateProgramWithSource(context, 1,
                                                   &programSourcePtr,
                                                   &programSourceLen,
                                                   &status);
    checkStatus(status);

    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        printCompilerError(program, device);
        exit(EXIT_FAILURE);
    }

    cl_kernel kernel = clCreateKernel(program, "gaussian_blur", &status);
    checkStatus(status);

    /* --- Set kernel arguments ------------------------------------------ */
    checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer));
    checkStatus(clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer));
    checkStatus(clSetKernelArg(kernel, 2, sizeof(cl_int), &width));
    checkStatus(clSetKernelArg(kernel, 3, sizeof(cl_int), &height));
    checkStatus(clSetKernelArg(kernel, 4, sizeof(cl_int), &bytesPerPixel));
    checkStatus(clSetKernelArg(kernel, 5, sizeof(cl_mem), &filterBuffer));
    checkStatus(clSetKernelArg(kernel, 6, sizeof(cl_int), &kernelSize));

    /* --- Enqueue 2D NDRange kernel ------------------------------------- */
    size_t globalWorkSize[2] = { (size_t)width, (size_t)height };

    printf("Launching kernel with global NDRange [%zu x %zu]...\n",
           globalWorkSize[0], globalWorkSize[1]);

    checkStatus(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
                                       globalWorkSize, NULL, 0, NULL, NULL));

    /* --- Read back results --------------------------------------------- */
    unsigned char* outputData = static_cast<unsigned char*>(malloc(imageDataSize));
    if (!outputData) { printf("Error: malloc failed.\n"); exit(EXIT_FAILURE); }

    checkStatus(clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0,
                                    imageDataSize, outputData, 0, NULL, NULL));

    /* --- Save output image --------------------------------------------- */
    tga::TGAImage outputImage;
    outputImage.width  = image.width;
    outputImage.height = image.height;
    outputImage.bpp    = image.bpp;
    outputImage.type   = image.type;
    outputImage.imageData.assign(outputData, outputData + imageDataSize);

    if (!tga::saveTGA(outputImage, outputFile))
    {
        printf("Error: could not save output image '%s'.\n", outputFile);
        exit(EXIT_FAILURE);
    }

    printf("Output image saved: %s\n", outputFile);

    /* --- Release resources --------------------------------------------- */
    free(outputData);
    free(gaussKernel);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(filterBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(inputBuffer);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    return EXIT_SUCCESS;
}