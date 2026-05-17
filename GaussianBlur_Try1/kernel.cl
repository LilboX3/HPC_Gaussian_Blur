/*
 * Gaussian blur kernel (simple single-pass 2D convolution).
 * Each work item computes exactly one output pixel.
 *
 * Arguments:
 *   input       - source image (row-major, bytesPerPixel channels)
 *   output      - destination image (same layout)
 *   width       - image width in pixels
 *   height      - image height in pixels
 *   bpp         - bytes per pixel (3 = RGB, 4 = RGBA)
 *   filterKernel- 1D array storing the (kernelSize x kernelSize) filter weights in row-major order
 *   kernelSize  - side length of the square filter (must be odd, 1..9)
 */
__kernel void gaussian_blur(
    __global const uchar* input,
    __global       uchar* output,
    int width,
    int height,
    int bpp,
    __global const float* filterKernel,
    int kernelSize)
{
    int x = (int)get_global_id(0);  /* column */
    int y = (int)get_global_id(1);  /* row    */

    if (x >= width || y >= height)
        return;

    int half = kernelSize / 2;

    /* Apply the 2D filter independently for each channel */
    for (int c = 0; c < bpp; c++)
    {
        float acc = 0.0f;

        for (int ky = 0; ky < kernelSize; ky++)
        {
            for (int kx = 0; kx < kernelSize; kx++)
            {
                /* Clamp sample coordinates to the nearest valid pixel */
                int sx = clamp(x + kx - half, 0, width  - 1);
                int sy = clamp(y + ky - half, 0, height - 1);

                float pixelValue = (float)input[(sy * width + sx) * bpp + c];
                acc += pixelValue * filterKernel[ky * kernelSize + kx];
            }
        }

        output[(y * width + x) * bpp + c] = (uchar)clamp(acc, 0.0f, 255.0f);
    }
}
