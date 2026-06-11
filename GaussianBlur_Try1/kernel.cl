// Two-pass separable Gaussian blur using local memory.
// Called twice with different direction values:
//   direction == 0: horizontal (row-wise),   work group size: (width, 1)
//   direction == 1: vertical (column-wise),  work group size: (1, height)
// Both input and output are float buffers; uchar conversion is done on the host.
__kernel void gaussian_blur(
	__global const float* input,
	__global float* output,
	__global const float* filter,
	const uint width,
	const uint height,
	const uint channels,
	const uint filterSize,
	const uint direction,
	__local float* localData)
{
	const uint x      = get_global_id(0);
	const uint y      = get_global_id(1);
	const uint outIdx = (y * width + x) * channels;
	const int radius  = (int)(filterSize / 2);

	if (x >= width || y >= height)
		return;

	if (direction == 0)
	{
		// Horizontal pass: cache one full image row in local memory
		const uint lx      = get_local_id(0);
		const uint locSize = get_local_size(0);

		for (uint c = 0; c < channels; ++c)
			localData[lx * channels + c] = input[outIdx + c];
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint c = 0; c < channels; ++c)
		{
			float sum = 0.0f;
			for (int fx = 0; fx < (int)filterSize; ++fx)
			{
				int sx = (int)lx + fx - radius;
				float val;
				if (sx >= 0 && sx < (int)locSize)
					val = localData[(uint)sx * channels + c];
				else
				{
					int gx = clamp((int)x + fx - radius, 0, (int)width - 1);
					val = input[(y * width + (uint)gx) * channels + c];
				}
				sum += val * filter[fx];
			}
			output[outIdx + c] = sum;
		}
	}
	else
	{
		// Vertical pass: cache one full image column in local memory
		const uint ly      = get_local_id(1);
		const uint locSize = get_local_size(1);

		for (uint c = 0; c < channels; ++c)
			localData[ly * channels + c] = input[outIdx + c];
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint c = 0; c < channels; ++c)
		{
			float sum = 0.0f;
			for (int fy = 0; fy < (int)filterSize; ++fy)
			{
				int sy = (int)ly + fy - radius;
				float val;
				if (sy >= 0 && sy < (int)locSize)
					val = localData[(uint)sy * channels + c];
				else
				{
					int gy = clamp((int)y + fy - radius, 0, (int)height - 1);
					val = input[((uint)gy * width + x) * channels + c];
				}
				sum += val * filter[fy];
			}
			output[outIdx + c] = sum;
		}
	}
}
