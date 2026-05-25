__kernel void gaussian_blur(
	__global const uchar* input,
	__global uchar* output,
	__global const float* filter,
	const uint width,
	const uint height,
	const uint channels,
	const uint filterSize,
	const uint isHorizontal,
	__local uchar* cachedPixels)
{
	const uint x = get_global_id(0);
	const uint y = get_global_id(1);
	const uint localX = get_local_id(0);
	const uint localY = get_local_id(1);
	const uint localCoordinate = isHorizontal != 0 ? localX : localY;
	const uint groupLength = isHorizontal != 0 ? get_local_size(0) : get_local_size(1);
	const int radius = (int)(filterSize / 2);

	if (x >= width || y >= height)
	{
		return;
	}

	const uint outputIndex = (y * width + x) * channels;

	for (uint channel = 0; channel < channels; ++channel)
	{
		cachedPixels[localCoordinate * channels + channel] = input[outputIndex + channel];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (uint channel = 0; channel < channels; ++channel)
	{
		float sum = 0.0f;

		for (uint tap = 0; tap < filterSize; ++tap)
		{
			const int sampleCoordinate = clamp((int)localCoordinate + (int)tap - radius, 0, (int)groupLength - 1);
			sum += (float)cachedPixels[(uint)sampleCoordinate * channels + channel] * filter[tap];
		}

		output[outputIndex + channel] = (uchar)clamp((int)(sum + 0.5f), 0, 255);
	}
}
