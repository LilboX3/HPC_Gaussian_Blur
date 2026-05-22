__kernel void gaussian_blur(
	__global const uchar* input,
	__global uchar* output,
	__global const float* filter,
	const uint width,
	const uint height,
	const uint channels,
	const uint filterSize)
{
	const uint x = get_global_id(0);
	const uint y = get_global_id(1);
	const int radius = (int)(filterSize / 2);

	if (x >= width || y >= height)
	{
		return;
	}

	const uint outputIndex = (y * width + x) * channels;

	for (uint channel = 0; channel < channels; ++channel)
	{
		float sum = 0.0f;

		for (uint filterY = 0; filterY < filterSize; ++filterY)
		{
			for (uint filterX = 0; filterX < filterSize; ++filterX)
			{
				const int sampleX = clamp((int)x + (int)filterX - radius, 0, (int)width - 1);
				const int sampleY = clamp((int)y + (int)filterY - radius, 0, (int)height - 1);
				const uint sampleIndex = ((uint)sampleY * width + (uint)sampleX) * channels + channel;
				const uint filterIndex = filterY * filterSize + filterX;
				sum += (float)input[sampleIndex] * filter[filterIndex];
			}
		}

		output[outputIndex + channel] = (uchar)clamp((int)(sum + 0.5f), 0, 255);
	}
}
