#define THREADS_NUM 512
#define BLOCKS_NUM 68
#define LARGE_NUM 1e38

namespace farthest_sampling
{

__device__ void reduceMaximum(float* __restrict__ candidate_distances, unsigned* __restrict__ candidate_indices)
{
  for (int k = 0; (1 << k) < blockDim.x; ++k)
  {
    __syncthreads();
    if (threadIdx.x < (blockDim.x >> (k + 1)))
    {
      int index1 = (threadIdx.x * 2) << k;
      int index2 = (threadIdx.x * 2 + 1) << k;
      if (candidate_distances[index1] < candidate_distances[index2])
      {
        candidate_distances[index1] = candidate_distances[index2];
        candidate_indices[index1] = candidate_indices[index2];
      }
    }
  }
}

// output_indices : [point_idx, ...] (size = sampling_point_num * batch_size)
__global__ void farthestPointSamplingKernel(unsigned batch_size, unsigned sampling_point_num,
                                            const unsigned* __restrict__ point_nums,
                                            const float* __restrict__ input_point_clouds, float* __restrict__ distances,
                                            unsigned* __restrict__ output_indices)
{
  __shared__ float candidate_distances[THREADS_NUM];
  __shared__ unsigned candidate_indices[THREADS_NUM];
  const unsigned buffer_size = 1024;
  __shared__ float point_buffer[buffer_size * 4];

  for (unsigned point_cloud_idx = blockIdx.x; point_cloud_idx < batch_size; point_cloud_idx += gridDim.x)
  {
    unsigned latest_added_index = 0;
    unsigned point_num = point_nums[point_cloud_idx];

    // Add offset to the input_point_cloud, distances to focus only on current point cloud
    for (unsigned i = 0; i < point_cloud_idx; ++i)
    {
      input_point_clouds += point_nums[i] * 4;
      distances += point_nums[i];
    }

    if (threadIdx.x == 0)
    {
      output_indices[point_cloud_idx * sampling_point_num] = latest_added_index;
    }

    for (unsigned point_idx = threadIdx.x; point_idx < point_num; point_idx += blockDim.x)
    {
      // distances size: total_point_num
      distances[point_idx] = INFINITY;
    }
    for (unsigned point_idx = threadIdx.x; point_idx < min(buffer_size, point_num) * 4; point_idx += blockDim.x)
    {
      // Prefetch partial points into shared memory to accelerate following computation
      point_buffer[point_idx] = input_point_clouds[point_idx];
    }

    __syncthreads();

    // point_idx = 1, since we already have one selected point index = 0
    for (unsigned point_idx = 1; point_idx < sampling_point_num; ++point_idx)
    {
      unsigned best_index = 0;
      float best_distance = -1.0f;
      float3 latest_added_point =
          make_float3(input_point_clouds[latest_added_index * 4 + 0],
                      input_point_clouds[latest_added_index * 4 + 1],
                      input_point_clouds[latest_added_index * 4 + 2]);

      for (unsigned k = threadIdx.x; k < point_num; k += blockDim.x)
      {
        float current_min_distance = distances[k];

        float3 current_point;
        if (k < buffer_size)
        {
          current_point = make_float3(point_buffer[k * 4 + 0], point_buffer[k * 4 + 1], point_buffer[k * 4 + 2]);
        }
        else  // cache missing
        {
          current_point = make_float3(input_point_clouds[k * 4 + 0],
                                      input_point_clouds[k * 4 + 1],
                                      input_point_clouds[k * 4 + 2]);
        }

        float distance1 = norm3df(current_point.x - latest_added_point.x, current_point.y - latest_added_point.y,
                                  current_point.z - latest_added_point.z);

        float distance2 = min(distance1, current_min_distance);
        if (distance2 != current_min_distance)
        {
          distances[k] = distance2;
        }
        if (distance2 > best_distance)
        {
          best_distance = distance2;
          best_index = k;
        }
      }

      candidate_distances[threadIdx.x] = best_distance;
      candidate_indices[threadIdx.x] = best_index;

      reduceMaximum(candidate_distances, candidate_indices);

      __syncthreads();

      latest_added_index = candidate_indices[0];
      if (threadIdx.x == 0)
      {
        output_indices[point_cloud_idx * sampling_point_num + point_idx] = latest_added_index;
      }
    }
  }
}

void farthestPointSamplingLauncher(unsigned batch_size, unsigned sampling_point_num, const unsigned* point_nums,
                                   const float* input_point_clouds, float* distances, unsigned* output_indices)
{
  dim3 grid(min(batch_size, BLOCKS_NUM));
  dim3 block(THREADS_NUM);

  farthestPointSamplingKernel<<<grid, block>>>(batch_size, sampling_point_num, point_nums, input_point_clouds, distances,
                                               output_indices);
}
} // namespace farthest_sampling
