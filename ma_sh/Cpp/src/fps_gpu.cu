#include "cuda_utils.h"

#ifdef USE_CUDA
__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    const float *__restrict__ points, const int *__restrict__ point_counts,
    const int *__restrict__ sample_point_nums,
    const int *__restrict__ point_start_idxs,
    const int *__restrict__ sample_point_bound_idxs, float *__restrict__ tmp,
    int *__restrict__ idxs) {
  int batch_index = blockIdx.x;
  int sample_point_num = sample_point_nums[batch_index];

  if (sample_point_num <= 0) {
    return;
  }

  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int start_point_idx = point_start_idxs[batch_index];

  points += start_point_idx * 3;
  tmp += start_point_idx;
  idxs += sample_point_bound_idxs[batch_index];

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0) {
    idxs[0] = old;
  }

  __syncthreads();

  for (int j = 1; j < sample_point_num; ++j) {
    int besti = 0;
    float best = -1;

    float x1 = points[old * 3 + 0];
    float y1 = points[old * 3 + 1];
    float z1 = points[old * 3 + 2];

    for (int k = tid; k < point_counts[batch_index]; k += stride) {
      float x2 = points[k * 3 + 0];
      float y2 = points[k * 3 + 1];
      float z2 = points[k * 3 + 2];

      float d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

      float d2 = min(d, tmp[k]);
      tmp[k] = d2;

      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }

    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 512) {
      if (tid < 256) {
        __update(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];

    if (tid == 0) {
      idxs[j] = old;
    }
  }
}

void furthest_point_sampling_kernel_wrapper(
    const int &point_batch_num, const int &max_point_num,
    const int &max_sample_point_num, const float *points,
    const int *point_counts, const int *sample_point_nums,
    const int *point_start_idxs, const int *sample_point_bound_idxs, float *tmp,
    int *idxs) {
  unsigned int n_threads = opt_n_threads(max_point_num);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (n_threads) {
  case 512: {
    furthest_point_sampling_kernel<512>
        <<<point_batch_num, n_threads, 0, stream>>>(
            points, point_counts, sample_point_nums, point_start_idxs,
            sample_point_bound_idxs, tmp, idxs);
    break;
  }
  case 256: {
    furthest_point_sampling_kernel<256>
        <<<point_batch_num, n_threads, 0, stream>>>(
            points, point_counts, sample_point_nums, point_start_idxs,
            sample_point_bound_idxs, tmp, idxs);
    break;
  }
  case 128: {
    furthest_point_sampling_kernel<128>
        <<<point_batch_num, n_threads, 0, stream>>>(
            points, point_counts, sample_point_nums, point_start_idxs,
            sample_point_bound_idxs, tmp, idxs);
    break;
  }
  case 64: {
    furthest_point_sampling_kernel<64>
        <<<point_batch_num, n_threads, 0, stream>>>(
            points, point_counts, sample_point_nums, point_start_idxs,
            sample_point_bound_idxs, tmp, idxs);
    break;
  }
  case 32: {
    furthest_point_sampling_kernel<32>
        <<<point_batch_num, n_threads, 0, stream>>>(
            points, point_counts, sample_point_nums, point_start_idxs,
            sample_point_bound_idxs, tmp, idxs);
    break;
  }
  case 16: {
    furthest_point_sampling_kernel<16>
        <<<point_batch_num, n_threads, 0, stream>>>(
            points, point_counts, sample_point_nums, point_start_idxs,
            sample_point_bound_idxs, tmp, idxs);
    break;
  }
  case 8: {
    furthest_point_sampling_kernel<8>
        <<<point_batch_num, n_threads, 0, stream>>>(
            points, point_counts, sample_point_nums, point_start_idxs,
            sample_point_bound_idxs, tmp, idxs);
    break;
  }
  case 4: {
    furthest_point_sampling_kernel<4>
        <<<point_batch_num, n_threads, 0, stream>>>(
            points, point_counts, sample_point_nums, point_start_idxs,
            sample_point_bound_idxs, tmp, idxs);
    break;
  }
  case 2: {
    furthest_point_sampling_kernel<2>
        <<<point_batch_num, n_threads, 0, stream>>>(
            points, point_counts, sample_point_nums, point_start_idxs,
            sample_point_bound_idxs, tmp, idxs);
    break;
  }
  case 1: {
    furthest_point_sampling_kernel<1>
        <<<point_batch_num, n_threads, 0, stream>>>(
            points, point_counts, sample_point_nums, point_start_idxs,
            sample_point_bound_idxs, tmp, idxs);
    break;
  }
  default: {
    furthest_point_sampling_kernel<512>
        <<<point_batch_num, n_threads, 0, stream>>>(
            points, point_counts, sample_point_nums, point_start_idxs,
            sample_point_bound_idxs, tmp, idxs);
  }
  }

  // CUDA_CHECK_ERRORS();
}
#endif
