#include <stdio.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void NmDistanceKernel(const int b, const int n, const float *xyz, const int m,
                                 const float *xyz2, float *result,
                                 int *result_i) {
  const int batch = 512;
  __shared__ float buf[batch * 3];
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int k2 = 0; k2 < m; k2 += batch) {
      int end_k = min(m, k2 + batch) - k2;
      for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
        buf[j] = xyz2[(i * m + k2) * 3 + j];
      }
      __syncthreads();
      for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n;
           j += blockDim.x * gridDim.y) {
        float x1 = xyz[(i * n + j) * 3 + 0];
        float y1 = xyz[(i * n + j) * 3 + 1];
        float z1 = xyz[(i * n + j) * 3 + 2];
        int best_i = 0;
        float best = 0;
        int end_ka = end_k - (end_k & 3);
        if (end_ka == batch) {
          for (int k = 0; k < batch; k += 4) {
            {
              float x2 = buf[k * 3 + 0] - x1;
              float y2 = buf[k * 3 + 1] - y1;
              float z2 = buf[k * 3 + 2] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (k == 0 || d < best) {
                best = d;
                best_i = k + k2;
              }
            }
            {
              float x2 = buf[k * 3 + 3] - x1;
              float y2 = buf[k * 3 + 4] - y1;
              float z2 = buf[k * 3 + 5] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 1;
              }
            }
            {
              float x2 = buf[k * 3 + 6] - x1;
              float y2 = buf[k * 3 + 7] - y1;
              float z2 = buf[k * 3 + 8] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 2;
              }
            }
            {
              float x2 = buf[k * 3 + 9] - x1;
              float y2 = buf[k * 3 + 10] - y1;
              float z2 = buf[k * 3 + 11] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 3;
              }
            }
          }
        } else {
          for (int k = 0; k < end_ka; k += 4) {
            {
              float x2 = buf[k * 3 + 0] - x1;
              float y2 = buf[k * 3 + 1] - y1;
              float z2 = buf[k * 3 + 2] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (k == 0 || d < best) {
                best = d;
                best_i = k + k2;
              }
            }
            {
              float x2 = buf[k * 3 + 3] - x1;
              float y2 = buf[k * 3 + 4] - y1;
              float z2 = buf[k * 3 + 5] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 1;
              }
            }
            {
              float x2 = buf[k * 3 + 6] - x1;
              float y2 = buf[k * 3 + 7] - y1;
              float z2 = buf[k * 3 + 8] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 2;
              }
            }
            {
              float x2 = buf[k * 3 + 9] - x1;
              float y2 = buf[k * 3 + 10] - y1;
              float z2 = buf[k * 3 + 11] - z1;
              float d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 3;
              }
            }
          }
        }
        for (int k = end_ka; k < end_k; k++) {
          float x2 = buf[k * 3 + 0] - x1;
          float y2 = buf[k * 3 + 1] - y1;
          float z2 = buf[k * 3 + 2] - z1;
          float d = x2 * x2 + y2 * y2 + z2 * z2;
          if (k == 0 || d < best) {
            best = d;
            best_i = k + k2;
          }
        }
        if (k2 == 0 || result[(i * n + j)] > best) {
          result[(i * n + j)] = best;
          result_i[(i * n + j)] = best_i;
        }
      }
      __syncthreads();
    }
  }
}

__global__ void NmDistanceGradKernel(const int b,const  int n, const float *xyz1,const  int m,
                                     const float *xyz2, const float *grad_dist1,
                                     const int *idx1, float *grad_xyz1,
                                     float *grad_xyz2) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n;
         j += blockDim.x * gridDim.y) {
      float x1 = xyz1[(i * n + j) * 3 + 0];
      float y1 = xyz1[(i * n + j) * 3 + 1];
      float z1 = xyz1[(i * n + j) * 3 + 2];
      int j2 = idx1[i * n + j];
      float x2 = xyz2[(i * m + j2) * 3 + 0];
      float y2 = xyz2[(i * m + j2) * 3 + 1];
      float z2 = xyz2[(i * m + j2) * 3 + 2];
      float g = grad_dist1[i * n + j] * 2;
      atomicAdd(&(grad_xyz1[(i * n + j) * 3 + 0]), g * (x1 - x2));
      atomicAdd(&(grad_xyz1[(i * n + j) * 3 + 1]), g * (y1 - y2));
      atomicAdd(&(grad_xyz1[(i * n + j) * 3 + 2]), g * (z1 - z2));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 3 + 0]), -(g * (x1 - x2)));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 3 + 1]), -(g * (y1 - y2)));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 3 + 2]), -(g * (z1 - z2)));
    }
  }
}

// int chamfer_cuda_forward(int b,int n,const float * xyz,int m,const float *
// xyz2,float * result,int * result_i,float * result2,int * result2_i,
// cudaStream_t stream){
int chamfer_cuda_forward(const torch::Tensor &xyz1,const  torch::Tensor &xyz2, torch::Tensor &dist1,
                         torch::Tensor &dist2, torch::Tensor &idx1, torch::Tensor &idx2) {

  const auto batch_size = xyz1.size(0);
  const auto n = xyz1.size(1); // num_points point cloud A
  const auto m = xyz2.size(1); // num_points point cloud B

  NmDistanceKernel<<<dim3(32, 16, 1), 512>>>(
      batch_size, n, xyz1.data_ptr<float>(), m, xyz2.data_ptr<float>(),
      dist1.data_ptr<float>(), idx1.data_ptr<int>());
  NmDistanceKernel<<<dim3(32, 16, 1), 512>>>(
      batch_size, m, xyz2.data_ptr<float>(), n, xyz1.data_ptr<float>(),
      dist2.data_ptr<float>(), idx2.data_ptr<int>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in nnd updateOutput: %s\n", cudaGetErrorString(err));
    // THError("aborting");
    return 0;
  }
  return 1;
}

// int chamfer_cuda_backward(int b,int n,const float * xyz1,int m,const float *
// xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const
// int * idx2,float * grad_xyz1,float * grad_xyz2, cudaStream_t stream){
int chamfer_cuda_backward(const torch::Tensor &xyz1,const  torch::Tensor &xyz2, 
                          const torch::Tensor &graddist1,
                          const torch::Tensor &graddist2,const  torch::Tensor &idx1,
                          const torch::Tensor &idx2, torch::Tensor &gradxyz1, torch::Tensor &gradxyz2) {
  // cudaMemset(grad_xyz1,0,b*n*3*4);
  // cudaMemset(grad_xyz2,0,b*m*3*4);

  const auto batch_size = xyz1.size(0);
  const auto n = xyz1.size(1); // num_points point cloud A
  const auto m = xyz2.size(1); // num_points point cloud B

  NmDistanceGradKernel<<<dim3(1, 16, 1), 256>>>(
      batch_size, n, xyz1.data_ptr<float>(), m, xyz2.data_ptr<float>(),
      graddist1.data_ptr<float>(), idx1.data_ptr<int>(), gradxyz1.data_ptr<float>(),
      gradxyz2.data_ptr<float>());
  NmDistanceGradKernel<<<dim3(1, 16, 1), 256>>>(
      batch_size, m, xyz2.data_ptr<float>(), n, xyz1.data_ptr<float>(),
      graddist2.data_ptr<float>(), idx2.data_ptr<int>(), gradxyz2.data_ptr<float>(),
      gradxyz1.data_ptr<float>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in nnd get grad: %s\n", cudaGetErrorString(err));
    // THError("aborting");
    return 0;
  }
  return 1;
}
