/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda_fp8.h>
#include <transformer_engine/multi_tensor.h>
#include <transformer_engine/transformer_engine.h>

#include "../utils.cuh"
#include "multi_tensor_apply.cuh"

namespace transformer_engine {
namespace multi_tensor_count_nonzero {

#define BLOCK_SIZE 512
#define ILP 4

template <typename T>
__device__ __forceinline__ bool is_aligned(T *p) {
  return ((uint64_t)p) % (ILP * sizeof(T)) == 0;
}

template <typename T>
__device__ __forceinline__ void load_store(T *dst, T *src, int dst_offset, int src_offset) {
  typedef typename std::aligned_storage<ILP * sizeof(T), ILP * alignof(T)>::type LT;
  ((LT *)dst)[dst_offset] = ((LT *)src)[src_offset];  // NOLINT(*)
}

// Sums the per-thread integer values across a block into thread 0.
// Per-block counts are bounded by chunk_size, so an int accumulator is always safe here.
__device__ __forceinline__ int block_sum(int *s, int val) {
  s[threadIdx.x] = val;
  __syncthreads();
  for (int i = BLOCK_SIZE >> 1; i > 0; i >>= 1) {
    if (threadIdx.x < i) s[threadIdx.x] += s[threadIdx.x + i];
    __syncthreads();
  }
  return s[0];
}

template <typename x_t>
struct CountNonzeroFunctor {
  __device__ __forceinline__ void operator()(int chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<1> &tl,  // NOLINT(*)
                                             int64_t *output) {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t *x = reinterpret_cast<x_t *>(tl.addresses[0][tensor_loc]);
    x += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    __shared__ int s_vals[BLOCK_SIZE];

    int count = 0;
    x_t r_x[ILP];

    // Casting to float and comparing against 0 matches torch.count_nonzero across dtypes:
    // NaN is counted as nonzero, and -0.0 == 0.0 is counted as zero. The comparison is exact
    // (we only test equality with zero), so the result is exact regardless of input dtype.

    // to make things simple, we put the aligned case in a different code path
    if (n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x)) {
      for (int i_start = threadIdx.x; i_start * ILP < n && i_start * ILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          if (static_cast<float>(r_x[ii]) != 0.0f) count++;
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            if (static_cast<float>(x[i]) != 0.0f) count++;
          }
        }
      }
    }

    int block_count = block_sum(s_vals, count);

    // The multi-tensor harness reuses the fixed-size `output` buffer across multiple kernel
    // launches, so we accumulate into it (mirrors multi_tensor_l2norm's `output[blockIdx.x] +=`).
    if (threadIdx.x == 0) output[blockIdx.x] += static_cast<int64_t>(block_count);
  }
};

// Reduces the (at most depth_to_max_blocks[0] == 320) per-block partial counts into the
// single int64 return value. 320 matches the `output` buffer size allocated in the binding.
__global__ void cleanup(const int64_t *output, int64_t *ret) {
  __shared__ int64_t vals[BLOCK_SIZE];

  int tid = threadIdx.x;
  vals[tid] = (tid < 320) ? output[tid] : static_cast<int64_t>(0);
  __syncthreads();

  for (int i = BLOCK_SIZE >> 1; i > 0; i >>= 1) {
    if (tid < i) vals[tid] += vals[tid + i];
    __syncthreads();
  }

  if (tid == 0) *ret = vals[0];
}

void multi_tensor_count_nonzero_cuda(int chunk_size, Tensor noop_flag,
                                     std::vector<std::vector<Tensor *>> tensor_lists, Tensor output,
                                     Tensor ret, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      tensor_lists[0][0]->dtype(), dtype,
      multi_tensor_apply<1>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                            CountNonzeroFunctor<dtype>(), stream,
                            reinterpret_cast<int64_t *>(output.data.dptr));)

  NVTE_CHECK_CUDA(cudaGetLastError());

  // One small kernel to reduce the per-block partials; negligible end to end.
  cleanup<<<1, BLOCK_SIZE, 0, stream>>>(reinterpret_cast<int64_t *>(output.data.dptr),
                                        reinterpret_cast<int64_t *>(ret.data.dptr));
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace multi_tensor_count_nonzero
}  // namespace transformer_engine

void nvte_multi_tensor_count_nonzero_cuda(int chunk_size, NVTETensor noop_flag,
                                          NVTETensor **tensor_lists, const size_t num_tensor_lists,
                                          const size_t num_tensors_per_list, NVTETensor output,
                                          NVTETensor ret, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_count_nonzero_cuda);
  using namespace transformer_engine;

  multi_tensor_count_nonzero::multi_tensor_count_nonzero_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list),
      *convertNVTETensorCheck(output), *convertNVTETensorCheck(ret), stream);
}
