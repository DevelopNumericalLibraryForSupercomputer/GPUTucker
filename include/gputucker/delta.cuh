#ifndef DELTA_CUH_
#define DELTA_CUH_

#include <cuda_runtime_api.h>

#include "common/cuda_helper.hpp"
#include "gputucker/helper.hpp"

namespace supertensor {
namespace gputucker {

/**
 * @brief CUDA kernel function for computing delta in Tucker decomposition.
 *
 * This CUDA kernel function computes the delta values used in Tucker decomposition. The delta
 * values are calculated based on the input tensor, core tensor, and factor matrices.
 *
 * @tparam IndexType The data type used for indices.
 * @tparam ValueType The data type used for values.
 * @param X_indices The indices of the input tensor.
 * @param X_values The values of the input tensor.
 * @param core_indices The indices of the core tensor.
 * @param core_values The values of the core tensor.
 * @param delta Array to store the computed delta values.
 * @param factors The factor matrices used in decomposition.
 * @param order The order (dimension) of the tensor.
 * @param rank The Tucker rank.
 * @param curr_factor_id The ID of the current factor matrix.
 * @param nnz_count The number of non-zero elements in the input tensor.
 * @param core_nnz_count The number of non-zero elements in the core tensor.
 */

template <typename IndexType, typename ValueType>
__global__ void computing_delta_kernel(std::uintptr_t *X_indices, ValueType *X_values, std::uintptr_t *core_indices, ValueType *core_values,
                                       ValueType *delta, std::uintptr_t *factors, const int order, const int rank, int curr_factor_id,
                                       uint64_t nnz_count, uint64_t core_nnz_count) {
  using index_t = IndexType;
  using value_t = ValueType;

  uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x; // per #NNZs
  uint64_t stride = blockDim.x * gridDim.x;

  __shared__ int sh_rank;
  __shared__ std::uintptr_t *sh_X_idx_addr[gputucker::constants::kMaxOrder];
  __shared__ std::uintptr_t *sh_core_idx_addr[gputucker::constants::kMaxOrder];
  __shared__ std::uintptr_t *sh_factors[gputucker::constants::kMaxOrder];

  if (threadIdx.x == 0) {
    sh_rank = rank;
    for (int axis = 0; axis < order; ++axis) {
      sh_X_idx_addr[axis] = reinterpret_cast<std::uintptr_t *>(X_indices[axis]);
      sh_core_idx_addr[axis] = reinterpret_cast<std::uintptr_t *>(core_indices[axis]);
      sh_factors[axis] = reinterpret_cast<std::uintptr_t *>(factors[axis]);
    }
  }
  __syncthreads();

  // tid == row for the delta
  int r, axis;
  uint64_t i;

  while (tid < nnz_count) {
    value_t tmp[50];
    index_t nnz[gputucker::constants::kMaxOrder];
    for (r = 0; r < sh_rank; ++r) {
      tmp[r] = 0.0f;
    }

    for (axis = 0; axis < order; ++axis) {
      nnz[axis] = ((index_t *)sh_X_idx_addr[axis])[tid];
    }
    for (i = 0; i < core_nnz_count; ++i) {
      index_t delta_col = ((index_t *)core_indices[curr_factor_id])[i];
      value_t beta = core_values[i];
      for (axis = 0; axis < order; ++axis) {
        // int axis = (curr_factor_id + 1 + iter + order) % order;

        if (axis != curr_factor_id) {
          // index_t row = ((index_t *)X_indices[axis])[tid];
          // index_t col = ((index_t *)core_indices[axis])[i];
          beta *= ((value_t *)(sh_factors[axis]))[nnz[axis] * sh_rank + ((index_t *)sh_core_idx_addr[axis])[i]];
        }
      }
      tmp[delta_col] += beta;
    }
    for (r = 0; r < sh_rank; ++r) {
      delta[tid * sh_rank + r] = tmp[r];
    }
    tid += stride;
  }
  __syncthreads();
}

/**
 * @brief Computes delta values for Tucker decomposition.
 *
 * This function orchestrates the computation of delta values for Tucker decomposition
 * by setting up the necessary GPU memory, launching the CUDA kernel, and managing
 * memory transfers between host and device.
 *
 * @tparam TensorType The type of the tensor.
 * @tparam MatrixType The type of the factor matrices.
 * @tparam DeltaType The type used for delta values.
 * @tparam CudaAgentType The type of the CUDA agent.
 * @tparam SchedulerType The type of the scheduler.
 * @param tensor The input tensor.
 * @param core_tensor The core tensor.
 * @param factor_matrices The factor matrices used in decomposition.
 * @param delta Array to store the computed delta values.
 * @param curr_factor_id The ID of the current factor matrix.
 * @param rank The Tucker rank.
 * @param cuda_agent Pointer to the CUDA agent managing device resources.
 * @param scheduler Pointer to the scheduler managing computational tasks.
 * @param device_id The ID of the CUDA device to be used.
 */
template <typename TensorType, typename MatrixType, typename DeltaType, typename CudaAgentType, typename SchedulerType>
void ComputingDelta(TensorType *tensor, TensorType *core_tensor, MatrixType ***factor_matrices, DeltaType **delta, int curr_factor_id, int rank,
                    CudaAgentType *cuda_agent, SchedulerType *scheduler, int device_id) {

  using tensor_t = TensorType;
  using block_t = typename tensor_t::block_t;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

  auto dev_bufs = cuda_agent->dev_buf;

  const int order = tensor->order;
  const uint64_t block_count = tensor->block_count;
  index_t *block_dims = tensor->block_dims;

  int dev_count = cuda_agent->get_device_count();

  common::cuda::_CUDA_API_CALL(cudaSetDevice(device_id));
  cudaStream_t *streams = static_cast<cudaStream_t *>(cuda_agent->get_cuda_streams());
  int stream_count = cuda_agent->get_stream_count();

  auto dev_prof = cuda_agent->get_device_properties();
  const int max_grid_size = dev_prof->maxGridSize[0] / stream_count;

  // Set GPU device memory address
  std::uintptr_t ***h_X_idx_addr = static_cast<std::uintptr_t ***>(common::cuda::pinned_malloc(sizeof(std::uintptr_t **) * stream_count));
  std::uintptr_t ***h_fact_addr = static_cast<std::uintptr_t ***>(common::cuda::pinned_malloc(sizeof(std::uintptr_t **) * stream_count));
  std::uintptr_t **h_core_idx_addr =
      static_cast<std::uintptr_t **>(common::cuda::pinned_malloc(sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder));

  for (int i = 0; i < stream_count; ++i) {
    h_X_idx_addr[i] = static_cast<std::uintptr_t **>(common::cuda::pinned_malloc(sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder));
    h_fact_addr[i] = static_cast<std::uintptr_t **>(common::cuda::pinned_malloc(sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder));
  }

  // Pre-transfer for core tensor
  for (int axis = 0; axis < order; ++axis) {
    h_core_idx_addr[axis] = reinterpret_cast<std::uintptr_t *>(dev_bufs.core_indices[axis].get_ptr(0));
  }
  common::cuda::h2dcpy(dev_bufs.core_idx_addr.get_ptr(0), h_core_idx_addr, sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder);

  for (int axis = 0; axis < order; ++axis) {
    common::cuda::h2dcpy(dev_bufs.core_indices[axis].get_ptr(0), core_tensor->blocks[0]->indices[axis], sizeof(index_t) * core_tensor->nnz_count);
  }
  common::cuda::h2dcpy(dev_bufs.core_values.get_ptr(0), core_tensor->blocks[0]->values, sizeof(value_t) * core_tensor->nnz_count);

  // Computing Blocks
  auto tasks = scheduler->tasks[device_id];

  for (uint64_t iter = 0; iter < tasks.size(); ++iter) {
    uint64_t block_id = tasks[iter].block_id;
    uint64_t avail_nnz_count = tasks[iter].nnz_count;
    uint64_t nnz_offset = tasks[iter].offset;
    int stream_offset = tasks[iter].stream_offset;

    block_t *curr_block = tensor->blocks[block_id];
    index_t *curr_block_coord = curr_block->get_block_coord();

    for (int axis = 0; axis < order; ++axis) {
      h_X_idx_addr[stream_offset][axis] = reinterpret_cast<std::uintptr_t *>(dev_bufs.X_indices[axis].get_ptr(stream_offset));
      common::cuda::h2dcpy_async(dev_bufs.X_indices[axis].get_ptr(stream_offset), &curr_block->indices[axis][nnz_offset],
                                 sizeof(index_t) * avail_nnz_count, streams[stream_offset]);
      // For factor matrices
      h_fact_addr[stream_offset][axis] = reinterpret_cast<std::uintptr_t *>(dev_bufs.factors[axis].get_ptr(stream_offset));
      common::cuda::h2dcpy_async(dev_bufs.factors[axis].get_ptr(stream_offset), factor_matrices[axis][curr_block_coord[axis]],
                                 sizeof(value_t) * block_dims[axis] * rank, streams[stream_offset]);
    }

    common::cuda::h2dcpy_async(dev_bufs.X_idx_addr.get_ptr(stream_offset), h_X_idx_addr[stream_offset],
                               sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder, streams[stream_offset]);
    common::cuda::h2dcpy_async(dev_bufs.factor_addr.get_ptr(stream_offset), h_fact_addr[stream_offset],
                               sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder, streams[stream_offset]);

    // For X_value
    common::cuda::h2dcpy_async(dev_bufs.X_values.get_ptr(stream_offset), &curr_block->values[nnz_offset], sizeof(value_t) * avail_nnz_count,
                               streams[stream_offset]);

    index_t block_size = 1024;
    index_t grid_size = min(max_grid_size, max(1, (int)((avail_nnz_count + block_size - 1) / block_size)));

    dim3 blocks_per_grid(grid_size, 1, 1);
    dim3 threads_per_block(block_size, 1, 1);

    gputucker::computing_delta_kernel<index_t, value_t><<<blocks_per_grid, threads_per_block, 0, streams[stream_offset]>>>(
        (std::uintptr_t *)dev_bufs.X_idx_addr.get_ptr(stream_offset), (value_t *)dev_bufs.X_values.get_ptr(stream_offset),
        (std::uintptr_t *)dev_bufs.core_idx_addr.get_ptr(0), (value_t *)dev_bufs.core_values.get_ptr(0),
        (value_t *)dev_bufs.delta.get_ptr(stream_offset), (std::uintptr_t *)dev_bufs.factor_addr.get_ptr(stream_offset), order, rank, curr_factor_id,
        avail_nnz_count, core_tensor->nnz_count);

    // For Delta
    common::cuda::d2hcpy_async(&delta[block_id][nnz_offset * rank], dev_bufs.delta.get_ptr(stream_offset), sizeof(value_t) * avail_nnz_count * rank,
                               streams[stream_offset]);

  } // block_count

  for (int stream_offset = 0; stream_offset < stream_count; ++stream_offset) {
    common::cuda::_CUDA_API_CALL(cudaStreamSynchronize(streams[stream_offset]));
  }
}

} // namespace gputucker
} // namespace supertensor

#endif /* DELTA_CUH_ */