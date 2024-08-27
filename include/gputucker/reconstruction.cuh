#ifndef RECONSTRUCTION_H_
#define RECONSTRUCTION_H_

#include "common/cuda_helper.hpp"
#include "gputucker/constants.hpp"
#include "gputucker/helper.hpp"
#include <cuda_runtime_api.h>
#include <omp.h>

namespace supertensor {
namespace gputucker {
/**
 * @brief CUDA kernel for computing the reconstruction error in Tucker decomposition.
 *
 * This kernel computes the reconstruction error for each block of the tensor during
 * Tucker decomposition. The error is calculated based on the difference between the
 * original tensor values and the values reconstructed from the core tensor and factor
 * matrices.
 *
 * @tparam IndexType The data type used for indices.
 * @tparam ValueType The data type used for values.
 * @param X_indices The indices of the input tensor.
 * @param core_indices The indices of the core tensor.
 * @param core_values The values of the core tensor.
 * @param error_T The array to store the reconstruction error.
 * @param factors The factor matrices used in decomposition.
 * @param order The order (rank) of the tensor.
 * @param rank The Tucker rank.
 * @param nnz_count The number of non-zero elements in the input tensor.
 * @param core_nnz_count The number of non-zero elements in the core tensor.
 */
template <typename IndexType, typename ValueType>
__global__ void ComputingReconstructionKernel(std::uintptr_t *X_indices, std::uintptr_t *core_indices, ValueType *core_values, ValueType *error_T,
                                              std::uintptr_t *factors, const int order, const int rank, uint64_t nnz_count, uint64_t core_nnz_count) {

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

  while (tid < nnz_count) {
    value_t res = 0.0f;

    for (uint64_t i = 0; i < core_nnz_count; ++i) {

      value_t temp = core_values[i];
      for (int axis = 0; axis < order; ++axis) {
        // index_t row = nnz_coo[axis];

        // // index_t row = ((index_t *)X_indices[axis])[tid];
        // // index_t col = ((index_t *)const_core_indices[axis])[i];
        // index_t col = ((index_t *)core_indices[axis])[i];

        // temp *= ((value_t *)(factors[axis]))[row * rank + col];

        temp *= ((value_t *)(sh_factors[axis]))[((index_t *)sh_X_idx_addr[axis])[tid] * sh_rank + ((index_t *)sh_core_idx_addr[axis])[i]];
      }

      res += temp;
    }

    error_T[tid] = res;
    tid += stride;
  }
  __syncthreads();
}

/**
 * @brief Computes the reconstruction error for each block of the tensor.
 *
 * This function computes the reconstruction error for each block of the tensor
 * by launching the CUDA kernel `ComputingReconstructionKernel`. It manages the
 * necessary memory transfers between host and device, and coordinates the computation
 * across multiple CUDA streams.
 *
 * @tparam TensorType The type of the tensor.
 * @tparam MatrixType The type of the factor matrices.
 * @tparam ErrorType The type used for error values.
 * @tparam CudaAgentType The type of the CUDA agent.
 * @tparam SchedulerType The type of the scheduler.
 * @param tensor The input tensor.
 * @param core_tensor The core tensor.
 * @param factor_matrices The factor matrices used in decomposition.
 * @param error_T The array to store the reconstruction error.
 * @param rank The Tucker rank.
 * @param cuda_agent The CUDA agent managing device resources.
 * @param scheduler The scheduler managing computational tasks.
 * @param device_id The ID of the CUDA device to be used.
 */
template <typename TensorType, typename MatrixType, typename ErrorType, typename CudaAgentType, typename SchedulerType>
void ComputingReconstruction(TensorType *tensor, TensorType *core_tensor, MatrixType ***factor_matrices, ErrorType **error_T, int rank,
                             CudaAgentType *cuda_agent, SchedulerType *scheduler, int device_id) {
  using tensor_t = TensorType;
  using block_t = typename tensor_t::block_t;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

  int order = tensor->order;
  uint64_t nnz_count = tensor->nnz_count;
  uint64_t core_nnz_count = core_tensor->nnz_count;

  const int dev_count = cuda_agent->get_device_count();
  auto dev_bufs = cuda_agent->dev_buf;
  auto dev_prof = cuda_agent->get_device_properties();

  const uint64_t block_count = tensor->block_count;
  index_t *block_dims = tensor->block_dims;

  common::cuda::_CUDA_API_CALL(cudaSetDevice(device_id));
  cudaStream_t *streams = static_cast<cudaStream_t *>(cuda_agent->get_cuda_streams());
  unsigned stream_count = cuda_agent->get_stream_count();
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

  double task_time = omp_get_wtime();
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

    double recon_kernel_time = omp_get_wtime();
    gputucker::ComputingReconstructionKernel<index_t, value_t><<<blocks_per_grid, threads_per_block, 0, streams[stream_offset]>>>(
        (std::uintptr_t *)dev_bufs.X_idx_addr.get_ptr(stream_offset), (std::uintptr_t *)dev_bufs.core_idx_addr.get_ptr(0),
        (value_t *)dev_bufs.core_values.get_ptr(0), (value_t *)dev_bufs.delta.get_ptr(stream_offset),
        (std::uintptr_t *)dev_bufs.factor_addr.get_ptr(stream_offset), order, rank, avail_nnz_count, core_tensor->nnz_count);

    // For Delta
    common::cuda::d2hcpy_async(&error_T[block_id][nnz_offset], dev_bufs.delta.get_ptr(stream_offset), sizeof(value_t) * avail_nnz_count,
                               streams[stream_offset]);

  } // task.size()

  for (int stream_offset = 0; stream_offset < stream_count; ++stream_offset) {
    common::cuda::_CUDA_API_CALL(cudaStreamSynchronize(streams[stream_offset]));
  }
}

/**
 * @brief Computes the reconstruction error and fit value for Tucker decomposition.
 *
 * This function computes the reconstruction error for each block of the tensor
 * by using the CUDA-enabled reconstruction computation and then calculates the
 * overall fit value based on the total error and the norm of the original tensor.
 * The function operates in parallel across multiple CUDA devices.
 *
 * @tparam TensorType The type of the tensor.
 * @tparam MatrixType The type of the factor matrices.
 * @tparam ErrorType The type used for error values.
 * @tparam CudaAgentType The type of the CUDA agent.
 * @tparam SchedulerType The type of the scheduler.
 * @param tensor The input tensor.
 * @param core_tensor The core tensor resulting from Tucker decomposition.
 * @param factor_matrices The factor matrices used in the decomposition.
 * @param fit Pointer to a double where the computed fit value will be stored.
 * @param error_T The array to store the reconstruction error for each block.
 * @param rank The Tucker rank.
 * @param device_count The number of CUDA devices available for computation.
 * @param cuda_agents Array of CUDA agents managing device resources.
 * @param scheduler The scheduler managing computational tasks.
 */
template <typename TensorType, typename MatrixType, typename ErrorType, typename CudaAgentType, typename SchedulerType>
void Reconstruction(TensorType *tensor, TensorType *core_tensor, MatrixType ***factor_matrices, double *fit, ErrorType **error_T, int rank,
                    int device_count, CudaAgentType **cuda_agents, SchedulerType *scheduler) {

  MYPRINT("[ Reconstruction ]\n");
  using tensor_t = TensorType;
  using block_t = typename tensor_t::block_t;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

  int order = tensor->order;
  uint64_t nnz_count = tensor->nnz_count;
  uint64_t core_nnz_count = core_tensor->nnz_count;

  uint64_t block_count = tensor->block_count;
  uint64_t max_nnz_count_in_block = tensor->get_max_nnz_count_in_block();
  index_t *block_dims = tensor->block_dims;

  for (unsigned dev_id = 0; dev_id < device_count; ++dev_id) {
    cuda_agents[dev_id]->SetDeviceBuffers(tensor, rank, scheduler->nnz_count_per_task);
  }
  double recons_time = omp_get_wtime();
#pragma omp parallel num_threads(device_count)
  {
    int dev_id = omp_get_thread_num();

    ComputingReconstruction(tensor, core_tensor, factor_matrices, error_T, rank, cuda_agents[dev_id], scheduler, dev_id);

    common::cuda::_CUDA_API_CALL(cudaDeviceSynchronize());
  }

  value_t Error = 0.0f;

  for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
    block_t *curr_block = tensor->blocks[block_id];
#pragma omp prallel for schedule(static) reduction(+ : Error)
    for (uint64_t nnz = 0; nnz < curr_block->nnz_count; ++nnz) {
      value_t err_tmp = curr_block->values[nnz] - error_T[block_id][nnz];
      Error += err_tmp * err_tmp;
    }
  }

  printf("Error:: %1.3f \t Norm:: %1.3f\n", Error, tensor->norm);

  if (tensor->norm == 0) {
    *fit = 1;
  } else {
    *fit = 1.0f - std::sqrt(Error) / tensor->norm;
  }
}
} // namespace gputucker
} // namespace supertensor

#endif /* RECONSTRUCTION_H_ */