#ifndef UPDATE_CUH_
#define UPDATE_CUH_

#include <omp.h>

#include <Eigen/Dense>
#include <cuda_runtime_api.h>

#include "common/cuda_helper.hpp"
#include "gputucker/constants.hpp"
#include "gputucker/delta.cuh"
#include "gputucker/helper.hpp"

namespace supertensor {
namespace gputucker {

/**
 * @brief Computes matrix B and vector C for factor matrix update.
 *
 * This function computes the matrix B and vector C needed to update the factor matrices
 * in the Tucker decomposition. The computation is performed for the specified factor matrix
 * identified by `curr_factor_id`.
 *
 * @tparam TensorType The type of the tensor being decomposed.
 * @tparam MatrixType The type used for matrices in the decomposition.
 * @tparam DeltaType The type used for delta values in the decomposition.
 *
 * @param tensor Pointer to the input tensor.
 * @param delta Array of delta values used in the decomposition.
 * @param B Array of matrices B to be computed.
 * @param C Array of vectors C to be computed.
 * @param curr_factor_id The ID of the current factor matrix being updated.
 * @param rank The Tucker rank.
 */
template <typename TensorType, typename MatrixType, typename DeltaType>
void ComputingBC(TensorType *tensor, DeltaType **delta, MatrixType **B, MatrixType **C, int curr_factor_id, int rank) {

  using tensor_t = TensorType;
  using block_t = typename tensor_t::block_t;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

  const uint64_t block_count = tensor->block_count;

  index_t *block_dims = tensor->block_dims;
  index_t *part_dims = tensor->partition_dims;

  const index_t row_count = block_dims[curr_factor_id];

  // Initialize B and C
  int k, l;
  for (uint64_t part_id = 0; part_id < part_dims[curr_factor_id]; ++part_id) {
#pragma omp parallel for schedule(static)
    for (index_t row = 0; row < row_count; ++row) {
      uint64_t pos_B = row * rank * rank;
      uint64_t pos_C = row * rank;
      for (k = 0; k < rank; ++k) {
        for (l = 0; l < rank; ++l) {
          B[part_id][pos_B] = 0.0f;
          if (k == l) {
            B[part_id][pos_B] = gputucker::constants::kLambda;
          }
          ++pos_B;
        }
        C[part_id][pos_C] = 0.0f;
        ++pos_C;
      }
    } // !omp parallel
  } // !part_dims

  index_t ii, jj;
  uint64_t kk;
  for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
    block_t *curr_block = tensor->blocks[block_id];
    index_t *curr_block_coord = curr_block->get_block_coord();
    index_t part_id = curr_block_coord[curr_factor_id];
    assert(part_id < part_dims[curr_factor_id]);

#pragma omp parallel for schedule(dynamic) // schedule(auto)
    for (index_t row = 0; row < row_count; ++row) {
      uint64_t nnz = curr_block->count_nnz[curr_factor_id][row + 1] - curr_block->count_nnz[curr_factor_id][row];
      index_t where_ptr = curr_block->count_nnz[curr_factor_id][row];
      for (kk = 0; kk < nnz; ++kk) {
        index_t pos_curr_entry = curr_block->where_nnz[curr_factor_id][where_ptr + kk];
        value_t curr_entry_val = curr_block->values[pos_curr_entry];

        uint64_t pos_delta = pos_curr_entry * rank;
        uint64_t pos_B = row * rank * rank;
        uint64_t pos_C = row * rank;

        for (ii = 0; ii < rank; ++ii) {
          value_t cach = delta[block_id][pos_delta + ii];
          for (jj = 0; jj < rank; ++jj) {
            B[part_id][pos_B++] += cach * delta[block_id][pos_delta + jj];
          }
          C[part_id][pos_C++] += cach * curr_entry_val;
        }
      }
    }
  }
}

/**
 * @brief Updates the factor matrices in Tucker decomposition.
 *
 * This function updates the factor matrices used in Tucker decomposition by first
 * computing the necessary matrices B and vectors C, and then using them to update
 * each factor matrix. The updates are performed for each dimension of the tensor
 * in a loop.
 *
 * @tparam TensorType The type of the tensor being decomposed.
 * @tparam MatrixType The type used for matrices in the decomposition.
 * @tparam ValueType The type used for values in the decomposition.
 * @tparam CudaAgentType The type of CUDA agent managing GPU resources.
 * @tparam SchedulerType The type of scheduler managing the decomposition tasks.
 *
 * @param tensor Pointer to the input tensor.
 * @param core_tensor Pointer to the core tensor.
 * @param factor_matrices Array of pointers to the factor matrices to be updated.
 * @param delta Array of delta values used in the decomposition.
 * @param B Array of matrices B computed during the update.
 * @param C Array of vectors C computed during the update.
 * @param rank The Tucker rank.
 * @param device_count The number of devices available for computation.
 * @param cuda_agents Array of pointers to CUDA agents managing the GPU resources.
 * @param scheduler Pointer to the scheduler managing the decomposition tasks.
 */
template <typename TensorType, typename MatrixType, typename ValueType, typename CudaAgentType, typename SchedulerType>
void UpdateFactorMatrices(TensorType *tensor, TensorType *core_tensor, ValueType ***factor_matrices, ValueType **delta, MatrixType **B,
                          MatrixType **C, int rank, int device_count, CudaAgentType **cuda_agents, SchedulerType *scheduler) {
  using tensor_t = TensorType;
  using block_t = typename tensor_t::block_t;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;
  using matrix_t = Eigen::MatrixXd;

  int order = tensor->order;

  index_t *block_dims = tensor->block_dims;
  index_t *part_dims = tensor->partition_dims;

  for (unsigned dev_id = 0; dev_id < device_count; ++dev_id) {
    cuda_agents[dev_id]->SetDeviceBuffers(tensor, rank, scheduler->nnz_count_per_task);
  }

  for (int curr_factor_id = 0; curr_factor_id < order; ++curr_factor_id) {
    MYPRINT("[ Update factor matrix %d ]\n", curr_factor_id);

    double delta_time = omp_get_wtime();
#pragma omp parallel num_threads(device_count)
    {
      int device_id = omp_get_thread_num();
      ComputingDelta(tensor, core_tensor, factor_matrices, delta, curr_factor_id, rank, cuda_agents[device_id], scheduler, device_id);
      common::cuda::_CUDA_API_CALL(cudaDeviceSynchronize());
    }
    printf("\t- Elapsed time for Computing Delta: %lf\n", omp_get_wtime() - delta_time);

    double bc_time = omp_get_wtime();
    ComputingBC(tensor, delta, B, C, curr_factor_id, rank);

    printf("\t- Elapsed time for Computing B and C: %lf\n", omp_get_wtime() - bc_time);

    double update_time = omp_get_wtime();
    index_t row_count = block_dims[curr_factor_id];
    index_t col_count = rank;

    for (uint64_t part_id = 0; part_id < part_dims[curr_factor_id]; ++part_id) {
#pragma omp parallel for schedule(static)
      for (index_t row = 0; row < row_count; ++row) {
        // Getting the inverse matrix of [B + lambda * I]
        uint64_t pos_B = row * col_count * col_count;
        uint64_t pos_C = row * col_count;
        matrix_t BB(col_count, col_count);
        for (index_t k = 0; k < col_count; ++k) {
          for (index_t l = 0; l < col_count; ++l) {
            BB(k, l) = B[part_id][pos_B + k * col_count + l];
          }
        }

        matrix_t B_inv = BB.inverse();

        index_t offset = row * col_count;
        for (index_t k = 0; k < col_count; ++k) {
          value_t res = 0;
          for (index_t l = 0; l < col_count; ++l) {
            res += C[part_id][pos_C + l] * B_inv(l, k);
          }
          factor_matrices[curr_factor_id][part_id][offset + k] = res;
        }
        BB.resize(0, 0);
        B_inv.resize(0, 0);

      } // row size
    } // part_dims

    printf("\t- row-wise update TIME : %lf\n", omp_get_wtime() - update_time);

  } // ! curr_factor
}

} // namespace gputucker
} // namespace supertensor

#endif /* UPDATE_CUH_ */