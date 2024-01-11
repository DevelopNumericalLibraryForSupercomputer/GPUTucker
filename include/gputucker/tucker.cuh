#ifndef TUCKER_CUH_
#define TUCKER_CUH_

#include <omp.h>

#include "gputucker/constants.hpp"
#include "common/cuda_helper.hpp"

namespace supertensor {
namespace gputucker {
template <typename OptimizerType, typename TensorType>
void TuckerDecomposition(OptimizerType* optimizer, 
                          TensorType* tensor, 
                          int tucker_rank) {

  using tensor_t = TensorType;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

  MYPRINT("... Ready to fill in the factor matrices and the core tensor\n");
  value_t **factor_matrices[gputucker::constants::kMaxOrder];

  printf("\t... Make the factor matrices\n");
  // Allocate sub_factor matrices
  for (int axis = 0; axis < order; ++axis) {
    factor_matrices[axis] = static_cast<value_t **>(common::cuda::pinned_malloc(sizeof(value_t *) * partition_dims[axis]));
    index_t sub_factor_row = block_dims[axis];
    for (index_t part = 0; part < partition_dims[axis]; ++part) {
      factor_matrices[axis][part] = static_cast<value_t *>(common::cuda::pinned_malloc(sizeof(value_t) * sub_factor_row * rank));
    }
  }

  // Initialize sub_factor matrices
  for (int axis = 0; axis < order; ++axis) {
    printf("\t\t... Fill the factor matrix [%d]\n", axis);
    index_t sub_factor_row = block_dims[axis];
    for (index_t part = 0; part < partition_dims[axis]; ++part) {
      if (part + 1 == partition_dims[axis]) {
        sub_factor_row = dims[axis] - part * block_dims[axis];
      }
      for (index_t row = 0; row < sub_factor_row; ++row) {
        for (int col = 0; col < rank; ++col) {
          factor_matrices[axis][part][row * rank + col] = gputucker::frand<double>(0, 1);
        }
      }
    }
  }


   // Core tensor
  printf("\t... Make the core tensor\n");
  tensor_t *core_tensor = new tensor_t(order);
  index_t *core_dims = gputucker::allocate<index_t>(order);
  index_t *core_part_dims = gputucker::allocate<index_t>(order);
  uint64_t core_nnz_count = 1;
  for (int axis = 0; axis < order; ++axis) {
    core_dims[axis] = rank;
    core_part_dims[axis] = 1;
    core_nnz_count *= rank;
  }

  core_tensor->set_dims(core_dims);
  core_tensor->set_nnz_count(core_nnz_count);
  core_tensor->make_blocks(core_part_dims, &core_nnz_count);

  block_t *curr_block = core_tensor->blocks[0];

  // #pragma omp parallel for
  for (uint64_t i = 0; i < core_nnz_count; ++i) {
    curr_block->values[i] = gputucker::frand<double>(0, 1);
    index_t mult = 1;
    for (short axis = order; --axis >= 0;) {
      index_t idx = 0;
      if (axis == order - 1) {
        idx = i % core_dims[axis];
      } else if (axis == 0) {
        idx = i / mult;
      } else {
        idx = (i / mult) % core_dims[axis];
      }
      curr_block->indices[axis][i] = idx;
      mult *= core_dims[axis];
    }
    assert(mult == core_nnz_count);
  }
  printf("\t... Initialize the intermediate data (delta, B and C, errorT)\n");
  const uint64_t block_count      = tensor->block_count;
  const index_t max_block_dim     = tensor->get_max_block_dim();
  const index_t max_partition_dim = tensor->get_max_partition_dim();

  using matrix_t = double;
  value_t **delta   = gputucker::allocate<value_t *>(block_count);
  matrix_t **B      = gputucker::allocate<matrix_t *>(max_partition_dim);
  matrix_t **C      = gputucker::allocate<matrix_t *>(max_partition_dim);
  value_t **error_T = gputucker::allocate<value_t *>(block_count);

  for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
    block_t *curr_block = tensor->blocks[block_id];
    delta[block_id]     = gputucker::allocate<value_t>(curr_block->nnz_count * rank);  
    error_T[block_id]   = gputucker::allocate<value_t>(curr_block->nnz_count);
  }
  for (index_t part = 0; part < max_partition_dim; ++part) {
    B[part] = gputucker::allocate<matrix_t>(max_block_dim * rank * rank);
    C[part] = gputucker::allocate<matrix_t>(max_block_dim * rank);
  }

  int iter = 0;
  double p_fit = -1;
  double fit = -1;

  double avg_time = omp_get_wtime();

  while (1) {
    // 1) update factor matrices
    // 2) reconstruction
  }

  MYPRINT("DONE\n");



}

}  // namespace gputucker
}  // namespace supertensor

#endif /* TUCKER_CUH_ */