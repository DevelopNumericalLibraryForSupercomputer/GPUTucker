#include <cassert>

#include "gputucker/constants.hpp"
#include "gputucker/helper.hpp"
#include "gputucker/optimizer.hpp"

namespace supertensor {
namespace gputucker {

OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::Initialize(unsigned short new_gpu_count, 
                                                    unsigned int new_rank,
                                                    uint64_t new_gpu_mem_size,
                                                    tensor_t* new_data) {
  cuda_stream_count = 1;
  rank = new_rank;
  gpu_count = new_gpu_count;
  gpu_mem_size = new_gpu_mem_size;
  
  // component_cost = gputucker::allocate<CostMetric>(static_cast<int>(Component::ComponentCount));
  partition_type = gputucker::enums::PartitionTypes::kDimensionPartition;
  this->_data = new_data;
}
// OPTIMIZER_TEMPLATE
// size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::GetDataSize(Component comp) {
//   switch (comp) {
//     case SubTensor:
//       return this->_get_data_size_sub_tensor();
//     case CoreTensor:
//       return this->_get_data_size_core_tensor();
//     case SubFactors:
//       return this->_get_data_size_each_sub_factor();
//     case Delta:
//       return this->_get_data_size_sub_delta();
//     default:
//       std::cout << "Invalid Component" << std::endl;
//       return 0;
//   }
// }

// OPTIMIZER_TEMPLATE
// size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::GetTransferSize(Component comp) {
//   switch (comp) {
//     case SubTensor:
//       return this->_get_transfer_size_sub_tensor();
//     case CoreTensor:
//       return this->_get_transfer_size_core_tensor();
//     case SubFactors:
//       return this->_get_transfer_size_sub_factors();
//     case Delta:
//       return this->_get_transfer_size_delta();
//     default:
//       std::cout << "Invalid Component" << std::endl;
//       return 0;
//   }
// }

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::GetAllDataSize() {
  size_t ret_size = 0;

  ret_size += this->_get_data_size_sub_tensor();
  ret_size += this->_get_data_size_core_tensor();
  ret_size += this->_get_data_size_sub_factors();
  ret_size += this->_get_data_size_sub_delta();

  // for (int i = 0; i < static_cast<int>(Component::ComponentCount); ++i) {
  //   Component c = static_cast<Component>(i);
  //   ret_size += this->component_cost[i].data_size;
  // }
  return ret_size;
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::GetAllTransferSize() {
  size_t ret_size = 0;

  ret_size += this->_get_transfer_size_sub_tensor();
  ret_size += this->_get_transfer_size_core_tensor();
  ret_size += this->_get_transfer_size_sub_factors();
  ret_size += this->_get_transfer_size_delta();
  // for (int i = 0; i < static_cast<int>(Component::ComponentCount); ++i) {
  //   Component c = static_cast<Component>(i);
  //   ret_size += this->component_cost[i].transfer_size;
  // }
  return ret_size;
}

// OPTIMIZER_TEMPLATE
// void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::estimate_component_costs() {
//   for (int i = 0; i < static_cast<int>(Component::ComponentCount); ++i) {
//     Component c = static_cast<Component>(i);
//     this->component_cost[i].data_size = this->get_data_size(c);
//     this->component_cost[i].transfer_size = this->get_transfer_size(c);
//   }
// }

OPTIMIZER_TEMPLATE
Optimizer<OPTIMIZER_TEMPLATE_ARGS>::index_t* Optimizer<OPTIMIZER_TEMPLATE_ARGS>::FindPartitionParms() {
  MYPRINT("Find partition find_partition_parms\n");

  unsigned short order = this->_data->order;
  index_t* dims = this->_data->dims;

  block_dims = gputucker::allocate<index_t>(order);
  partition_dims = gputucker::allocate<index_t>(order);

  for (unsigned short axis = 0; axis < order; ++axis) {
    block_dims[axis] = dims[axis];
    partition_dims[axis] = 1;
  }
  this->_RefreshBlockDims();
  // Determine partition type (Nonzero-based Partitioning OR Dimension-based Partitioning)
  this->_DeterminePartitionType();

  return partition_dims;
}

OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_DeterminePartitionType() {
  MYPRINT("Determine partition type (Nonzero-based Partitioning OR Dimension-based Partitioning)\n");

  size_t gpu_total_mem_size = gpu_count * gpu_mem_size;

  size_t required_size = 0;
  required_size += this->_get_data_size_input_tensor();
  required_size += gpu_count * this->_get_data_size_core_tensor();
  required_size += gpu_count * this->_get_data_size_sub_factors();
  required_size += this->_get_data_size_delta();

  printf("Required Size / Total %d GPUs Mem size\t: ", gpu_count);
  std::cout << common::HumanReadable{(std::uintmax_t)required_size} << " / "
            << common::HumanReadable{(std::uintmax_t)gpu_total_mem_size}
            << std::endl;

  if (required_size <= gpu_total_mem_size) {
    partition_type = gputucker::enums::PartitionTypes::kNonzeroPartition;
    std::cout << "\t- Partitioning Type: Nonzero-based (Small-scale)\n" << std::endl;
    this->_NonzeroBasedPartitioning();
  } else {
    partition_type = gputucker::enums::PartitionTypes::kDimensionPartition;
    std::cout << "\t- Partitioning Type: Dimension-based (Large-scale)\n" << std::endl;
    this->_DimensionBasedPartitioning();
  }
  this->_AvailableNonzeroCountPerTask();
}

OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_NonzeroBasedPartitioning(){

  cuda_stream_count = 1;
  // size_t avail_buffer_size = this->_gpu_mem_size - (this->_get_data_size_core_tensor() + this->_get_data_size_factors());
  // avail_nnz_count_per_task = avail_buffer_size / (this->_data->order * sizeof(index_t) + sizeof(value_t) + rank * sizeof(value_t));
  // printf("\t- Max. available non-zero count per a device: %lu\n", avail_nnz_count_per_task);
}

OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_DimensionBasedPartitioning(){

  cuda_stream_count = 4;
  size_t gpu_stream_buffer_size = gpu_mem_size / cuda_stream_count;
  uint32_t total_cuda_stream_count = gpu_count * cuda_stream_count;

  int iter = 0;
  do {
    ++iter;

    unsigned short partition_axis = this->_NextPartitionAxis();
    partition_dims[partition_axis]++;
    this->_RefreshBlockDims();
    // this->estimate_component_costs();
  } while (!(GetAllDataSize() < gpu_stream_buffer_size && block_count >= total_cuda_stream_count));

  // assert(avail_memory > 0);
  // std::cout << "core tensor size\t"
  //           << common::HumanReadable{(std::uintmax_t)this->_get_data_size_core_tensor()}
  //           << std::endl;
  // std::cout << "each sub factors size\t"
  //           << common::HumanReadable{(std::uintmax_t)this->_get_data_size_sub_factors()}
  //           << std::endl;
  // std::cout << "avail_memory\t"
  //           << common::HumanReadable{(std::uintmax_t)avail_memory}
  //           << std::endl;
  // avail_nnz_count_per_task = avail_memory / (order * sizeof(index_t) + (1 + this->rank) * sizeof(value_t));
}

OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_AvailableNonzeroCountPerTask() {
  size_t gpu_stream_buffer_size = gpu_mem_size / cuda_stream_count;
  size_t avail_buffer_size = gpu_stream_buffer_size - (this->_get_data_size_core_tensor() + this->_get_data_size_sub_factors());
  avail_nnz_count_per_task = avail_buffer_size / (this->_data->order * sizeof(index_t) + sizeof(value_t) + rank * sizeof(value_t));
}

OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::ToString() {
  PrintLine();
  printf("< OPTIMIZER >\n");

  std::cout << "Partition Type: ";
  if(partition_type == gputucker::enums::kNonzeroPartition) {
    std::cout << "Nonzero-based Partitioning" << std::endl;
  } else {
    std::cout << "Dimension-based Partitioning" << std::endl;
  }

  unsigned short order = this->_data->order;
  for (int axis = 0; axis < order; ++axis) {
    printf("Partition dim[%d] = %lu\n", axis, partition_dims[axis]);
  }

  std::cout << "\t@ All data size for a CUDA execution seq.\t"
            << common::HumanReadable{(std::uintmax_t) GetAllDataSize()} << std::endl;

  std::cout << "\t@ All amount of transfer data size \t"
            << common::HumanReadable{(std::uintmax_t) GetAllTransferSize()} << std::endl;

  printf("Max. Available nonzeros per task: %lu\n", avail_nnz_count_per_task);
  printf("The number of CUDA Streams in a GPU: %d\n", cuda_stream_count);
  printf("The number of GPUs: %d\n", gpu_count);
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_input_tensor() {
  size_t ret_size = this->_data->nnz_count * (this->_data->order * sizeof(index_t) + sizeof(value_t));
  return ret_size;
}

// Calculates and returns the size of a sub-tensor
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_sub_tensor() {
  uint64_t block_count = 1;
  unsigned short order = this->_data->order;
  for (unsigned short axis = 0; axis < order; ++axis) {
    block_count *= this->partition_dims[axis];
  }
  size_t ret_size = avg_nnz_count_per_block * (order * sizeof(index_t) + sizeof(value_t));
  // std::cout << ">>> Sub-Tensor Size \t: " <<
  // common::HumanReadable{(std::uintmax_t)ret_size} << std::endl;

  return ret_size;
}

// Calculates and returns the size of a core tensor
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_core_tensor() {
  unsigned int order = this->_data->order;
  size_t core_nnz_count = std::pow(rank, order);
  size_t ret_size = core_nnz_count * (order * sizeof(index_t) + sizeof(value_t));
  return ret_size;
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_all_factors() {
  size_t ret_size = 0;
  for (unsigned short axis = 0; axis < this->_data->order; ++axis) {
    ret_size += this->_data->dims[axis] * rank;
  }
  return ret_size * sizeof(value_t);
}

// Calculates and returns the size of sub-factor matrices
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_sub_factors() {
  // sum of each sub-factor for the factor
  size_t ret_size = 0;
  for (unsigned short axis = 0; axis < this->_data->order; ++axis) {
    ret_size += block_dims[axis] * rank;
  }
  return ret_size * sizeof(value_t);
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_delta() {
  size_t ret_size = this->_data->nnz_count * rank * sizeof(value_t);
  return ret_size;
}
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_sub_delta() {
  uint64_t block_count = 1;
  unsigned short order = order;
  for (unsigned short axis = 0; axis < order; ++axis) {
    block_count *= this->partition_dims[axis];
  }
  size_t ret_size = this->avg_nnz_count_per_block * this->rank * sizeof(value_t);
  return ret_size;
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_transfer_size_sub_tensor() {
  size_t ret_size = this->_data->nnz_count * (this->_data->order * sizeof(index_t) + sizeof(value_t));
  return ret_size;
}
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_transfer_size_core_tensor() {
  return this->_get_data_size_core_tensor();
}
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_transfer_size_sub_factors() {
  uint64_t block_count = 1;
  unsigned short order = this->_data->order;
  size_t ret_size = 0;
  for (unsigned short axis = 0; axis < order; ++axis) {
    block_count *= this->partition_dims[axis];
  }
  for (unsigned short axis = 0; axis < order; ++axis) {
    ret_size += block_count * this->block_dims[axis] * this->rank;
  }

  ret_size *= sizeof(value_t);
  return ret_size;
}
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_transfer_size_delta() {
  size_t ret_size = this->_data->nnz_count * this->rank * sizeof(value_t);
  return ret_size;
}

/* determining the next axis or dimension along which the data will be
 * partitioned. */
OPTIMIZER_TEMPLATE
unsigned short Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_NextPartitionAxis() {
  unsigned short max_axis = 0;

  for (unsigned short axis = 1; axis < this->_data->order; ++axis) {
    if (block_dims[max_axis] < block_dims[axis]) {
      max_axis = axis;
    }
  }
  return max_axis;
}

/* Adjusting block dimensions using partition dimensions */
OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_RefreshBlockDims() {
  // Initialize block dimensions
  int order = this->_data->order;
  index_t* dims = this->_data->dims;

  block_count = 1;
  for (unsigned short axis = 0; axis < order; ++axis) {
    block_dims[axis] = (dims[axis] + partition_dims[axis] - 1) / partition_dims[axis];
    index_t check_dim = (dims[axis] + block_dims[axis] - 1) / block_dims[axis];
    if (check_dim != partition_dims[axis]) {
      throw std::runtime_error(ERROR_LOG("[ERROR] Block dimension is larger than the tensor dimension."));
    }
    // if (check_dim < partition_dims[axis]) {
    //   partition_dims[axis] = check_dim;
    // }
    block_count *= partition_dims[axis];
  }

  avg_nnz_count_per_block = (this->_data->nnz_count + block_count - 1) / block_count;
}
}  // namespace gputucker
}  // namespace supertensor
