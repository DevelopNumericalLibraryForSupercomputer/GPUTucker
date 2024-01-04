#include <cassert>

#include "gputucker/constants.hpp"
#include "gputucker/helper.hpp"
#include "gputucker/optimizer.hpp"

namespace supertensor {
namespace gputucker {
OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::Initialize(tensor_t *tensor, 
                                                    unsigned short new_gpu_count, 
                                                    unsigned int new_rank,
                                                    uint64_t new_gpu_mem_size) {
  this->Initialize(tensor->order, 
                  tensor->dims, 
                  tensor->nnz_count,
                  new_gpu_count, 
                  new_rank, 
                  new_gpu_mem_size);
}
OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::Initialize(unsigned short new_order, 
                                                    index_t *new_dims, 
                                                    uint64_t new_nnz_count,
                                                    unsigned short new_gpu_count, 
                                                    unsigned int new_rank,
                                                    uint64_t new_gpu_mem_size) {
  this->order = new_order;
  this->nnz_count = new_nnz_count;
  this->cuda_stream_count = 1;
  this->rank = new_rank;
  this->gpu_count = new_gpu_count;
  this->_gpu_mem_size = new_gpu_mem_size;

  this->component_cost = gputucker::allocate<CostMetric>(static_cast<int>(Component::ComponentCount));

  this->dims = gputucker::allocate<index_t>(this->order);
  this->block_dims = gputucker::allocate<index_t>(this->order);
  this->partition_dims = gputucker::allocate<index_t>(this->order);

  for (unsigned short axis = 0; axis < this->order; ++axis) {
    this->dims[axis] = new_dims[axis];
    this->block_dims[axis] = new_dims[axis];
    this->partition_dims[axis] = 1;
  }

  this->_update_block_dims();
  this->estimate_component_costs();

  this->partition_type = gputucker::enums::PartitionTypes::kDimensionPartition;
}
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::get_data_size(Component comp) {
  switch (comp) {
    case SubTensor:
      return this->_get_data_size_sub_tensor();
    case CoreTensor:
      return this->_get_data_size_core_tensor();
    case SubFactors:
      return this->_get_data_size_each_sub_factor();
    case Delta:
      return this->_get_data_size_sub_delta();
    default:
      std::cout << "Invalid Component" << std::endl;
      return 0;
  }
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::get_transfer_size(Component comp) {
  switch (comp) {
    case SubTensor:
      return this->_get_transfer_size_sub_tensor();
    case CoreTensor:
      return this->_get_transfer_size_core_tensor();
    case SubFactors:
      return this->_get_transfer_size_sub_factors();
    case Delta:
      return this->_get_transfer_size_delta();
    default:
      std::cout << "Invalid Component" << std::endl;
      return 0;
  }
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::get_all_data_size() {
  size_t ret_size = 0;

  for (int i = 0; i < static_cast<int>(Component::ComponentCount); ++i) {
    Component c = static_cast<Component>(i);
    ret_size += this->component_cost[i].data_size;
  }

  return ret_size;
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::get_all_transfer_size() {
  size_t ret_size = 0;

  for (int i = 0; i < static_cast<int>(Component::ComponentCount); ++i) {
    Component c = static_cast<Component>(i);
    ret_size += this->component_cost[i].transfer_size;
  }
  return ret_size;
}

OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::estimate_component_costs() {
  for (int i = 0; i < static_cast<int>(Component::ComponentCount); ++i) {
    Component c = static_cast<Component>(i);
    this->component_cost[i].data_size = this->get_data_size(c);
    this->component_cost[i].transfer_size = this->get_transfer_size(c);
  }
}

OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::find_partition_parms() {
  MYPRINT("Find partition find_partition_parms\n");

  this->determine_partition_type();

  if (this->partition_type == gputucker::enums::kNonzeroPartition) {
    this->cuda_stream_count = 1;
    size_t avail_memory =
        this->_gpu_mem_size -
        (this->_get_data_size_core_tensor() + this->_get_data_size_factors());
    printf("\t- core tensor size: %lu\n", _get_data_size_core_tensor());
    this->avail_nnz_count_per_task =
        avail_memory /
        (this->order * sizeof(index_t) + (1 + this->rank) * sizeof(value_t));

    printf("\t- Max. available non-zero count per a device: %lu\n",
           this->avail_nnz_count_per_task);
  } else {
    this->cuda_stream_count = 4;
    size_t gpu_stream_buffer_size =
        this->_gpu_mem_size / this->cuda_stream_count;
    uint32_t total_cuda_stream_count =
        this->gpu_count * this->cuda_stream_count;

    int iter = 0;

    do {
      ++iter;

      unsigned short partition_axis = this->_get_next_partition_axis();
      this->partition_dims[partition_axis]++;

      this->_update_block_dims();
      this->estimate_component_costs();
    } while (!(this->get_all_data_size() < gpu_stream_buffer_size &&
               this->block_count >= total_cuda_stream_count));
    printf("Done partitioning\n");
    size_t avail_memory =
        gpu_stream_buffer_size - (this->_get_data_size_core_tensor() +
                                  this->_get_data_size_each_sub_factor());
    assert(avail_memory > 0);
    std::cout << "core tensor size\t"
              << common::HumanReadable{(std::uintmax_t)this
                                           ->_get_data_size_core_tensor()}
              << std::endl;
    std::cout << "each sub factors size\t"
              << common::HumanReadable{(std::uintmax_t)this
                                           ->_get_data_size_each_sub_factor()}
              << std::endl;
    std::cout << "avail_memory\t"
              << common::HumanReadable{(std::uintmax_t)avail_memory}
              << std::endl;
    this->avail_nnz_count_per_task =
        avail_memory /
        (this->order * sizeof(index_t) + (1 + this->rank) * sizeof(value_t));
  }

  PrintLine();
}

OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::determine_partition_type() {
  MYPRINT("Determine partition type\n");

  size_t gpu_total_mem_size = this->gpu_count * this->_gpu_mem_size;

  size_t required_size = 0;
  required_size += this->_get_data_size_input_tensor();
  required_size += this->gpu_count * this->_get_data_size_core_tensor();
  required_size += this->gpu_count * this->_get_data_size_factors();
  required_size += this->_get_data_size_delta();

  printf("Required Size / Total %d GPUs Mem size\t: ", this->gpu_count);
  std::cout << common::HumanReadable{(std::uintmax_t)required_size} << " / "
            << common::HumanReadable{(std::uintmax_t)gpu_total_mem_size}
            << std::endl;

  if (required_size <= gpu_total_mem_size) {
    this->partition_type = gputucker::enums::PartitionTypes::kNonzeroPartition;
    printf("\t- Partitioning Type: Nonzeros (Small-scale)\n");
  } else {
    this->partition_type =
        gputucker::enums::PartitionTypes::kDimensionPartition;
    printf("\t- Partitioning Type: Dimensions (Large-scale)\n");
  }
}

OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::to_string() {
  printf("< OPTIMIZER >\n");

  for (int axis = 0; axis < this->order; ++axis) {
    printf("Tensor dim[%d] = %u\n", axis, this->dims[axis]);
  }
  PrintLine();
  for (int axis = 0; axis < this->order; ++axis) {
    printf("Partition dim[%d] = %u\n", axis, this->partition_dims[axis]);
  }
  PrintLine();
  printf("# blocks: %lu\n", this->block_count);
  PrintLine();
  for (int axis = 0; axis < this->order; ++axis) {
    printf("Block dim[%d] = %u\n", axis, this->block_dims[axis]);
  }

  std::cout << "@@@ All data size for a CUDA execution seq.\t"
            << common::HumanReadable{(std::uintmax_t)this->get_all_data_size()}
            << std::endl;

  std::cout << "@@@ All amount of transfer data size \t"
            << common::HumanReadable{(std::uintmax_t)this->get_all_transfer_size()}
            << std::endl;

  PrintLine();
  printf("Max. Available nonzeros per task: %lu\n",
         this->avail_nnz_count_per_task);
  printf("The number of CUDA Streams in a GPU: %d\n", this->cuda_stream_count);
  printf("The number of GPUs: %d\n", this->gpu_count);
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_input_tensor() {
  size_t ret_size =
      this->nnz_count * (this->order * sizeof(index_t) + sizeof(value_t));
  return ret_size;
}

// Calculates and returns the size of a sub-tensor
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_sub_tensor() {
  uint64_t block_count = 1;
  unsigned short order = this->order;
  for (unsigned short axis = 0; axis < order; ++axis) {
    block_count *= this->partition_dims[axis];
  }
  size_t ret_size = this->predicted_nnz_count *
                    (this->order * sizeof(index_t) + sizeof(value_t));
  // std::cout << ">>> Sub-Tensor Size \t: " <<
  // common::HumanReadable{(std::uintmax_t)ret_size} << std::endl;

  return ret_size;
}

// Calculates and returns the size of a core tensor
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_core_tensor() {
  size_t core_nnz_count = std::pow(this->rank, this->order);
  size_t ret_size =
      core_nnz_count * (this->order * sizeof(index_t) + sizeof(value_t));
  // std::cout << ">>> Core Tensor Size \t: " <<
  // common::HumanReadable{(std::uintmax_t)ret_size} << std::endl;
  return ret_size;
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_factors() {
  size_t ret_size = 1;
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    ret_size += this->dims[axis] * this->rank;
  }
  return ret_size * sizeof(value_t);
}

// Calculates and returns the size of sub-factor matrices
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_each_sub_factor() {
  // sum of each sub-factor for the factor
  size_t element_count = 0;
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    element_count += this->block_dims[axis] * this->rank;
  }
  size_t ret_size = element_count * sizeof(value_t);

  // std::cout << ">>> Sub-Factors Size \t: " <<
  // common::HumanReadable{(std::uintmax_t)ret_size} << std::endl;
  return ret_size;
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_delta() {
  size_t ret_size = this->nnz_count * this->rank * sizeof(value_t);
  return ret_size;
}
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_sub_delta() {
  uint64_t block_count = 1;
  unsigned short order = this->order;
  for (unsigned short axis = 0; axis < order; ++axis) {
    block_count *= this->partition_dims[axis];
  }
  size_t ret_size = this->predicted_nnz_count * this->rank * sizeof(value_t);
  // std::cout << ">>> Delta Size \t: " <<
  // common::HumanReadable{(std::uintmax_t)ret_size} << std::endl;
  return ret_size;
}

OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_transfer_size_sub_tensor() {
  size_t ret_size =
      this->nnz_count * (this->order * sizeof(index_t) + sizeof(value_t));
  return ret_size;
}
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_transfer_size_core_tensor() {
  return this->_get_data_size_core_tensor();
}
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_transfer_size_sub_factors() {
  uint64_t block_count = 1;
  unsigned short order = this->order;
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
  size_t ret_size = this->nnz_count * this->rank * sizeof(value_t);
  return ret_size;
}

/* determining the next axis or dimension along which the data will be
 * partitioned. */
OPTIMIZER_TEMPLATE
unsigned short Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_next_partition_axis() {
  unsigned short max_axis = 0;

  for (unsigned short axis = 1; axis < this->order; ++axis) {
    if (this->block_dims[max_axis] < this->block_dims[axis]) {
      max_axis = axis;
    }
  }
  return max_axis;
}

/* Adjusting block dimensions using partition dimensions */
OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_update_block_dims() {
  // Initialize block dimensions
  this->block_count = 1;
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    this->block_dims[axis] =
        (this->dims[axis] + this->partition_dims[axis] - 1) /
        this->partition_dims[axis];
    index_t check_dim = (this->dims[axis] + this->block_dims[axis] - 1) /
                        this->block_dims[axis];
    if (check_dim < this->partition_dims[axis]) {
      this->partition_dims[axis] = check_dim;
    }
    this->block_count *= this->partition_dims[axis];
  }

  this->predicted_nnz_count = (this->nnz_count + block_count - 1) / block_count;
}
}  // namespace gputucker
}  // namespace supertensor
