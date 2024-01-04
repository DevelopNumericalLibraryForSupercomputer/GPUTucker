
#include <omp.h>

#include "gputucker/tensor.hpp"
namespace supertensor {
namespace gputucker {
TENSOR_TEMPLATE
Tensor<TENSOR_TEMPLATE_ARGS>::Tensor(unsigned short new_order)
    : order(new_order) {
  this->_initialize();
}
TENSOR_TEMPLATE
Tensor<TENSOR_TEMPLATE_ARGS>::Tensor(this_t *other) {
  this->order = other->order;
  this->_initialize();
  this->set_dims(other->dims);
  this->set_nnz_count(other->nnz_count);
}
TENSOR_TEMPLATE
Tensor<TENSOR_TEMPLATE_ARGS>::Tensor() : Tensor(0) {}

TENSOR_TEMPLATE
Tensor<TENSOR_TEMPLATE_ARGS>::~Tensor() {}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::make_blocks(const index_t *new_partition_dims, 
                                              uint64_t *histogram) {
  gputucker::deallocate(this->blocks);
  this->set_partition_dims(new_partition_dims);

  this->blocks = gputucker::allocate<block_t *>(this->block_count);
  index_t *block_coord = gputucker::allocate<index_t>(this->order);

  this->max_nnz_count_in_block = 0;

  for (uint64_t block_id = 0; block_id < this->block_count; ++block_id) {
    this->block_id_to_block_coord(block_id, block_coord);
    this->blocks[block_id] = new block_t(block_id, block_coord, this->order, this->block_dims);
    this->blocks[block_id]->set_nnz_count(histogram[block_id]);
    this->blocks[block_id]->AllocateData();

    // this->blocks[block_id]->SetupData(histogram[block_id]);
    if (histogram[block_id] == 0) {
      this->empty_block_count++;
    }

    this->max_nnz_count_in_block = std::max<uint64_t>(histogram[block_id], this->max_nnz_count_in_block);
  }
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::assign_indices() {
  printf("Assign indices in blocks\n");
  assert(this->block_count != 0);
    
  #pragma omp parallel for
  for (uint64_t block_id = 0; block_id < this->block_count; ++block_id) {
    this->blocks[block_id]->assign_indices();
  }
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::block_id_to_block_coord(uint64_t block_id,
                                                           index_t *coord) {
  index_t mult = 1;

  for (unsigned short iter = 0; iter < order; ++iter) {
    unsigned short axis = order - iter - 1;
    index_t idx = 0;
    if (axis == this->order - 1) {
      idx = block_id % this->partition_dims[axis];
    } else if (axis == 0) {
      idx = block_id / mult;
    } else {
      idx = (block_id / mult) % this->partition_dims[axis];
    }
    coord[axis] = idx;
    mult *= this->partition_dims[axis];
  }
}


TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::set_dims(index_t *new_dims) {
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    this->dims[axis] = new_dims[axis];
  }
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::set_partition_dims(const index_t *new_partition_dims) {
  // Initialize block dimensions
  this->block_count = 1;
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    this->partition_dims[axis] = new_partition_dims[axis];
    this->block_dims[axis] = (this->dims[axis] + this->partition_dims[axis] - 1) / this->partition_dims[axis];
    index_t check_dim = (this->dims[axis] + this->block_dims[axis] - 1) / this->block_dims[axis];
    if (check_dim < this->partition_dims[axis]) {
      this->partition_dims[axis] = check_dim;
    }
    this->block_count *= this->partition_dims[axis];
  }
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::set_nnz_count(uint64_t new_nnz_count) {
  this->nnz_count = new_nnz_count;
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::set_data(uint64_t block_id,
                                            index_t *new_indices[],
                                            value_t *new_values) {
  assert(this->order > 1);

  for (unsigned short axis = 0; axis < this->order; ++axis) {
    this->blocks[block_id]->indices[axis] = (index_t *)new_indices[axis];
  }
  this->blocks[block_id]->values = (value_t *)new_values;
}

TENSOR_TEMPLATE
Tensor<TENSOR_TEMPLATE_ARGS>::index_t
Tensor<TENSOR_TEMPLATE_ARGS>::get_max_partition_dim() {
  index_t dim = this->partition_dims[0];

  for (unsigned short axis = 1; axis < order; ++axis) {
    dim = std::max<index_t>(dim, this->partition_dims[axis]);
  }
  return dim;
}
TENSOR_TEMPLATE
Tensor<TENSOR_TEMPLATE_ARGS>::index_t
Tensor<TENSOR_TEMPLATE_ARGS>::get_max_block_dim() {
  index_t dim = this->block_dims[0];

  for (unsigned short axis = 1; axis < order; ++axis) {
    dim = std::max<index_t>(dim, this->block_dims[axis]);
  }
  return dim;
}
TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::_initialize() {
  if (this->order < 1) {
    throw std::runtime_error(ERROR_LOG("[ERROR] Tensor order should be larger than 1."));
  }

  this->dims = gputucker::allocate<index_t>(this->order);
  this->nnz_count = 0;
  this->norm = 0;

  this->partition_dims = gputucker::allocate<index_t>(this->order);
  this->block_dims = gputucker::allocate<index_t>(this->order);
  this->max_block_dim = 0;
  this->max_nnz_count_in_block = 0;
  this->empty_block_count = 0;
  this->blocks = NULL;

  for (unsigned short axis = 0; axis < this->order; ++axis) {
    this->dims[axis] = std::numeric_limits<index_t>::min();
    this->block_dims[axis] = std::numeric_limits<index_t>::min();
    this->partition_dims[axis] = 1;
  }
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::to_string() {
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    printf("Max. dim[%d] = %lu\n", axis, this->dims[axis]);
  }
  printf("# nnzs: %lu\n", this->nnz_count);
  PrintLine();
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    printf("Partition dim[%d] = %lu\n", axis, this->partition_dims[axis]);
  }
  printf("# blocks: %lu\n", this->block_count);
  PrintLine();
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    printf("Block dim[%d] = %lu\n", axis, this->block_dims[axis]);
  }
  printf("# empty blocks: %lu / %lu\n", this->empty_block_count, this->block_count);
}

}  // namespace gputucker
}  // namespace supertensor