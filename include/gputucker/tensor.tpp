
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
void Tensor<TENSOR_TEMPLATE_ARGS>::reset(this_t *other) {
  // this->set_dims(other->dims);
  // this->set_nnz_count(other->nnz_count);
  // this->set_partition_dims(other->partition_dims);
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::make_blocks(
    const index_t *new_partition_dims, uint64_t *histogram) {}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::assign_indices() {}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::block_id_to_block_coord(uint64_t block_id,
                                                           index_t *coord) {}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::set_dims(index_t *new_dims) {
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    this->dims[axis] = new_dims[axis];
  }
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::set_partition_dims(
    const index_t *new_partition_dims) {}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::set_nnz_count(uint64_t new_nnz_count) {
  this->nnz_count = new_nnz_count;
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::set_data(uint64_t block_id,
                                            index_t *new_indices[],
                                            value_t *new_values) {
  assert(this->order > 1);

  std::cout << "tensor->set_data()" << std::endl;
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
    throw std::runtime_error(
        ERROR_LOG("[ERROR] Tensor order should be larger than 1."));
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
  }
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::to_string() {
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    printf("Max. dim[%d] = %lu\n", axis, this->dims[axis]);
  }
  printf("# nnzs: %lu\n", this->nnz_count);
  printf("-------------\n");
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    printf("Partition dim[%d] = %lu\n", axis, this->partition_dims[axis]);
  }
  printf("# blocks: %lu\n", this->block_count);
  printf("-------------\n");
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    printf("Block dim[%d] = %lu\n", axis, this->block_dims[axis]);
  }
  printf("# empty blocks: %lu / %lu\n", this->empty_block_count,
         this->block_count);

  // for (int nnz = 0; nnz < 10; ++nnz)
  // {
  // 	printf("\t (");
  // 	for (unsigned short axis = 0; axis < this->order - 1; ++axis)
  // 	{
  // 		printf("%lu, ", this->indices[axis][nnz]);
  // 	}
  // 	printf("%lu) %1.3f\n", this->indices[this->order - 1][nnz],
  // this->values[nnz]);
  // }
}

}  // namespace gputucker
}  // namespace supertensor