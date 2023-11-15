// Purpose: Source file for Block class
#include <cassert>

#include "gputucker/block.hpp"
#include "gputucker/constants.hpp"
#include "gputucker/helper.hpp"

namespace supertensor {
namespace gputucker {
BLOCK_TEMPLATE
Block<BLOCK_TEMPLATE_ARGS>::Block(uint64_t new_block_id,
                                  index_t *new_block_coord,
                                  unsigned short new_order, index_t *new_dims)
    : order(new_order), _block_id(new_block_id) {
  this->_initialize();
  for (int axis = 0; axis < order; ++axis) {
    this->_block_coord[axis] = new_block_coord[axis];  // for setting base_dims
  }
  this->set_dims(new_dims);

  mean = gputucker::allocate<index_t>(new_order);
  median = gputucker::allocate<index_t>(new_order);
  mode = gputucker::allocate<index_t>(new_order);
}

BLOCK_TEMPLATE
Block<BLOCK_TEMPLATE_ARGS>::Block(uint64_t new_block_id,
                                  unsigned short new_order)
    : Block(new_block_id, NULL, new_order, NULL) {}
BLOCK_TEMPLATE
Block<BLOCK_TEMPLATE_ARGS>::Block() : Block(0, NULL, 0, NULL) {}
BLOCK_TEMPLATE
Block<BLOCK_TEMPLATE_ARGS>::~Block() {}

/**
 * Sets up the data for a Block object by allocating memory for indices and
 * values
 * @param new_nnz_count the number of non-zero elements in the Block
 */
BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::setup_data(uint64_t new_nnz_count) {
  // Set the number of non-zero elements in the Block
  this->set_nnz_count(new_nnz_count);
  this->_allocate_data();
}

BLOCK_TEMPLATE
bool Block<BLOCK_TEMPLATE_ARGS>::is_empty() { return this->nnz_count == 0; }

BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::insert_nonzero(uint64_t pos,
                                                index_t *new_coord,
                                                value_t new_value) {
  for (unsigned short axis = 0; axis < order; ++axis) {
    this->indices[axis][nnz_count - pos] =
        new_coord[axis] - this->_base_dims[axis];
  }
  // printf("\t %1.1f\n", new_value);
  this->values[this->nnz_count - pos] = new_value;
}

BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::set_dims(index_t *new_dims) {
  for (int axis = 0; axis < this->order; ++axis) {
    this->dims[axis] = new_dims[axis];
    this->_base_dims[axis] = this->dims[axis] * this->_block_coord[axis];
  }
}

BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::set_nnz_count(uint64_t new_nnz_count) {
  this->nnz_count = new_nnz_count;
}

BLOCK_TEMPLATE
Block<BLOCK_TEMPLATE_ARGS>::index_t *
Block<BLOCK_TEMPLATE_ARGS>::get_block_coord() {
  return this->_block_coord;
}

BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::assign_indices() {
  uint64_t *temp_nnz[gputucker::constants::kMaxOrder];

  // Loop through each axis and allocate memory for temporary variables to store
  // the number of non-zero elements, as well as where they are located
  for (unsigned short axis = 0; axis < order; ++axis) {
    temp_nnz[axis] = gputucker::allocate<uint64_t>((this->dims[axis] + 1));
    this->count_nnz[axis] =
        gputucker::allocate<uint64_t>((this->dims[axis] + 1));
    this->where_nnz[axis] = gputucker::allocate<index_t>(this->nnz_count);
  }

  // Loop through each axis and initialize temporary arrays to zero
  for (unsigned short axis = 0; axis < order; ++axis) {
    for (index_t k = 0; k < this->dims[axis]; ++k) {
      this->count_nnz[axis][k] = 0;
      temp_nnz[axis][k] = 0;
    }
  }

  // Loop through each axis and non-zero element,
  // and count the number of non-zero elements for each axis
  for (unsigned short axis = 0; axis < order; ++axis) {
    for (uint64_t nnz = 0; nnz < this->nnz_count; ++nnz) {
      index_t k = this->indices[axis][nnz];
      assert(k < dims[axis]);
      this->count_nnz[axis][k]++;
      temp_nnz[axis][k]++;
    }
  }

  index_t now = 0;
  index_t k;
  index_t j = 0;

  // Loop through each axis and calculate the starting index of each block
  for (unsigned short axis = 0; axis < order; ++axis) {
    now = 0;
    uint64_t frequent_idx = 0;
    uint64_t max_count = 0;

    for (j = 0; j < dims[axis]; ++j) {
      k = this->count_nnz[axis][j];
      if (max_count < k) {
        frequent_idx = j;
        max_count = k;
      }
      this->count_nnz[axis][j] = now;
      temp_nnz[axis][j] = now;
      now += k;
    }

    this->mode[axis] = frequent_idx;
    this->count_nnz[axis][j] = now;
    temp_nnz[axis][j] = now;
  }

  // Loop through each axis and non-zero element,
  // and store where each non-zero element is located
  for (unsigned short axis = 0; axis < order; ++axis) {
    uint64_t sum_idx = 0;
    for (uint64_t nnz = 0; nnz < this->nnz_count; ++nnz) {
      k = this->indices[axis][nnz];
      now = temp_nnz[axis][k];
      this->where_nnz[axis][now] = nnz;
      temp_nnz[axis][k]++;
      sum_idx += k;
    }
    this->mean[axis] = sum_idx / this->nnz_count;
  }
  // Median, mean, mode
  for (unsigned short axis = 0; axis < order; ++axis) {
    uint64_t offset = nnz_count % 2 == 0 ? nnz_count / 2 : (nnz_count + 1) / 2;
    uint64_t nnz = this->where_nnz[axis][offset];
    this->median[axis] = this->indices[axis][nnz];
  }

  // Deallocates
  for (unsigned short axis = 0; axis < order; ++axis) {
    gputucker::deallocate<uint64_t>(temp_nnz[axis]);
  }
}
BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::_initialize() {
  if (this->order < 1) {
    throw std::runtime_error(
        ERROR_LOG("[ERROR] Block order should be larger than 1."));
  }
  this->dims = gputucker::allocate<index_t>(this->order);
  this->nnz_count = 0;

  this->_base_dims = gputucker::allocate<index_t>(this->order);
  this->_block_coord = gputucker::allocate<index_t>(this->order);
}

BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::_allocate_data() {
  assert(this->nnz_count != 0);
  // Allocate memory for indices in each axis
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    this->indices[axis] = gputucker::allocate<index_t>(this->nnz_count);
  }
  // Allocate memory for values
  this->values = gputucker::allocate<value_t>(this->nnz_count);
}

BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::to_string() {
  printf("********** BLOCK[%lu] Information **********\n", this->_block_id);

  int axis;

  printf("Block coord: ");
  for (axis = 0; axis < this->order; ++axis) {
    printf("[%lu]", this->_block_coord[axis]);
  }
  printf("\n");

  printf("Block order: %d\n", this->order);

  printf("Block dims: ");
  for (axis = 0; axis < this->order; ++axis) {
    printf("%lu", this->dims[axis]);
    if (axis < this->order - 1) {
      printf(" X ");
    } else {
      printf("\n");
    }
  }

  printf("# nnzs: %lu\n", this->nnz_count);

  printf("MEDIAN: \t(");
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    printf("%u, ", this->median[axis]);
  }
  printf(")\n");

  printf("Mode: \t(");
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    printf("%u, ", this->mode[axis]);
  }
  printf(")\n");

  printf("Mean: \t(");
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    printf("%u, ", this->mean[axis]);
  }
  printf(")\n");

  printf("****************************************\n");
}

}  // namespace gputucker
}  // namespace supertensor