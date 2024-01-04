#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "gputucker/block.hpp"

namespace supertensor {
namespace gputucker {

#define TENSOR_TEMPLATE template <typename BlockType>
#define TENSOR_TEMPLATE_ARGS BlockType

TENSOR_TEMPLATE
class Tensor {
 public:
  using this_t = Tensor<TENSOR_TEMPLATE_ARGS>;

  using block_t = BlockType;
  using index_t = typename block_t::index_t;
  using value_t = typename block_t::value_t;

 public:
  /* Tensor Description */
  unsigned short order;  // Number of dimensions
  index_t *dims;
  uint64_t nnz_count;  // Number of nonzeros
  value_t norm;

  /* Block Description */
  index_t *partition_dims;
  index_t *block_dims;
  uint64_t block_count;
  index_t max_block_dim;
  uint64_t max_nnz_count_in_block;
  uint64_t empty_block_count;
  block_t **blocks;

 public:
  Tensor(unsigned short new_order);
  Tensor(this_t *other);
  Tensor();
  ~Tensor();

  void allocate_blocks();  // allocate memory for the blocks

  void make_blocks(const index_t *new_partition_dims, uint64_t *histogram);
  void assign_indices();
  uint64_t offset_to_block_id(uint64_t offset);  // nnz offset
  uint64_t coord_to_block_id(index_t *coord);
  void block_id_to_block_coord(uint64_t block_id, index_t *coord);
  void to_string();

  /* Setter */
  void set_dims(index_t *new_dims);
  void set_partition_dims(const index_t *new_partition_dims);
  void set_nnz_count(uint64_t new_nnz_count);
  void set_data(uint64_t block_id, index_t *new_indices[], value_t *new_values);
  index_t get_max_partition_dim();
  index_t get_max_block_dim();

 protected:
  void _initialize();
};  // class Tensor

}  // namespace gputucker
}  // namespace supertensor
#include "gputucker/tensor.tpp"
#endif /* TENSOR_HPP_ */