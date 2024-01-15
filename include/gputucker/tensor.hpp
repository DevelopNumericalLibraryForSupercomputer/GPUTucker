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
  Tensor(unsigned short new_order);
  Tensor(this_t* other);
  Tensor();
  ~Tensor();

  void MakeBlocks(uint64_t new_block_count, 
                  uint64_t *histogram);
  void InsertData(uint64_t block_id,
                  index_t* indices[],
                  value_t* values);
  void AssignIndicesOfEachBlock();
  void ToString();

  /* Setter */
  void set_dims(index_t *new_dims);
  void set_partition_dims(const index_t *new_partition_dims);
  void set_nnz_count(uint64_t new_nnz_count) { nnz_count = new_nnz_count; }

  /* Getter */
  index_t get_max_partition_dim() { return this->_max_partition_dim; }
  index_t get_max_block_dim() { return this->_max_block_dim;}
  index_t get_max_nnz_count_in_block() { return this->_max_nnz_count_in_block; }

  private:
    void _BlockIDtoBlockCoord(uint64_t block_id, index_t *coord);
    void _RefreshDims();


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
  block_t **blocks;

  private:
  uint64_t _max_nnz_count_in_block;
  index_t _max_block_dim;
  uint64_t _empty_block_count;
  index_t _max_partition_dim;


};  // class Tensor

}  // namespace gputucker
}  // namespace supertensor
#include "gputucker/tensor.tpp"
#endif /* TENSOR_HPP_ */