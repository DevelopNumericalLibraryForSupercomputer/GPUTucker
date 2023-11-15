#ifndef BLOCK_HPP_
#define BLOCK_HPP_

#include "gputucker/constants.hpp"
namespace supertensor {

namespace gputucker {
#define BLOCK_TEMPLATE template <typename IndexType, typename ValueType>

#define BLOCK_TEMPLATE_ARGS IndexType, ValueType

BLOCK_TEMPLATE
class Block {
 public:
  using this_t = Block<IndexType, ValueType>;
  using index_t = IndexType;
  using value_t = ValueType;

  unsigned short order;
  index_t *dims;
  uint64_t nnz_count;

  value_t *values;

  index_t *indices[constants::kMaxOrder];
  index_t *where_nnz[constants::kMaxOrder];
  uint64_t *count_nnz[constants::kMaxOrder];

  index_t *mean;
  index_t *median;
  index_t *mode;

 public:
  Block();
  Block(uint64_t new_block_id, unsigned short new_order);
  Block(uint64_t new_block_id, index_t *new_block_coord,
        unsigned short new_order, index_t *new_dims);
  ~Block();

  bool is_empty();
  void setup_data(uint64_t new_nnz_count);
  void insert_nonzero(uint64_t pos, index_t *new_coord, value_t new_value);
  void set_dims(index_t *new_dims);
  void set_nnz_count(uint64_t new_nnz_count);
  index_t *get_block_coord();

  void assign_indices();
  void to_string();

 protected:
  void _initialize();
  void _allocate_data();

 private:
  index_t *_base_dims;
  uint64_t _block_id;
  index_t *_block_coord;
};
}  // namespace gputucker
}  // namespace supertensor

#include "gputucker/block.tpp"
#endif