#ifndef BLOCK_HPP_
#define BLOCK_HPP_

#include <iostream>
#include <cstdint>

#include "gputucker/constants.hpp"
namespace supertensor {
namespace gputucker {

#define BLOCK_TEMPLATE template <typename IndexType, typename ValueType>
#define BLOCK_TEMPLATE_ARGS IndexType, ValueType

BLOCK_TEMPLATE
class Block {
 public:
  using this_t = Block<BLOCK_TEMPLATE_ARGS>;
  using index_t = IndexType;
  using value_t = ValueType;

  Block();
  Block(uint64_t new_block_id, unsigned short new_order);
  Block(uint64_t new_block_id, 
        index_t *new_block_coord,
        unsigned short new_order, 
        index_t *new_dims,
        uint64_t new_nnz_count);
  ~Block();

  bool IsEmpty(){ return this->nnz_count == 0; }
  bool IsAllocated(){ return this->_is_allocated; }
  void AllocateData();
  // void SetupData(uint64_t new_nnz_count);
  void InsertNonzero(uint64_t pos, index_t *new_coord, value_t new_value);
  void AssignIndicesToEachMode();
  void ToString();
  
  /* Getter */
  index_t* get_block_coord()  { return this->_block_coord; }
  index_t get_block_id()      { return this->_block_id; }
  index_t* get_base_dims()    { return this->_base_dims; }

  /* Setter */
  void set_nnz_count(uint64_t new_nnz_count) { this->nnz_count = new_nnz_count; }
  void set_dims(index_t *new_dims);
  void set_is_allocated(bool new_is_allocated) { this->_is_allocated = new_is_allocated; }


public:
  /* Metadata */
  unsigned short order;
  index_t *dims;
  uint64_t nnz_count;

  /* Data */
  value_t *values;
  index_t *indices[constants::kMaxOrder];

  /* Metadata for each mode */
  index_t *where_nnz[constants::kMaxOrder];
  uint64_t *count_nnz[constants::kMaxOrder];

 private:
  index_t *_base_dims;
  uint64_t _block_id;
  index_t *_block_coord;
  bool _is_allocated; // whether the data is allocated

}; // class Block
  
} // namespace gputucker
} // namespace supertensor

#include "gputucker/block.tpp"
#endif