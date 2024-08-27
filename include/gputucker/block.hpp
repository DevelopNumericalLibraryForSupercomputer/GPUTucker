#ifndef BLOCK_HPP_
#define BLOCK_HPP_

#include <cstdint>
#include <iostream>

#include "gputucker/constants.hpp"
namespace supertensor {
namespace gputucker {

#define BLOCK_TEMPLATE template <typename IndexType, typename ValueType>
#define BLOCK_TEMPLATE_ARGS IndexType, ValueType

/**
 * @brief A class representing a block in the Tucker decomposition.
 *
 * The `Block` class is a fundamental data structure used in Tucker decomposition.
 * It stores non-zero elements of a tensor along with their coordinates and provides
 * methods for memory allocation, data insertion, and other operations necessary
 * for Tucker decomposition.
 *
 * @tparam IndexType The data type used for indexing, typically an integer type.
 * @tparam ValueType The data type used for storing values, typically a floating-point type.
 *
 * @author Jihye Lee
 * @date 2023-08-10
 * @version 1.0.0
 */
BLOCK_TEMPLATE
class Block {
public:
  using this_t = Block<BLOCK_TEMPLATE_ARGS>; ///< Type alias for the Block class.
  using index_t = IndexType;                 ///< Type alias for the index type.
  using value_t = ValueType;                 ///< Type alias for the value type.

  /**
   * @brief Default constructor.
   *
   * Initializes a Block object with default values.
   */
  Block();

  /**
   * @brief Constructor that initializes a block with a specified ID and order.
   *
   * @param new_block_id The unique identifier for the block.
   * @param new_order The order (or rank) of the tensor.
   */
  Block(uint64_t new_block_id, unsigned short new_order);

  /**
   * @brief Constructor that initializes a block with specific parameters.
   *
   * @param new_block_id The unique identifier for the block.
   * @param new_block_coord Pointer to the coordinates of the block.
   * @param new_order The order (or rank) of the tensor.
   * @param new_dims Pointer to the dimensions of the block.
   * @param new_nnz_count The number of non-zero elements in the block.
   */
  Block(uint64_t new_block_id, index_t *new_block_coord, unsigned short new_order, index_t *new_dims, uint64_t new_nnz_count);

  /**
   * @brief Destructor.
   *
   * Deallocates any allocated memory and cleans up the Block object.
   */
  ~Block();

  /**
   * @brief Checks if the block is empty.
   *
   * @return `true` if the block contains no non-zero elements, `false` otherwise.
   */
  bool IsEmpty() { return this->nnz_count == 0; }

  /**
   * @brief Checks if memory for the block's data has been allocated.
   *
   * @return `true` if the data has been allocated, `false` otherwise.
   */
  bool IsAllocated() { return this->_is_allocated; }

  /**
   * @brief Allocates memory for the block's indices and values.
   *
   * Allocates memory necessary for storing the non-zero elements and their
   * indices for each mode of the tensor.
   *
   */
  void AllocateData();

  /**
   * @brief Inserts a non-zero element into the block.
   *
   * @param pos The position in the flattened tensor where the non-zero element is located.
   * @param new_coord Pointer to the coordinates of the non-zero element.
   * @param new_value The value of the non-zero element.
   *
   */
  void InsertNonzero(uint64_t pos, index_t *new_coord, value_t new_value);

  /**
   * @brief Assigns indices to each mode of the block tensor.
   *
   * Processes the block tensor and assigns indices to each mode based on the
   * number of non-zero elements in each mode.
   *
   */
  void AssignIndicesToEachMode();

  /**
   * @brief Converts the block data to a string representation.
   *
   * This function generates a string that represents the block's contents, including
   * its indices, values, and metadata.
   *
   * @note Useful for debugging or logging the block's state.
   */
  void ToString();

  /* Getter methods */

  /**
   * @brief Retrieves the coordinates of the block.
   *
   * @return Pointer to the block's coordinates.
   */
  index_t *get_block_coord() { return this->_block_coord; }

  /**
   * @brief Retrieves the block's unique identifier.
   *
   * @return The block's ID.
   */
  index_t get_block_id() { return this->_block_id; }

  /**
   * @brief Retrieves the base dimensions of the block.
   *
   * @return Pointer to the block's base dimensions.
   */
  index_t *get_base_dims() { return this->_base_dims; }

  /* Setter methods */

  /**
   * @brief Sets the number of non-zero elements in the block.
   *
   * @param new_nnz_count The new count of non-zero elements.
   */
  void set_nnz_count(uint64_t new_nnz_count) { this->nnz_count = new_nnz_count; }
  /**
   * @brief Sets the dimensions of the block.
   *
   * @param new_dims Pointer to the new dimensions of the block.
   */
  void set_dims(index_t *new_dims);
  /**
   * @brief Sets the allocation status of the block's data.
   *
   * @param new_is_allocated `true` if data is allocated, `false` otherwise.
   */
  void set_is_allocated(bool new_is_allocated) { this->_is_allocated = new_is_allocated; }

public:
  unsigned short order; ///< The order (rank) of the tensor.
  index_t *dims;        ///< Dimensions of the block.
  uint64_t nnz_count;   ///< The number of non-zero elements.

  value_t *values;                        ///< Array storing the values of non-zero elements.
  index_t *indices[constants::kMaxOrder]; ///< Array of pointers to indices for each mode.

  index_t *where_nnz[constants::kMaxOrder];  ///< Metadata for tracking non-zero element locations.
  uint64_t *count_nnz[constants::kMaxOrder]; ///< Metadata for counting non-zero elements per mode.

private:
  index_t *_base_dims;   ///< Base dimensions of the block.
  uint64_t _block_id;    ///< Unique identifier for the block.
  index_t *_block_coord; ///< Coordinates of the block.
  bool _is_allocated;    ///< Flag indicating whether the block's data is allocated.

}; // class Block

} // namespace gputucker
} // namespace supertensor

#include "gputucker/block.tpp"
#endif