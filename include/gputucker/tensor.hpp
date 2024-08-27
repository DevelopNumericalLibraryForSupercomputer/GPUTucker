#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "gputucker/block.hpp"

namespace supertensor {
namespace gputucker {

#define TENSOR_TEMPLATE template <typename BlockType>
#define TENSOR_TEMPLATE_ARGS BlockType

/**
 * @brief Tensor class for Tucker decomposition.
 *
 * This class represents a tensor used in the Tucker decomposition program. It manages
 * the creation of blocks for the tensor data and handles the insertion of data into
 * these blocks. The class also provides various methods for managing and accessing
 * tensor properties and dimensions.
 *
 * @tparam BlockType The type of block used to represent sub-tensors in the decomposition.
 *
 * @details The Tensor class is central to the Tucker decomposition process, as it organizes
 * the tensor data into blocks and provides methods for manipulating and querying the tensor's
 * structure.
 *
 * @version 1.0.0
 * @date 2023-08-10
 */
TENSOR_TEMPLATE
class Tensor {
public:
  using this_t = Tensor<TENSOR_TEMPLATE_ARGS>; ///< Type alias for this tensor type.
  using block_t = BlockType;                   ///< Type alias for the block type.
  using index_t = typename block_t::index_t;   ///< Type alias for the index type.
  using value_t = typename block_t::value_t;   ///< Type alias for the value type.

public:
  /**
   * @brief Constructs a Tensor with a specified order.
   *
   * @param new_order The number of dimensions (order) of the tensor.
   */
  Tensor(unsigned short new_order);

  /**
   * @brief Copy constructor for Tensor.
   *
   * @param other Pointer to another Tensor to copy.
   */
  Tensor(this_t *other);

  /**
   * @brief Default constructor for Tensor.
   */
  Tensor();

  /**
   * @brief Destructor for Tensor.
   */
  ~Tensor();

  /**
   * @brief Creates blocks for the tensor data.
   *
   * This method initializes blocks for the tensor based on a specified block count
   * and a histogram of non-zero elements.
   *
   * @param new_block_count The number of blocks to create.
   * @param histogram A histogram of non-zero elements used to define the blocks.
   */
  void MakeBlocks(uint64_t new_block_count, uint64_t *histogram);

  /**
   * @brief Inserts data into a specific block of the tensor.
   *
   * This method inserts indices and values into the specified block of the tensor.
   *
   * @param block_id The ID of the block where data will be inserted.
   * @param indices Array of indices for the data.
   * @param values Array of values to insert.
   */
  void InsertData(uint64_t block_id, index_t *indices[], value_t *values);

  /**
   * @brief Assigns indices to each block.
   *
   * This method assigns the appropriate indices to each block of the tensor,
   * ensuring that data is correctly organized for decomposition.
   */
  void AssignIndicesOfEachBlock();

  /**
   * @brief Converts the tensor's state to a string representation.
   *
   * This method is useful for debugging or logging the tensor's current state.
   */
  void ToString();

  /* Setter methods */

  /**
   * @brief Sets the dimensions of the tensor.
   *
   * @param new_dims Pointer to an array representing the dimensions of the tensor.
   */
  void set_dims(index_t *new_dims);

  /**
   * @brief Sets the partition dimensions of the tensor.
   *
   * @param new_partition_dims Pointer to an array representing the partition dimensions.
   */
  void set_partition_dims(const index_t *new_partition_dims);

  /**
   * @brief Sets the number of non-zero elements in the tensor.
   *
   * @param new_nnz_count The new count of non-zero elements.
   */
  void set_nnz_count(uint64_t new_nnz_count) { nnz_count = new_nnz_count; }

  /* Getter methods */

  /**
   * @brief Gets the maximum partition dimension.
   *
   * @return The maximum partition dimension.
   */
  index_t get_max_partition_dim() { return this->_max_partition_dim; }

  /**
   * @brief Gets the maximum block dimension.
   *
   * @return The maximum block dimension.
   */
  index_t get_max_block_dim() { return this->_max_block_dim; }

  /**
   * @brief Gets the maximum number of non-zero elements in any block.
   *
   * @return The maximum number of non-zero elements in any block.
   */
  index_t get_max_nnz_count_in_block() { return this->_max_nnz_count_in_block; }

private:
  /**
   * @brief Converts a block ID to block coordinates.
   *
   * This private method translates a block ID into its corresponding block coordinates
   * within the tensor.
   *
   * @param block_id The ID of the block.
   * @param coord Pointer to an array where the block coordinates will be stored.
   */
  void _BlockIDtoBlockCoord(uint64_t block_id, index_t *coord);

  /**
   * @brief Refreshes the dimensions of the tensor.
   *
   * This method updates the internal dimensions of the tensor, ensuring they are consistent
   * with the current state of the tensor.
   */
  void _RefreshDims();

public:
  /* Tensor Description */
  unsigned short order; ///< The number of dimensions (order) of the tensor.
  index_t *dims;        ///< Array representing the dimensions of the tensor.
  uint64_t nnz_count;   ///< The number of non-zero elements in the tensor.
  value_t norm;         ///< The norm of the tensor.

  /* Block Description */
  index_t *partition_dims; ///< The dimensions used for partitioning the tensor.
  index_t *block_dims;     ///< The dimensions of each block in the tensor.
  uint64_t block_count;    ///< The total number of blocks in the tensor.
  block_t **blocks;        ///< Array of pointers to blocks representing sub-tensors.

private:
  uint64_t _max_nnz_count_in_block; ///< The maximum number of non-zero elements in any block.
  index_t _max_block_dim;           ///< The maximum dimension size among the blocks.
  uint64_t _empty_block_count;      ///< The count of empty blocks.
  index_t _max_partition_dim;       ///< The maximum partition dimension.
}; // class Tensor

} // namespace gputucker
} // namespace supertensor
#include "gputucker/tensor.tpp"
#endif /* TENSOR_HPP_ */