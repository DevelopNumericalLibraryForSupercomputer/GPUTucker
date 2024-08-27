#ifndef TENSOR_MANAGER_HPP_
#define TENSOR_MANAGER_HPP_

namespace supertensor {
namespace gputucker {

/**
 * @brief Parses tensor data from a file.
 *
 * This method reads tensor data from the specified file and initializes a tensor object.
 *
 * @param file_name The name of the file containing the tensor data.
 * @param tensor Pointer to a pointer where the tensor object will be stored.
 * @return `true` if the file is successfully parsed, `false` otherwise.
 */
template <typename TensorType>
bool ParseFromFile(const std::string &file_name, TensorType **tensor);

/**
 * @brief Creates tensor blocks for Tucker decomposition.
 *
 * This method uses the source tensor and an optimizer to create tensor blocks,
 * which are necessary for the Tucker decomposition process.
 *
 * @tparam OptimizerType The type of the optimizer used for block creation.
 * @param src Pointer to the source tensor.
 * @param dest Pointer to a pointer where the destination tensor blocks will be stored.
 * @param optimizer Pointer to the optimizer used to create the tensor blocks.
 */
template <typename TensorType, typename OptimizerType>
void CreateTensorBlocks(TensorType **src, TensorType **dest, OptimizerType *optimizer);

/**
 * @brief Reads tensor data from a buffer.
 *
 * This private method reads tensor data from a buffer and initializes a tensor object.
 *
 * @param buffer The buffer containing the tensor data.
 * @param buffer_length The length of the buffer.
 * @param tensor Pointer to a pointer where the tensor object will be stored.
 * @return `true` if the data is successfully read, `false` otherwise.
 */
template <typename TensorType>
bool _ReadData(const char *buffer, const size_t buffer_length, TensorType **tensor);


} // namespace gputucker
} // namespace supertensor
#include "gputucker/tensor_manager.tpp"
#endif // TENSOR_READER_HPP_