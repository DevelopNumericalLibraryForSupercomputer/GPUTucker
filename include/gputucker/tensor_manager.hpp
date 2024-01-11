#ifndef TENSOR_MANAGER_HPP_
#define TENSOR_MANAGER_HPP_

namespace supertensor {
namespace gputucker {

#define TENSOR_MANAGER_TEMPLATE template <typename TensorType>
#define TENSOR_MANAGER_ARGS TensorType

TENSOR_MANAGER_TEMPLATE
class TensorManager {
  using tensor_t = TensorType;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

 public:
  TensorManager();
  ~TensorManager();
  
  bool ParseFromFile(const std::string &file_name, tensor_t** tensor);
  template<typename OptimizerType>
  void CreateTensorBlocks(tensor_t** src, tensor_t** dest, OptimizerType* optimizer);

private:
  bool _ReadData(const char *buffer, const size_t buffer_length, tensor_t** tensor);

};  // class TensorMANAGER
}  // namespace gputucker
}  // namespace supertensor
#include "gputucker/tensor_manager.tpp"
#endif  // TENSOR_READER_HPP_