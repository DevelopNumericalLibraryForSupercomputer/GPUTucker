#ifndef TENSOR_READER_HPP_
#define TENSOR_READER_HPP_

namespace supertensor {

namespace gputucker {
#define TENSOR_READER_TEMPLATE template <typename TensorType>

#define TENSOR_READER_ARGS TensorType

TENSOR_READER_TEMPLATE
class TensorReader {
  using tensor_t = TensorType;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

 public:
  TensorReader(unsigned short new_order);
  TensorReader();
  ~TensorReader();
  bool parse_from_memory(const char *buffer, const size_t buffer_length);
  bool parse_from_file(const std::string &file_name);
  void to_string();
  // void make_tensor(tensor_t *tensor);

 protected:
  void _read_data(const char *buffer, const size_t buffer_length);
  void _count_nonzeros_per_block(std::vector<uint64_t> &global_histogram);

 public:
  tensor_t *data;
};  // class TensorReader

}  // namespace gputucker
}  // namespace supertensor
#include "gputucker/tensor_reader.tpp"
#endif  // TENSOR_READER_HPP_