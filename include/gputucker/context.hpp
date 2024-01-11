#ifndef CONTEXT_HPP_
#define CONTEXT_HPP_

namespace supertensor {
namespace gputucker {

#define CONTEXT_TEMPLATE \
  template <typename TensorType>
#define CONTEXT_TEMPLATE_ARGS \
  TensorType

CONTEXT_TEMPLATE
class Context {

  using tensor_t = TensorType;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

  public:
  Context() = default;
  ~Context() = default;

  

}; // class Context

} // namespace gputucker
} // namespace supertensor

#endif /* CONTEXT_HPP_ */