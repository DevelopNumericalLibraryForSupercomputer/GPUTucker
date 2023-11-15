#ifndef CONSTANTS_HPP_
#define CONSTANTS_HPP_

namespace supertensor {
namespace gputucker {
namespace constants {
constexpr int kMaxOrder{8};
constexpr int kMaxIteration{1};
constexpr double kLambda{0.0001f};
}  // namespace constants

namespace enums {
enum PartitionTypes {
  kDimensionPartition,  // Large-scale with CUDA streaming
  kNonzeroPartition,    // Small-scale without CUDA streaming
  kPartitionTypeCount
};
}
}  // namespace gputucker
}  // namespace supertensor
#endif /* CONSTANTS_HPP_ */