#ifndef CONSTANTS_HPP_
#define CONSTANTS_HPP_

namespace supertensor {
namespace gputucker {

/**
 * @brief Contains constant values used in the Tucker decomposition algorithm.
 *
 * This namespace defines a set of constant values that are used throughout
 * the Tucker decomposition implementation. These constants are used to configure
 * various aspects of the algorithm, such as the maximum order of tensors, the
 * maximum number of iterations, and regularization parameters.
 */
namespace constants {
constexpr int kMaxOrder{8};        ///< Maximum order (rank) of tensors supported.
constexpr int kMaxIteration{3};    ///< Maximum number of iterations for the algorithm.
constexpr double kLambda{0.0001f}; ///< Regularization parameter used in the decomposition.
} // namespace constants

/**
 * @brief Enumerations used in the Tucker decomposition algorithm.
 *
 * This namespace contains enumerations that define various types of partitions
 * used during the Tucker decomposition process. The partitions determine how
 * the tensor data is divided for processing, especially when utilizing CUDA
 * streaming for large-scale tensors.
 */
namespace enums {

/**
 * @brief Enumeration of partition types used in the decomposition.
 *
 * These enumeration values indicate different strategies for partitioning the
 * tensor data. The chosen partitioning method can affect the efficiency and
 * performance of the Tucker decomposition, particularly in GPU-accelerated environments.
 */
enum PartitionTypes {
  kDimensionPartition, ///< Partitioning based on tensor dimensions, suitable for large-scale tensors with CUDA streaming.
  kNonzeroPartition,   ///< Partitioning based on non-zero elements, suitable for small-scale tensors without CUDA streaming.
  kPartitionTypeCount  ///< The total count of partition types available.
};

} // namespace enums

} // namespace gputucker
} // namespace supertensor

#endif /* CONSTANTS_HPP_ */
