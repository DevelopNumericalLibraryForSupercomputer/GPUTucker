#include <omp.h>

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "common/human_readable.hpp"
#include "gputucker/helper.hpp"
#include "gputucker/tensor_reader.hpp"

namespace supertensor {
namespace gputucker {
TENSOR_READER_TEMPLATE
TensorReader<TENSOR_READER_ARGS>::TensorReader(unsigned short new_order) {
  data = new tensor_t(new_order);
}

TENSOR_READER_TEMPLATE
TensorReader<TENSOR_READER_ARGS>::TensorReader() : TensorReader(0) {}

TENSOR_READER_TEMPLATE
bool TensorReader<TENSOR_READER_ARGS>::parse_from_memory(
    const char *buffer, const size_t buffer_length) {
  if (buffer_length < 1) {
    throw std::runtime_error(ERROR_LOG("[ERROR] Buffer length is too short."));
    return false;
  }
  if (this->data->order < 1) {
    throw std::runtime_error(
        ERROR_LOG("[ERROR] Tensor order should be larger than 1."));
    return false;
  }
  MYPRINT("... 0) Read the input tensor\n");
  this->_read_data(buffer, buffer_length);

  return true;
}

TENSOR_READER_TEMPLATE
bool TensorReader<TENSOR_READER_ARGS>::parse_from_file(
    const std::string &file_name) {
  std::ifstream file(file_name);
  if (!file.is_open()) {
    throw std::runtime_error(ERROR_LOG("[ERROR] Canot open file \"" +
                                       file_name + "\" for reading..."));
    return false;
  }

  file.seekg(0, file.end);
  size_t file_size = static_cast<size_t>(file.tellg());
  file.seekg(0, file.beg);

  std::cout << "Input Size \t: "
            << common::HumanReadable{(std::uintmax_t)file_size} << std::endl;

  // The file is now loaded in the memory buffer
  std::string buffer(file_size, '\0');
  file.read(&buffer[0], file_size);

  file.close();

  return parse_from_memory(buffer.c_str(), file_size);
}
TENSOR_READER_TEMPLATE
void TensorReader<TENSOR_READER_ARGS>::to_string() { this->data->to_string(); }

TENSOR_READER_TEMPLATE
void TensorReader<TENSOR_READER_ARGS>::_read_data(const char *buffer,
                                                  const size_t buffer_length) {
  int thread_id = 0;
  int thread_count = 0;
  int order = this->data->order;

  std::vector<uint64_t> *pos;
  std::vector<index_t> *local_max_dims;
  std::vector<index_t> *local_dim_offset;
  std::vector<uint64_t> nnz_prefix_sum;

  index_t *global_max_dims;
  uint64_t global_nnz_count = 0;
  value_t *values;

  index_t *indices[order];

#pragma omp parallel private(thread_id)
  {
    thread_id = omp_get_thread_num();
    thread_count = omp_get_num_threads();

// Initialize local variables
#pragma omp single
    {
      pos = new std::vector<uint64_t>[thread_count];
      local_max_dims = new std::vector<index_t>[thread_count];
      local_dim_offset = new std::vector<index_t>[thread_count];
      nnz_prefix_sum.resize(thread_count);
    }
    pos[thread_id].push_back(0);
    local_max_dims[thread_id].resize(order);
    local_dim_offset[thread_id].resize(order);

    for (unsigned short axis = 0; axis < order; ++axis) {
      local_max_dims[thread_id][axis] = std::numeric_limits<index_t>::min();
      local_dim_offset[thread_id][axis] = std::numeric_limits<index_t>::max();
    }
    // 1. Find '\n' : the number of nonzeros
#pragma omp for reduction(+ : global_nnz_count)
    for (size_t sz = 0; sz < buffer_length; ++sz) {
      if (buffer[sz] == '\n') {
        global_nnz_count++;
        pos[thread_id].push_back(sz + 1);
      }
    }

#pragma omp barrier
    if (thread_id > 0) {
      pos[thread_id].front() = pos[thread_id - 1].back();
    }
#pragma omp barrier
#pragma omp single
    {
      // prefix sum
      nnz_prefix_sum[0] = 0;
      for (int tid = 1; tid < thread_count; ++tid) {
        nnz_prefix_sum[tid] =
            nnz_prefix_sum[tid - 1] + (pos[tid - 1].size() - 1);
      }
      assert(nnz_prefix_sum.back() + pos[thread_count - 1].size() - 1 ==
             global_nnz_count);

      global_max_dims = gputucker::allocate<index_t>(order);
      for (unsigned short axis = 0; axis < order; ++axis) {
        global_max_dims[axis] = std::numeric_limits<index_t>::min();
        indices[axis] = gputucker::allocate<index_t>(global_nnz_count);
      }
      values = gputucker::allocate<value_t>(global_nnz_count);
    }

    for (uint64_t nnz = 1; nnz < pos[thread_id].size(); ++nnz) {
      // Calculate the stargin position of the current slice in the buffer
      const int len = pos[thread_id][nnz] - pos[thread_id][nnz - 1] - 1;
      uint64_t buff_ptr = pos[thread_id][nnz - 1];
      char *buff = const_cast<char *>(&buffer[buff_ptr]);

      // Tokenize	the slice by newline characters
      char *rest = strtok_r(buff, "\n", &buff);

      char *token;
      if (rest != NULL) {
        // Calculate the offset of the current thread in the global index
        uint64_t offset = nnz_prefix_sum[thread_id];
        unsigned short axis = 0;
        /* Coordinate */
        // Loop through each coordinate in the slice
        while ((token = strtok_r(rest, " \t", &rest)) && (axis < order)) {
          index_t idx = strtoull(token, NULL, 10);

          // Update the maximum and minimum indices for the current axis
          local_max_dims[thread_id][axis] =
              std::max<index_t>(local_max_dims[thread_id][axis], idx);
          local_dim_offset[thread_id][axis] =
              std::min<index_t>(local_dim_offset[thread_id][axis], idx);

          // Store the current index in the global indices array, with
          // 1-indexing (subtract 1 from idx)

          indices[axis][offset + nnz - 1] = idx - 1;  // 1-Indexing
          ++axis;
        }  // !while

        /* Value */
        // Parse the value of the current slice
        value_t val;
        if (token != NULL) {
          val = std::stod(token);
        } else {
          // If the slice does not have a value,
          // generate a random one between 0 and 1
          val = gputucker::frand<value_t>(0, 1);
        }
        values[offset + nnz - 1] = val;
      }
    }  // !for
// 2. extract metadata for the tensor (dims, offsets, and #nnzs)
#pragma omp critical
    {
      // Update the global max dimension for the axis
      for (unsigned short axis = 0; axis < order; ++axis) {
        global_max_dims[axis] = std::max<index_t>(
            global_max_dims[axis], local_max_dims[thread_id][axis]);
        if (local_dim_offset[thread_id][axis] < 1) {
          // outputs are based on base-0 indexing
          throw std::runtime_error(ERROR_LOG(
              "We note that input tensors must follow base-1 indexing"));
        }
      }
    }

  }  //! omp

  uint64_t block_id = 0;

  this->data->set_dims(global_max_dims);
  this->data->set_nnz_count(global_nnz_count);

  index_t *new_partition_dims = gputucker::allocate<index_t>(order);
  uint64_t *histogram = gputucker::allocate<uint64_t>(1);
  histogram[0] = global_nnz_count;
  for (unsigned axis = 0; axis < order; ++axis) {
    new_partition_dims[axis] = 1;
  }
  this->data->make_blocks(new_partition_dims, histogram);
  this->data->set_data(block_id, &indices[0], values);

  // Deallocate
  delete[] pos;
  delete[] local_max_dims;
  delete[] local_dim_offset;
  nnz_prefix_sum.clear();
  std::vector<uint64_t>().swap(nnz_prefix_sum);

  gputucker::deallocate<index_t>(new_partition_dims);
  gputucker::deallocate<uint64_t>(histogram);
}

TENSOR_READER_TEMPLATE
void TensorReader<TENSOR_READER_ARGS>::_count_nonzeros_per_block(
    std::vector<uint64_t> &global_histogram) {
  MYPRINT("... 2) Counting nonzeros per block\n");

  uint64_t nnz_count = this->data->nnz_count;
  uint64_t block_count = this->data->block_count;
  int order = this->data->order;

  std::vector<uint64_t> *local_histogram;
  std::vector<index_t> *local_coord;
  int thread_count = 0;
  int thread_id = 0;

#pragma omp parallel private(thread_id)
  {
    thread_id = omp_get_thread_num();
    thread_count = omp_get_num_threads();

#pragma omp single
    {
      local_histogram = new std::vector<uint64_t>[thread_count];
      local_coord = new std::vector<index_t>[thread_count];
    }

    local_histogram[thread_id].resize(block_count);
    local_coord[thread_id].resize(order);

    for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
      local_histogram[thread_id][block_id] = 0;
    }
#pragma omp barrier

#pragma omp for
    for (uint64_t nnz = 0; nnz < nnz_count; ++nnz) {
      uint64_t block_id = this->data->offset_to_block_id(nnz);
      local_histogram[thread_id][block_id]++;
    }
  }  // !omp parallel

  for (index_t block_id = 0; block_id < block_count; ++block_id) {
    for (int tid = 0; tid < thread_count; ++tid) {
      global_histogram[block_id] += local_histogram[tid][block_id];
    }
  }
}
}  // namespace gputucker
}  // namespace supertensor