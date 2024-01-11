#include "gputucker/scheduler.hpp"

#include "gputucker/helper.hpp"
#include "gputucker/tensor.hpp"
#include "gputucker/constants.hpp"
// #include 'gputucker/optimizer.hpp'

#include <iostream>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace supertensor {
namespace gputucker {

  SCHEDULER_TEMPLATE
  void Scheduler<SCHEDULER_TEMPLATE_ARGS>::Initialize(unsigned short new_gpu_count) {
    this->task_count = 0;
    this->gpu_count = new_gpu_count;
    this->tasks = new std::vector<Task>[this->gpu_count];
  }

  SCHEDULER_TEMPLATE
  void Scheduler<SCHEDULER_TEMPLATE_ARGS>::Schedule(tensor_t *tensor, optimizer_t *optimizer) {
   if(optimizer->partition_type == gputucker::enums::PartitionTypes::kNonzeroPartition) {
      this->_NonzeroBasedPartitioning(tensor, optimizer);
    } else if(optimizer->partition_type == gputucker::enums::PartitionTypes::kDimensionPartition) {
      this->_DimensionbasedPartitioning(tensor, optimizer);
    } else {
      printf("Invalid partition type\n");
      exit(1);
    }

  }


  SCHEDULER_TEMPLATE
  void Scheduler<SCHEDULER_TEMPLATE_ARGS>::_NonzeroBasedPartitioning(tensor_t *tensor, optimizer_t *optimizer) {
    // Nonzero partitioning
    this->task_count = this->gpu_count;
    this->nnz_count_per_task = (tensor->nnz_count + this->task_count - 1) / this->task_count;
    uint64_t avail_nnz_count_per_task = optimizer->avail_nnz_count_per_task;
    printf("nnz count per task: %lu vs. avail nnz count %lu\n", this->nnz_count_per_task, avail_nnz_count_per_task);

    if (this->nnz_count_per_task > avail_nnz_count_per_task) {
      printf("Require iteration for task");
      this->task_count = (tensor->nnz_count + avail_nnz_count_per_task - 1) / avail_nnz_count_per_task;
      this->nnz_count_per_task = avail_nnz_count_per_task;
    }

    uint64_t block_id = 0;
    int stream_offset = 0;
    int device_id;

    uint64_t ptr;
    uint64_t task_nnz_count = this->nnz_count_per_task;
    for (uint64_t task = 0; task < this->task_count; ++task) {
      ptr = task * task_nnz_count;
      if (task == this->task_count - 1) {
        // Last task
        assert(tensor->nnz_count > ptr);
        task_nnz_count = tensor->nnz_count - ptr;
        assert(task_nnz_count <= this->nnz_count_per_task);
      }
      device_id = task % this->gpu_count;

      Task new_task = Task(block_id, task_nnz_count, ptr, stream_offset);
      tasks[device_id].push_back(new_task);

      printf("In Device [%d]\t", device_id);
      tasks[device_id][tasks[device_id].size() - 1].to_string();
    }
  }
  
  SCHEDULER_TEMPLATE
  void Scheduler<SCHEDULER_TEMPLATE_ARGS>::_DimensionBasedPartitioning(tensor_t *tensor, optimizer_t *optimizer) {
    // Dimension partitioning
    std::vector<uint64_t> sort_nnz_count, sort_block_id;
    sort_nnz_count.resize(tensor->block_count);
    sort_block_id.resize(tensor->block_count);

    for (uint64_t block_id = 0; block_id < tensor->block_count; ++block_id)
    {
      sort_block_id[block_id] = block_id;
      sort_nnz_count[block_id] = tensor->blocks[block_id]->nnz_count;
    }
    // Sort block id by the number of nonzeros
    std::sort(sort_block_id.begin(), sort_block_id.end(),
              [&](const uint64_t a, const uint64_t &b)
              {
                return (sort_nnz_count[a] < sort_nnz_count[b]);
              });
    for (uint64_t block_id = 0; block_id < tensor->block_count; ++block_id)
    {
      printf("[%lu] block has %lu nnzs.\n", sort_block_id[block_id], sort_nnz_count[sort_block_id[block_id]]);
    }

    // Dimension partitioning
    this->task_count = 0;
    int global_stream_offset = 0;
    this->nnz_count_per_task = optimizer->avail_nnz_count_per_task;

    for (uint64_t i = 0; i < tensor->block_count; ++i)
    {
      int tasks_in_block = 1;
      uint64_t block_id = sort_block_id[i];
      int total_stream_count = this->gpu_count * optimizer->cuda_stream_count;
      uint64_t nnz_count_per_stream = tensor->blocks[block_id]->nnz_count;
      uint64_t avail_nnz_count_per_stream = optimizer->avail_nnz_count_per_task;

      if (tensor->blocks[block_id]->nnz_count > avail_nnz_count_per_stream)
      {
        tasks_in_block = (tensor->blocks[block_id]->nnz_count + avail_nnz_count_per_stream - 1) / avail_nnz_count_per_stream;
        nnz_count_per_stream = avail_nnz_count_per_stream;
      }

      int stream_offset = 0;
      int device_id;
      uint64_t t = 0;

      for (t = 0; t < tasks_in_block - 1; ++t)
      {
        // printf("%d/%d tasks :: \t", t, tasks_in_block);
        if (global_stream_offset >= total_stream_count)
        {
          global_stream_offset = 0;
        }

        device_id = global_stream_offset % this->gpu_count;
        stream_offset = global_stream_offset / this->gpu_count;
        assert(device_id < this->gpu_count);
        assert(stream_offset < optimizer->cuda_stream_count);

        std::cout << "device id: " << device_id;
        std::cout << "\t stream offset: " << stream_offset;

        tasks[device_id].push_back(Task(block_id, nnz_count_per_stream, t * nnz_count_per_stream, stream_offset));
        ++this->task_count;

        printf("[%d] block in Device [%d] - %d th stream\n", block_id, device_id, stream_offset);
        // tasks[device_id][tasks[device_id].size() - 1].to_string();
        ++global_stream_offset;
      }
      // last nnz_cout
      printf("last :: \t");
      uint64_t last_nnz_count = tensor->blocks[block_id]->nnz_count - (t * nnz_count_per_stream);
      assert(last_nnz_count >= 0);
      assert(last_nnz_count <= nnz_count_per_stream);

      if (global_stream_offset >= total_stream_count)
      {
        global_stream_offset = 0;
      }
      device_id = global_stream_offset % this->gpu_count;
      stream_offset = global_stream_offset / this->gpu_count;
      assert(device_id < this->gpu_count);

      std::cout << "device id: " << device_id;
      std::cout << "\t stream offset: " << stream_offset;

      tasks[device_id].push_back(Task(block_id, last_nnz_count, t * nnz_count_per_stream, stream_offset));
      tasks[device_id][0].to_string();
      ++this->task_count;

      // printf("In Device [%d] - %d th stream\n", device_id, stream_offset);
      // tasks[device_id][tasks[device_id].size() - 1].to_string();
      ++global_stream_offset;
    }
  }
  
}
}