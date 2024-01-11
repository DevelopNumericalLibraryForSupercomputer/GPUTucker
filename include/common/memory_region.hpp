#ifndef _MEMORY_REGION_H__
#define _MEMORY_REGION_H__

#include <cstdint>
#include <cstdlib>
#include <cassert>

#include "common/size.hpp"

namespace common
{

#define MEMORY_REGION_TEMPLATE \
  template <typename T>

#define MEMORY_REGION_TEMPLATE_ARGS \
  T

  MEMORY_REGION_TEMPLATE
  class MemoryRegion
  {

  public:
    using this_t = common::MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>;

    MemoryRegion(void *new_ptr, uint64_t new_size, unsigned new_count);
    MemoryRegion(uint64_t new_size, unsigned new_count);
    MemoryRegion();
    ~MemoryRegion() = default;

    void Initialize(void *new_ptr, uint64_t new_size, unsigned new_count);
    bool IsNull() const;
    void set_memory_region(this_t *other);
    void shift_ptr(uint64_t shift_size);
    void set_ptr(void *new_ptr);
    void set_size(uint64_t new_size);
    void set_count(unsigned new_count);
    T *get_ptr(unsigned offset);
    T *get_shift_ptr();
    uint64_t get_total_size() const; // total memory size
    uint64_t get_size() const;       // 1 region size
    unsigned get_count();

  private:
    T *_base_ptr;
    T *_shift_ptr;
    uint64_t _size;         // per 1 region
    uint64_t _aligned_size; // per 1 region
    unsigned _count;        // depending on the number of CUDA streams for device memory region
  };

  MEMORY_REGION_TEMPLATE
  MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::MemoryRegion(void *new_ptr, uint64_t new_size, unsigned new_count)
  {
    initialize(new_ptr, new_size, new_count);
  }
  MEMORY_REGION_TEMPLATE
  MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::MemoryRegion(uint64_t new_size, unsigned new_count) : MemoryRegion(NULL, new_size, new_count)
  {
  }
  MEMORY_REGION_TEMPLATE
  MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::MemoryRegion() : MemoryRegion(NULL, 0, 1)
  {
  }
  MEMORY_REGION_TEMPLATE
  void MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::Initialize(void *new_ptr, uint64_t new_size, unsigned new_count)
  {
    this->set_ptr(new_ptr);
    this->set_size(new_size);
    this->set_count(new_count);
  }

  MEMORY_REGION_TEMPLATE
  bool MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::IsNull() const
  {
    return this->_base_ptr == NULL;
  }
  MEMORY_REGION_TEMPLATE
  void MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::set_memory_region(this_t *other)
  {
    initialize(other->get_ptr(0), other->get_size(), other->get_count());
  }

  MEMORY_REGION_TEMPLATE
  void MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::shift_ptr(uint64_t shift_size)
  {
    assert(this->_shift_ptr != NULL);
    uint64_t aligned_shift_size = common::aligned_size(shift_size, 256);
    void *tmp_ptr = (void *)(this->_shift_ptr);
    tmp_ptr = (void *)((uintptr_t)(tmp_ptr) + aligned_shift_size);
    this->_shift_ptr = static_cast<T *>(tmp_ptr);
  }

  MEMORY_REGION_TEMPLATE
  void MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::set_ptr(void *new_ptr)
  {
    this->_base_ptr = static_cast<T *>(new_ptr);
    this->_shift_ptr = this->_base_ptr;
  }
  MEMORY_REGION_TEMPLATE
  void MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::set_size(uint64_t new_size)
  {
    this->_size = new_size;
    this->_aligned_size = common::aligned_size(new_size, 256); // cudaAligmentBytes: 256
  }
  MEMORY_REGION_TEMPLATE
  void MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::set_count(unsigned new_count)
  {
    this->_count = new_count;
  }

  MEMORY_REGION_TEMPLATE
  T *MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::get_ptr(unsigned offset)
  {
    if (is_null())
    {
      return NULL;
    }

    assert(offset < this->_count);
    void *tmp_ptr = (void *)(this->_base_ptr);
    tmp_ptr = (void *)((uintptr_t)(tmp_ptr) + this->_aligned_size * offset);
    return static_cast<T *>(tmp_ptr);
  }

  MEMORY_REGION_TEMPLATE
  T *MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::get_shift_ptr()
  {
    void *tmp_ptr = (void *)(this->_shift_ptr);
    tmp_ptr = (void *)((uintptr_t)(tmp_ptr));
    return static_cast<T *>(tmp_ptr);
  }

  MEMORY_REGION_TEMPLATE
  uint64_t MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::get_total_size() const
  {
    assert(this->_aligned_size > 0);
    assert(this->_count > 0);
    return this->_aligned_size * this->_count;
  }

  MEMORY_REGION_TEMPLATE
  uint64_t MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::get_size() const
  {
    assert(this->_aligned_size > 0);
    return this->_aligned_size;
  }

  MEMORY_REGION_TEMPLATE
  unsigned MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::get_count()
  {
    return this->_count;
  }
} // namespace common

#endif //_MEMORY_REGION_H__