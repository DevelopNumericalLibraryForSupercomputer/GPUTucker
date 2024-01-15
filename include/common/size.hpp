#ifndef _SIZE_H__
#define _SIZE_H__

#include <cstddef>

namespace common
{
    template <typename T = size_t>
    constexpr T KiB(size_t const i)
    {
        return i << 10;
    }

    template <typename T = size_t>
    constexpr T MiB(size_t const i)
    {
        return i << 20;
    }

    template <typename T = size_t>
    constexpr T GiB(size_t const i)
    {
        return i << 30;
    }

    template <typename T = size_t>
    constexpr T TiB(size_t const i)
    {
        return i << 40;
    }

    template <typename T1, typename T2>
    constexpr T1 aligned_size(const T1 size, const T2 align)
    {
        return align * ((size + align - 1) / align);
    }

}
#endif // _SIZE_H__