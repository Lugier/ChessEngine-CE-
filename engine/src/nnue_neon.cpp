// ARM64 NEON accelerated dot product with scalar fallback.
#include "nnue.hpp"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace cortex {

int32_t dot_i16_i8(const int16_t* a, const int8_t* b, int n) {
  int32_t sum = 0;
#if defined(__ARM_NEON)
  int i = 0;
  int32x4_t acc = vdupq_n_s32(0);
  for (; i + 8 <= n; i += 8) {
    int16x8_t va = vld1q_s16(a + i);
    int8x8_t vb8 = vld1_s8(b + i);
    int16x8_t vb = vmovl_s8(vb8);
    int32x4_t lo = vmull_s16(vget_low_s16(va), vget_low_s16(vb));
    int32x4_t hi = vmull_s16(vget_high_s16(va), vget_high_s16(vb));
    acc = vaddq_s32(acc, lo);
    acc = vaddq_s32(acc, hi);
  }
  sum += vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) + vgetq_lane_s32(acc, 2) +
         vgetq_lane_s32(acc, 3);
  for (; i < n; ++i) sum += int32_t(a[i]) * int32_t(b[i]);
#else
  for (int i = 0; i < n; ++i) sum += int32_t(a[i]) * int32_t(b[i]);
#endif
  return sum;
}

}  // namespace cortex
