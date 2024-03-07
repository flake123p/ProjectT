#pragma once

#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>

inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32 = {f};
    return fp32.as_bits;
}

inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32 = {w};
    return fp32.as_value;
}

inline float fp16_ieee_to_fp32_value(uint16_t h) {

    const uint32_t w = (uint32_t)h << 16;

    // Extract the sign of the input number into the high bit of the 32-bit word:
    const uint32_t sign = w & UINT32_C(0x80000000);

    // Extract mantissa and biased exponent of the input number into the high bits of the 32-bit word:
    const uint32_t two_w = w + w;

    // First, we adjust the exponent by (0xFF - 0x1F) = 0xE0 (see exp_offset below) 
    // rather than by 0x70 suggested by the difference in the exponent bias (see above).
    constexpr uint32_t exp_offset = UINT32_C(0xE0) << 23;

    // const float exp_scale = 0x1.0p-112f;
    constexpr uint32_t scale_bits = (uint32_t)15 << 23;
    float exp_scale_val = 0;

    memcpy(&exp_scale_val, &scale_bits, sizeof(exp_scale_val));

    const float exp_scale = exp_scale_val;
    const float normalized_value =
        fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    // Convert denormalized half-precision inputs into single-precision results
    constexpr uint32_t magic_mask = UINT32_C(126) << 23;
    constexpr float magic_bias = 0.5f;
    const float denormalized_value =
        fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    constexpr uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                    : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

inline uint16_t fp16_ieee_from_fp32_value(float f) {
    // const float scale_to_inf = 0x1.0p+112f;
    // const float scale_to_zero = 0x1.0p-110f;
    constexpr uint32_t scale_to_inf_bits = (uint32_t)239 << 23;
    constexpr uint32_t scale_to_zero_bits = (uint32_t)17 << 23;
    float scale_to_inf_val = 0, scale_to_zero_val = 0;

    memcpy(&scale_to_inf_val, &scale_to_inf_bits, sizeof(scale_to_inf_val));
    memcpy(&scale_to_zero_val, &scale_to_zero_bits, sizeof(scale_to_zero_val));
    
    const float scale_to_inf = scale_to_inf_val;
    const float scale_to_zero = scale_to_zero_val;

    // #if defined(_MSC_VER) && _MSC_VER == 1916
    //   float base = ((signbit(f) != 0 ? -f : f) * scale_to_inf) * scale_to_zero;
    // #else
    //   float base = (fabsf(f) * scale_to_inf) * scale_to_zero;
    // #endif

    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);

    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return static_cast<uint16_t>(
        (sign >> 16) |
        (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}
