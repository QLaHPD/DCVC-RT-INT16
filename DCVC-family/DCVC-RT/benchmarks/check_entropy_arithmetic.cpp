// Compare reciprocal rANS updates with the original integer division, including
// emitted renormalization bytes, for every nonzero uint16 frequency.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "../src/cpp/py_rans/rans_byte.h"

int main()
{
    uint32_t random = 0x12345678;
    uint64_t cases = 0;
    for (uint32_t freq = 1; freq < 65536; ++freq) {
        const uint32_t reciprocal = freq > 1 ? uint32_t((uint64_t(1) << 32) / freq) : 0;
        const uint32_t threshold = freq << ENC_RENORM_SHIFT_BITS;
        for (int sample = 0; sample < 128; ++sample) {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            const uint32_t edges[] = {0, 1, freq-1, freq, freq+1, threshold-1,
                                      threshold, threshold+1, 0x7fffffff, 0xffffffff};
            const uint32_t state = sample < 10 ? edges[sample] : random;
            RansState expected = state, actual = state;
            uint8_t old_buffer[32]{}, new_buffer[32]{};
            uint8_t *old_ptr = old_buffer+32, *new_ptr = new_buffer+32;
            const uint32_t start = 65536-freq;
            RansEncPut(expected, old_ptr, start, freq);
            RansEncPutReciprocal(actual, new_ptr, start, freq, reciprocal);
            const auto old_size = old_buffer+32-old_ptr;
            const auto new_size = new_buffer+32-new_ptr;
            if (expected != actual || old_size != new_size ||
                std::memcmp(old_ptr, new_ptr, old_size) != 0) {
                std::printf("FAIL freq=%u state=%u\n", freq, state);
                return 1;
            }
            ++cases;
        }
    }
    std::printf("PASS: %llu exact rANS states/byte sequences across all 65,535 frequencies\n",
                static_cast<unsigned long long>(cases));
}
