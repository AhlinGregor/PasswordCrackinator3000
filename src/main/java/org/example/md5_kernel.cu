#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include "common_kernel.h"


// === Device Constants ===
__device__ __constant__ char d_charset[64];
__device__ __constant__ int d_charsetLen;
__device__ __constant__ unsigned char d_targetHash[16];
__device__ bool d_found;
__device__ char d_result[8];

// === MD5 Helper Macros ===
__device__ inline uint32_t F(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
__device__ inline uint32_t G(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
__device__ inline uint32_t H(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
__device__ inline uint32_t I(uint32_t x, uint32_t y, uint32_t z) { return y ^ (x | ~z); }
__device__ inline uint32_t rotate_left(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

__device__ void md5(const unsigned char* input, int len, unsigned char* output) {
    // MD5 constants
    uint32_t a0 = 0x67452301;
    uint32_t b0 = 0xefcdab89;
    uint32_t c0 = 0x98badcfe;
    uint32_t d0 = 0x10325476;

    uint8_t msg[64] = {0};
    memcpy(msg, input, len);
    msg[len] = 0x80;
    uint64_t bit_len = len * 8;
    msg[56] = bit_len & 0xFF;
    msg[57] = (bit_len >> 8) & 0xFF;
    msg[58] = (bit_len >> 16) & 0xFF;
    msg[59] = (bit_len >> 24) & 0xFF;

    uint32_t* w = (uint32_t*)msg;

    uint32_t a = a0, b = b0, c = c0, d = d0;

    const uint32_t s[] = {
        7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
        5,9,14,20, 5,9,14,20, 5,9,14,20, 5,9,14,20,
        4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
        6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21
    };

    const uint32_t K[] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
        0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
        0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
    };

    for (int i = 0; i < 64; ++i) {
        uint32_t f, g;
        if (i < 16) {
            f = F(b, c, d);
            g = i;
        } else if (i < 32) {
            f = G(b, c, d);
            g = (5 * i + 1) % 16;
        } else if (i < 48) {
            f = H(b, c, d);
            g = (3 * i + 5) % 16;
        } else {
            f = I(b, c, d);
            g = (7 * i) % 16;
        }
        uint32_t temp = d;
        d = c;
        c = b;
        b = b + rotate_left(a + f + K[i] + w[g], s[i]);
        a = temp;
    }

    a0 += a;
    b0 += b;
    c0 += c;
    d0 += d;

    memcpy(output, &a0, 4);
    memcpy(output + 4, &b0, 4);
    memcpy(output + 8, &c0, 4);
    memcpy(output + 12, &d0, 4);
}

// === Password permutation & cracking ===
// __device__ void indexToPassword(unsigned long long index, int length, char* out) {
//     for (int i = 0; i < length; ++i) {
//         out[length - 1 - i] = d_charset[index % d_charsetLen];
//         index /= d_charsetLen;
//     }
//     out[length] = '\0';
// }

__global__ void crackMd5Kernel(int length, unsigned long long totalCombos) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;

    char candidate[8];
    unsigned char hash[16];

    for (unsigned long long i = tid; i < totalCombos && !d_found; i += stride) {
        indexToPassword(i, length, candidate);
        md5((unsigned char*)candidate, length, hash);

        bool match = true;
        for (int j = 0; j < 16; ++j) {
            if (hash[j] != d_targetHash[j]) {
                match = false;
                break;
            }
        }

        if (match && !d_found) {
            d_found = true;
            for (int j = 0; j <= length; j++) d_result[j] = candidate[j];
        }
    }
}

extern "C" void launchMd5CrackKernel(const char* charset, int length, const unsigned char* targetHash, char* hostResult) {
    int charsetLen = strlen(charset);
    unsigned long long total = 1;
    for (int i = 0; i < length; ++i)
        total *= charsetLen;

    cudaMemcpyToSymbol(d_charset, charset, charsetLen);
    cudaMemcpyToSymbol(d_charsetLen, &charsetLen, sizeof(int));
    cudaMemcpyToSymbol(d_targetHash, targetHash, 16);
    bool found = false;
    cudaMemcpyToSymbol(d_found, &found, sizeof(bool));

    crackMd5Kernel<<<32, 256>>>(length, total);
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(hostResult, d_result, 8);
}
