#include <cuda_runtime.h>
#include <string.h>
#include <stdint.h>
#include "common_kernel.h"


// === Device Constants ===
__device__ __constant__ char d_charset[64];
__device__ __constant__ int d_charsetLen;
__device__ __constant__ unsigned char d_targetHash[32];
__device__ bool d_found;
__device__ char d_result[8];

// === Device-side SHA-256 implementation ===
__device__ __constant__ uint32_t k[64] = {
    0x428a2f98ul,0x71374491ul,0xb5c0fbcful,0xe9b5dba5ul,0x3956c25bul,0x59f111f1ul,0x923f82a4ul,0xab1c5ed5ul,
    0xd807aa98ul,0x12835b01ul,0x243185beul,0x550c7dc3ul,0x72be5d74ul,0x80deb1feul,0x9bdc06a7ul,0xc19bf174ul,
    0xe49b69c1ul,0xefbe4786ul,0x0fc19dc6ul,0x240ca1ccul,0x2de92c6ful,0x4a7484aaul,0x5cb0a9dcul,0x76f988daul,
    0x983e5152ul,0xa831c66dul,0xb00327c8ul,0xbf597fc7ul,0xc6e00bf3ul,0xd5a79147ul,0x06ca6351ul,0x14292967ul,
    0x27b70a85ul,0x2e1b2138ul,0x4d2c6dfcul,0x53380d13ul,0x650a7354ul,0x766a0abbul,0x81c2c92eul,0x92722c85ul,
    0xa2bfe8a1ul,0xa81a664bul,0xc24b8b70ul,0xc76c51a3ul,0xd192e819ul,0xd6990624ul,0xf40e3585ul,0x106aa070ul,
    0x19a4c116ul,0x1e376c08ul,0x2748774cul,0x34b0bcb5ul,0x391c0cb3ul,0x4ed8aa4aul,0x5b9cca4ful,0x682e6ff3ul,
    0x748f82eeul,0x78a5636ful,0x84c87814ul,0x8cc70208ul,0x90befffaul,0xa4506cebul,0xbef9a3f7ul,0xc67178f2ul
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ void sha256(const uint8_t* msg, int len, uint8_t* hash) {
    uint32_t h[8] = {
        0x6a09e667ul, 0xbb67ae85ul, 0x3c6ef372ul, 0xa54ff53aul,
        0x510e527ful, 0x9b05688cul, 0x1f83d9abul, 0x5be0cd19ul
    };

    uint8_t data[64] = {0};
    memcpy(data, msg, len);
    data[len] = 0x80;
    uint64_t bitlen = len * 8;
    data[63] = bitlen & 0xFF;
    data[62] = (bitlen >> 8) & 0xFF;

    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = (data[i * 4] << 24) |
               (data[i * 4 + 1] << 16) |
               (data[i * 4 + 2] << 8) |
               (data[i * 4 + 3]);
    }

    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], h7 = h[7];

    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h7 + S1 + ch + k[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h7 = g; g = f; f = e;
        e = d + temp1;
        d = c; c = b; b = a;
        a = temp1 + temp2;
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += h7;

    for (int i = 0; i < 8; i++) {
        hash[i * 4]     = (h[i] >> 24) & 0xFF;
        hash[i * 4 + 1] = (h[i] >> 16) & 0xFF;
        hash[i * 4 + 2] = (h[i] >> 8) & 0xFF;
        hash[i * 4 + 3] = h[i] & 0xFF;
    }
}

// __device__ void indexToPassword(unsigned long long index, int length, char* out) {
//     for (int i = 0; i < length; ++i) {
//         out[length - 1 - i] = d_charset[index % d_charsetLen];
//         index /= d_charsetLen;
//     }
//     out[length] = '\0';
// }

__global__ void crackSha256Kernel(int length, unsigned long long totalCombos) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;

    char candidate[8];
    unsigned char hash[32];

    for (unsigned long long i = tid; i < totalCombos && !d_found; i += stride) {
        indexToPassword(i, length, candidate);
        sha256((unsigned char*)candidate, length, hash);

        bool match = true;
        for (int j = 0; j < 32; ++j) {
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

extern "C" void launchCrackerSHA(const unsigned char* targetHash, const char* charset, int charsetLen, int length, char* hostResult) {
    unsigned long long total = 1;
    for (int i = 0; i < length; ++i)
        total *= charsetLen;

    cudaMemcpyToSymbol(d_charset, charset, charsetLen);
    cudaMemcpyToSymbol(d_charsetLen, &charsetLen, sizeof(int));
    cudaMemcpyToSymbol(d_targetHash, targetHash, 32);
    bool found = false;
    cudaMemcpyToSymbol(d_found, &found, sizeof(bool));

    crackSha256Kernel<<<32, 256>>>(length, total);
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(hostResult, d_result, 8);
}
