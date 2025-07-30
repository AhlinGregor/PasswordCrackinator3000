#include <jni.h>
#include <cstring>
#include <algorithm>
#include "config.h"
#include "org_example_CUDASolution.h"
#include <cuda_runtime.h>



extern "C" {
void mcm_cuda_sha256_hash_batch(BYTE* in, WORD inlen, BYTE* out, WORD n_batch);
void mcm_cuda_md5_hash_batch(BYTE* in, WORD inlen, BYTE* out, WORD n_batch);
}

JNIEXPORT jstring JNICALL Java_org_example_CUDASolution_nativeBruteForce(
    JNIEnv* env, jclass, jstring jCharset, jint length,
    jbyteArray jTargetHash, jint hashMode)
{
    const char* charset = env->GetStringUTFChars(jCharset, nullptr);                            // Convert the charset from java string to C string
    int charsetLen = strlen(charset);                                                           // Gets the length of the charset
    jbyte* targetHash = env->GetByteArrayElements(jTargetHash, nullptr);                        // Convert java byte array into a native C pointer

    uint64_t total = 1;                                                                         // Calculate total permutations
    for (int i = 0; i < length; ++i) total *= charsetLen;

    const uint32_t batchSize = 100000;                                                          // Number of candidates to process at once
    char* h_input = new char[batchSize * length];                                               // CPU input buffer
    BYTE* h_output = new BYTE[batchSize * (hashMode ? SHA256_BLOCK_SIZE : MD5_BLOCK_SIZE)];     // CPU output buffer
    BYTE *d_input, *d_output;

    cudaMalloc(&d_input, batchSize * length);                                                   // GPU input
    cudaMalloc(&d_output, batchSize * (hashMode ? SHA256_BLOCK_SIZE : MD5_BLOCK_SIZE));         // GPU output

    bool found = false;
    char result[33] = {0};

    for (uint64_t start = 0; start < total && !found; start += batchSize) {                     // Loop over all combinations in chunks
        int count = std::min<uint64_t>(batchSize, total - start);                               // Either batchSize or the remainder if at the end of

        for (int i = 0; i < count; ++i) {                                                       // Convert index to password permutation
            uint64_t idx = start + i;
            uint64_t tmp = idx;
            for (int j = 0; j < length; ++j) {
                h_input[i * length + (length - 1 - j)] = charset[tmp % charsetLen];
                tmp /= charsetLen;
            }
        }

        cudaMemcpy(d_input, h_input, count * length, cudaMemcpyHostToDevice);                   // Copy batch of permutations to GPU

        if (hashMode == 1)                                                                      // Run the kernels
            mcm_cuda_sha256_hash_batch((BYTE*)d_input, length, d_output, count);
        else
            mcm_cuda_md5_hash_batch((BYTE*)d_input, length, d_output, count);

        cudaMemcpy(h_output, d_output, count * (hashMode ? SHA256_BLOCK_SIZE : MD5_BLOCK_SIZE),
                   cudaMemcpyDeviceToHost);                                                     // Copy results back to CPU

        for (int i = 0; i < count && !found; ++i) {                                             // Compare GPU hashes to target
            BYTE* h = h_output + i * (hashMode ? SHA256_BLOCK_SIZE : MD5_BLOCK_SIZE);
            bool match = true;
            for (int b = 0; b < (hashMode ? SHA256_BLOCK_SIZE : MD5_BLOCK_SIZE); ++b) {
                if (h[b] != (BYTE)targetHash[b]) { match = false; break; }
            }
            if (match) {
                memcpy(result, h_input + i * length, length);
                found = true;
            }
        }
    }
                                                                                                // Free memory and release references
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    env->ReleaseStringUTFChars(jCharset, charset);
    env->ReleaseByteArrayElements(jTargetHash, targetHash, 0);

    return found ? env->NewStringUTF(result) : nullptr;                                         // Return matching password as a Java string or null
}
