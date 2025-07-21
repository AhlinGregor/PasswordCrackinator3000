#pragma once

__device__ __inline__ void indexToPassword(unsigned long long index, int length, char* out) {
    for (int i = 0; i < length; ++i) {
        out[length - 1 - i] = d_charset[index % d_charsetLen];
        index /= d_charsetLen;
    }
    out[length] = '\0';
}
