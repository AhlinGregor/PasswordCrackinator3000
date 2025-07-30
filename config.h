/*
 * Type Definitions for CUDA Hashing Algos
 *
 * Date: 12 June 2019
 * Revision: 1
 *
 * This file is released into the Public Domain.
 */

#pragma once
#define USE_MD2 1
#define USE_MD5 1
#define USE_SHA1 1
#define USE_SHA256 1

#define CUDA_HASH 1
#define OCL_HASH 0

#define SHA256_BLOCK_SIZE 32 // SHA-256 hash is 32 bytes (256 bits)
#define MD5_BLOCK_SIZE    16 // MD5 hash is 16 bytes (128 bits)

typedef unsigned char BYTE;
typedef unsigned int  WORD;
typedef unsigned long long LONG;

#include <stdlib.h>
#include <string.h>
#include <stdio.h>