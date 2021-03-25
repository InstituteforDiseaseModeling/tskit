#pragma once

#include <stddef.h>     // size_t
#include <stdint.h>     // uintN_t types

#define SHA224_256_BLOCK_SIZE   (512 / 8)   // 512 bits, in bytes == 64
#define SHA256_DIGEST_SIZE      (256 / 8)   // 256 bits, in bytes == 32

typedef struct _SHA256_
{
    uint32_t m_tot_len;
    uint32_t m_len;
    uint8_t m_block[2 * SHA224_256_BLOCK_SIZE];
    uint32_t m_h[8];
} SHA256;

// digest should point to SHA256_DIGEST_SIZE bytes/uint8_t
// string should point to 2 * SHA256_DIGEST_SIZE + 1 chars
void sha256(const uint8_t* pdata, size_t num_bytes, uint8_t *digest, char *string);
