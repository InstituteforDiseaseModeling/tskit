#include "sha256.h"
#include <string.h>     // memset and memcpy
#include <stdio.h>      // sprintf

#define SHA2_SHFR(x, n)    (x >> n)
#define SHA2_ROTR(x, n)   ((x >> n) | (x << ((sizeof(x) << 3) - n)))
#define SHA2_ROTL(x, n)   ((x << n) | (x >> ((sizeof(x) << 3) - n)))
#define SHA2_CH(x, y, z)  ((x & y) ^ (~x & z))
#define SHA2_MAJ(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define SHA256_F1(x) (SHA2_ROTR(x,  2) ^ SHA2_ROTR(x, 13) ^ SHA2_ROTR(x, 22))
#define SHA256_F2(x) (SHA2_ROTR(x,  6) ^ SHA2_ROTR(x, 11) ^ SHA2_ROTR(x, 25))
#define SHA256_F3(x) (SHA2_ROTR(x,  7) ^ SHA2_ROTR(x, 18) ^ SHA2_SHFR(x,  3))
#define SHA256_F4(x) (SHA2_ROTR(x, 17) ^ SHA2_ROTR(x, 19) ^ SHA2_SHFR(x, 10))
#define SHA2_UNPACK32(x, str)                 \
{                                             \
    *((str) + 3) = (uint8_t) ((x)      );     \
    *((str) + 2) = (uint8_t) ((x) >>  8);     \
    *((str) + 1) = (uint8_t) ((x) >> 16);     \
    *((str) + 0) = (uint8_t) ((x) >> 24);     \
}
#define SHA2_PACK32(str, x)                   \
{                                             \
    *(x) =   ((uint32_t) *((str) + 3)      )  \
           | ((uint32_t) *((str) + 2) <<  8)  \
           | ((uint32_t) *((str) + 1) << 16)  \
           | ((uint32_t) *((str) + 0) << 24); \
}

const uint32_t sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

void transform(const uint8_t *message, uint32_t block_nb, SHA256 *psha)
{
    uint32_t w[64];
    uint32_t wv[8];
    uint32_t t1, t2;
    const uint8_t *sub_block;
    int i;
    int j;
    for (i = 0; i < (int) block_nb; i++) {
        sub_block = message + (i << 6);
        for (j = 0; j < 16; j++) {
            SHA2_PACK32(&sub_block[j << 2], &w[j]);
        }
        for (j = 16; j < 64; j++) {
            w[j] =  SHA256_F4(w[j -  2]) + w[j -  7] + SHA256_F3(w[j - 15]) + w[j - 16];
        }
        for (j = 0; j < 8; j++) {
            wv[j] = psha->m_h[j];
        }
        for (j = 0; j < 64; j++) {
            t1 = wv[7] + SHA256_F2(wv[4]) + SHA2_CH(wv[4], wv[5], wv[6])
                + sha256_k[j] + w[j];
            t2 = SHA256_F1(wv[0]) + SHA2_MAJ(wv[0], wv[1], wv[2]);
            wv[7] = wv[6];
            wv[6] = wv[5];
            wv[5] = wv[4];
            wv[4] = wv[3] + t1;
            wv[3] = wv[2];
            wv[2] = wv[1];
            wv[1] = wv[0];
            wv[0] = t1 + t2;
        }
        for (j = 0; j < 8; j++) {
            psha->m_h[j] += wv[j];
        }
    }

    return;
}

void init(SHA256 *psha)
{
    psha->m_h[0] = 0x6a09e667;
    psha->m_h[1] = 0xbb67ae85;
    psha->m_h[2] = 0x3c6ef372;
    psha->m_h[3] = 0xa54ff53a;
    psha->m_h[4] = 0x510e527f;
    psha->m_h[5] = 0x9b05688c;
    psha->m_h[6] = 0x1f83d9ab;
    psha->m_h[7] = 0x5be0cd19;
    psha->m_len = 0;
    psha->m_tot_len = 0;

    return;
}

void update(const uint8_t *message, uint32_t len, SHA256 *psha)
{
    uint32_t block_nb;
    uint32_t new_len, rem_len, tmp_len;
    const uint8_t *shifted_message;
    tmp_len = SHA224_256_BLOCK_SIZE - psha->m_len;
    rem_len = len < tmp_len ? len : tmp_len;
    memcpy(&psha->m_block[psha->m_len], message, rem_len);
    if (psha->m_len + len < SHA224_256_BLOCK_SIZE) {
        psha->m_len += len;
        return;
    }
    new_len = len - rem_len;
    block_nb = new_len / SHA224_256_BLOCK_SIZE;
    shifted_message = message + rem_len;
    transform(psha->m_block, 1, psha);
    transform(shifted_message, block_nb, psha);
    rem_len = new_len % SHA224_256_BLOCK_SIZE;
    memcpy(psha->m_block, &shifted_message[block_nb << 6], rem_len);
    psha->m_len = rem_len;
    psha->m_tot_len += (block_nb + 1) << 6;

    return;
}

void final(uint8_t *digest, SHA256 *psha)
{
    uint32_t block_nb;
    uint32_t pm_len;
    uint32_t len_b;
    int i;
    block_nb = (1 + ((SHA224_256_BLOCK_SIZE - 9)
                     < (psha->m_len % SHA224_256_BLOCK_SIZE)));
    len_b = (psha->m_tot_len + psha->m_len) << 3;
    pm_len = block_nb << 6;
    memset(psha->m_block + psha->m_len, 0, pm_len - psha->m_len);
    psha->m_block[psha->m_len] = 0x80;
    SHA2_UNPACK32(len_b, psha->m_block + pm_len - 4);
    transform(psha->m_block, block_nb, psha);
    for (i = 0 ; i < 8; i++) {
        SHA2_UNPACK32(psha->m_h[i], &digest[i << 2]);
    }
}

// digest should point to SHA256_DIGEST_SIZE bytes/uint8_t
// string should point to 2 * SHA256_DIGEST_SIZE + 1 chars
void sha256(const uint8_t* pdata, size_t num_bytes, uint8_t *digest, char *string)
{
    memset(digest, 0, SHA256_DIGEST_SIZE);

    SHA256 sha;
    init(&sha);

    update(pdata, num_bytes, &sha);
    final(digest, &sha);

    string[2*SHA256_DIGEST_SIZE] = 0;
    for (int i = 0; i < SHA256_DIGEST_SIZE; i++)
        sprintf(string+i*2, "%02x", digest[i]);

    return;
}
