//***************************************************************************/
// This software is released under the 2-Clause BSD license, included
// below.
//
// Copyright (c) 2022, Aous Naman
// Copyright (c) 2022, Kakadu Software Pty Ltd, Australia
// Copyright (c) 2022, The University of New South Wales, Australia
// Copyright (c) 2024, Intel Corporation
// Copyright (c) 2026, Osamu Watanabe
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//***************************************************************************/
// This file is part of the OpenJpeg software implementation.
// File: ht_dec_avx2.c
// A C port of OpenJPH 0.32's ojph_block_decoder_avx2.cpp and of the UVLC
// table construction of its ojph_block_common.cpp.
//***************************************************************************/

//***************************************************************************/
/** @file ht_dec_avx2.c
 *  @brief implements an AVX2 HTJ2K block decoder
 *
 *  This is a C port of the AVX2 block decoder of OpenJPH 0.32.  It is
 *  compiled with AVX2 code generation enabled and must only be called
 *  after a run-time check that the CPU supports AVX2 (see ht_dec.c).
 *
 *  Differences from the OpenJPH original, all needed to reproduce the
 *  scalar decoder of ht_dec.c bit for bit, including on malformed input:
 *  - the decoder rejects U_q > zero_bplanes + 1 (the scalar decoder's
 *    bound) instead of U_q > zero_bplanes + 2;
 *  - VLC codewords that mark samples outside the code-block significant
 *    are rejected, as the scalar decoder does;
 *  - the 32-bit path computes samples at the scalar decoder's bitplane
 *    (numbps); the 16-bit path, used when zero_bplanes <= 12 and
 *    Mb <= 28, computes them at bitplane 29 - zero_bplanes, which leaves
 *    one bit of headroom, and the result is shifted down by 28 - Mb;
 *  - all buffers live in a heap workspace; samples are decoded into a
 *    padded, aligned buffer and then converted to two's complement in
 *    the Tier-1 buffer, whose stride is the code-block width.
 */

#define OPJ_SKIP_POISON
#include "opj_includes.h"

#include <immintrin.h>

#if defined(__GNUC__)
#pragma GCC poison malloc calloc realloc free
#endif

#include "ht_dec_simd.h"
#include "t1_ht_luts.h"

#if defined(_MSC_VER)
#define OPJ_HT_FORCE_INLINE static __forceinline
#define OPJ_HT_NO_INLINE static __declspec(noinline)
#elif defined(__GNUC__)
#define OPJ_HT_FORCE_INLINE static INLINE __attribute__((always_inline))
#define OPJ_HT_NO_INLINE static __attribute__((noinline))
#else
#define OPJ_HT_FORCE_INLINE static INLINE
#define OPJ_HT_NO_INLINE static
#endif

//************************************************************************/
/** @brief MEL state structure for reading and decoding the MEL bitstream
  *
  *  A number of events is decoded from the MEL bitstream ahead of time
  *  and stored in run/num_runs.
  *  Each run represents the number of zero events before a one event.
  */
typedef struct dec_mel {
    // data decoding machinery
    const OPJ_UINT8* data; //!<the address of data (or bitstream)
    OPJ_UINT64 tmp;        //!<temporary buffer for read data
    int bits;              //!<number of bits stored in tmp
    int size;              //!<number of bytes in MEL code
    OPJ_BOOL unstuff;      //!<true if the next bit needs to be unstuffed
    int k;                 //!<state of MEL decoder

    // queue of decoded runs
    int num_runs;    //!<number of decoded runs left in runs (maximum 8)
    OPJ_UINT64 runs; //!<runs of decoded MEL codewords (7 bits/run)
} dec_mel_t;

//************************************************************************/
/** @brief Reads and unstuffs the MEL bitstream; identical to the scalar
  *        decoder's mel_read
  *
  *  @param [in]  melp is a pointer to dec_mel_t structure
  */
static INLINE
void mel_read(dec_mel_t *melp)
{
    OPJ_UINT32 val;
    int bits;
    OPJ_UINT32 t;
    OPJ_BOOL unstuff;

    if (melp->bits > 32) { //there are enough bits in the tmp variable
        return;    // return without reading new data
    }

    val = 0xFFFFFFFF;      // feed in 0xFF if buffer is exhausted
    if (melp->size > 4) {  // if there is more than 4 bytes the MEL segment
        memcpy(&val, melp->data, 4);  // read 32 bits from MEL data
        melp->data += 4;           // advance pointer
        melp->size -= 4;           // reduce counter
    } else if (melp->size > 0) { // 4 or less
        OPJ_UINT32 m, v;
        int i = 0;
        while (melp->size > 1) {
            OPJ_UINT32 v = *melp->data++; // read one byte at a time
            OPJ_UINT32 m = ~(0xFFu << i); // mask of location
            val = (val & m) | (v << i);   // put byte in its correct location
            --melp->size;
            i += 8;
        }
        // size equal to 1
        v = *melp->data++;  // the one before the last is different
        v |= 0xF;                         // MEL and VLC segments can overlap
        m = ~(0xFFu << i);
        val = (val & m) | (v << i);
        --melp->size;
    }

    // next we unstuff them before adding them to the buffer
    bits = 32 - melp->unstuff;      // number of bits in val, subtract 1 if
    // the previously read byte requires
    // unstuffing

    // data is unstuffed and accumulated in t
    // bits has the number of bits in t
    t = val & 0xFF;
    unstuff = ((val & 0xFF) == 0xFF); // true if the byte needs unstuffing
    bits -= unstuff; // there is one less bit in t if unstuffing is needed
    t = t << (8 - unstuff); // move up to make room for the next byte

    //this is a repeat of the above
    t |= (val >> 8) & 0xFF;
    unstuff = (((val >> 8) & 0xFF) == 0xFF);
    bits -= unstuff;
    t = t << (8 - unstuff);

    t |= (val >> 16) & 0xFF;
    unstuff = (((val >> 16) & 0xFF) == 0xFF);
    bits -= unstuff;
    t = t << (8 - unstuff);

    t |= (val >> 24) & 0xFF;
    melp->unstuff = (((val >> 24) & 0xFF) == 0xFF);

    // move t to tmp, and push the result all the way up, so we read from
    // the MSB
    melp->tmp |= ((OPJ_UINT64)t) << (64 - bits - melp->bits);
    melp->bits += bits; //increment the number of bits in tmp
}

//************************************************************************/
/** @brief Decodes unstuffed MEL segment bits stored in tmp to runs; see
  *        the scalar decoder's mel_decode
  *
  *  @param [in]  melp is a pointer to dec_mel_t structure
  */
static INLINE
void mel_decode(dec_mel_t *melp)
{
    static const int mel_exp[13] = { //MEL exponents
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5
    };

    if (melp->bits < 6) { // if there are less than 6 bits in tmp
        mel_read(melp);    // then read from the MEL bitstream
    }
    // 6 bits is the largest decodable MEL cwd

    //repeat so long that there is enough decodable bits in tmp,
    // and the runs store is not full (num_runs < 8)
    while (melp->bits >= 6 && melp->num_runs < 8) {
        int eval = mel_exp[melp->k]; // number of bits associated with state
        int run = 0;
        if (melp->tmp & (1ull << 63)) { //The next bit to decode (stored in MSB)
            //one is found
            run = 1 << eval;
            run--; // consecutive runs of 0 events - 1
            melp->k = melp->k + 1 < 12 ? melp->k + 1 : 12;//increment, max is 12
            melp->tmp <<= 1; // consume one bit from tmp
            melp->bits -= 1;
            run = run << 1; // a stretch of zeros not terminating in one
        } else {
            //0 is found
            run = (int)(melp->tmp >> (63 - eval)) & ((1 << eval) - 1);
            melp->k = melp->k - 1 > 0 ? melp->k - 1 : 0; //decrement, min is 0
            melp->tmp <<= eval + 1; //consume eval + 1 bits (max is 6)
            melp->bits -= eval + 1;
            run = (run << 1) + 1; // a stretch of zeros terminating with one
        }
        eval = melp->num_runs * 7;                 // 7 bits per run
        melp->runs &= ~((OPJ_UINT64)0x3F << eval); // 6 bits are sufficient
        melp->runs |= ((OPJ_UINT64)run) << eval;   // store the value in runs
        melp->num_runs++;                          // increment count
    }
}

//************************************************************************/
/** @brief Initiates a dec_mel_t structure for MEL decoding and reads
  *         some bytes in order to get the read address to a multiple
  *         of 4; identical to the scalar decoder's mel_init, including
  *         the rejection of a stuffing violation
  *
  *  @param [in]  melp is a pointer to dec_mel_t structure
  *  @param [in]  bbuf is a pointer to byte buffer
  *  @param [in]  lcup is the length of MagSgn+MEL+VLC segments
  *  @param [in]  scup is the length of MEL+VLC segments
  */
static INLINE
OPJ_BOOL mel_init(dec_mel_t *melp, const OPJ_UINT8* bbuf, int lcup, int scup)
{
    int num;
    int i;

    melp->data = bbuf + lcup - scup; // move the pointer to the start of MEL
    melp->bits = 0;                  // 0 bits in tmp
    melp->tmp = 0;                   //
    melp->unstuff = OPJ_FALSE;       // no unstuffing
    melp->size = scup - 1;           // size is the length of MEL+VLC-1
    melp->k = 0;                     // 0 for state
    melp->num_runs = 0;              // num_runs is 0
    melp->runs = 0;                  //

    //These few lines take care of the case where data is not at a multiple
    // of 4 boundary.  It reads 1,2,3 up to 4 bytes from the MEL segment
    num = 4 - (int)((intptr_t)(melp->data) & 0x3);
    for (i = 0; i < num; ++i) { // this code is similar to mel_read
        OPJ_UINT64 d;
        int d_bits;

        if (melp->unstuff == OPJ_TRUE && melp->data[0] > 0x8F) {
            return OPJ_FALSE;
        }
        d = (melp->size > 0) ? *melp->data : 0xFF; // if buffer is consumed
        // set data to 0xFF
        if (melp->size == 1) {
            d |= 0xF;    //if this is MEL+VLC-1, set LSBs to 0xF
        }
        // see the standard
        melp->data += melp->size-- > 0; //increment if the end is not reached
        d_bits = 8 - melp->unstuff; //if unstuffing is needed, reduce by 1
        melp->tmp = (melp->tmp << d_bits) | d; //store bits in tmp
        melp->bits += d_bits;  //increment tmp by number of bits
        melp->unstuff = ((d & 0xFF) == 0xFF); //true of next byte needs
        //unstuffing
    }
    melp->tmp <<= (64 - melp->bits); //push all the way up so the first bit
    // is the MSB
    return OPJ_TRUE;
}

//************************************************************************/
/** @brief Retrieves one run from dec_mel_t; if there are no runs stored
  *         MEL segment is decoded
  *
  * @param [in]  melp is a pointer to dec_mel_t structure
  */
static INLINE
int mel_get_run(dec_mel_t *melp)
{
    int t;
    if (melp->num_runs == 0) { //if no runs, decode more bit from MEL segment
        mel_decode(melp);
    }

    t = (int)(melp->runs & 0x7F); //retrieve one run
    melp->runs >>= 7;  // remove the retrieved run
    melp->num_runs--;
    return t; // return run
}

//************************************************************************/
/** @brief De-stuffs a forward-growing bitstream into a flat buffer
  *
  *  This replicates, in one pass, the bit production of the frwd_struct
  *  bitstream reader of the scalar decoder: a byte following a 0xFF byte
  *  contributes only 7 bits (its MSB -- guaranteed 0 by the encoder --
  *  is OR'ed with the following bit), and bits beyond the end of the
  *  data read as X (1s for MagSgn, 0s for SPP).  With the stream
  *  flattened, bits for any quad can be fetched at an absolute bit
  *  position (see dfetch), removing the serial fetch/advance state from
  *  the decode loops.
  *
  *  Writes at most cap + 1 destuffed bytes followed by 64 bytes of
  *  padding; dst must have room for cap + 65 bytes.  cap must be
  *  no smaller than the maximum number of bits the decoder can consume
  *  divided by 8, so clipping the data at cap loses no decodable bits.
  *
  *  @param [in]  src is a pointer to the start of the bitstream data
  *  @param [in]  size is the number of bytes in the bitstream data
  *  @param [in]  dst is the output buffer, of at least cap + 65 bytes
  *  @param [in]  cap is the maximum number of destuffed bytes to write
  *  @param [in]  X is the value fed in when the bitstream is exhausted,
  *               0xFF or 0
  *  @return clamp offset for dfetch; bytes at or beyond this offset hold
  *          no stream bits (they read as X)
  */
static OPJ_UINT32 destuff_frwd(const OPJ_UINT8* src, int size,
                               OPJ_UINT8* dst, OPJ_UINT32 cap, OPJ_UINT32 X)
{
    OPJ_UINT8* o = dst;
    OPJ_UINT8* o_end = dst + cap;
    const OPJ_UINT8* s = src;
    const OPJ_UINT8* s_end;
    OPJ_UINT64 acc = 0;       // partial output byte, low nb bits are valid
    OPJ_UINT32 nb = 0;        // number of valid bits in acc; always < 8
    OPJ_BOOL prev_ff = OPJ_FALSE;
    OPJ_UINT32 fill;
    __m256i pad;

    if (size < 0) {
        size = 0;
    }
    s_end = src + size;

    // fast path; 16 source bytes at a time when they contain no 0xFF
    while (s + 16 <= s_end && o + 24 <= o_end) {
        __m128i v = _mm_loadu_si128((const __m128i*)(const void*)s);
        int ff = _mm_movemask_epi8(_mm_cmpeq_epi8(v, _mm_set1_epi8(-1)));
        OPJ_UINT64 v0, v1, w0, w1;
        if (ff != 0 || prev_ff) {
            // process these 16 bytes one at a time
            int i;
            for (i = 0; i < 16; ++i) {
                OPJ_UINT8 b = *s++;
                acc |= (OPJ_UINT64)b << nb;
                nb += prev_ff ? 7u : 8u;
                prev_ff = (b == 0xFFu);
                if (nb >= 8) {
                    *o++ = (OPJ_UINT8)acc;
                    acc >>= 8;
                    nb -= 8;
                }
            }
            continue;
        }
        memcpy(&v0, s, 8);
        memcpy(&v1, s + 8, 8);
        w0 = acc | (v0 << nb);
        w1 = (v1 << nb) | (nb ? (v0 >> (64 - nb)) : 0);
        memcpy(o, &w0, 8);
        memcpy(o + 8, &w1, 8);
        acc = nb ? (v1 >> (64 - nb)) : 0;
        o += 16;
        s += 16;
    }
    // tail; one byte at a time
    while (s < s_end && o < o_end) {
        OPJ_UINT8 b = *s++;
        acc |= (OPJ_UINT64)b << nb;
        nb += prev_ff ? 7u : 8u;
        prev_ff = (b == 0xFFu);
        if (nb >= 8) {
            *o++ = (OPJ_UINT8)acc;
            acc >>= 8;
            nb -= 8;
        }
    }
    // fill the bits above nb with X, and pad with X bytes
    fill = (X == 0xFF) ? (0xFFu << nb) : 0;
    *o = (OPJ_UINT8)((OPJ_UINT32)acc | fill);
    pad = _mm256_set1_epi8((char)X);
    _mm256_storeu_si256((__m256i*)(void*)(o + 1), pad);
    _mm256_storeu_si256((__m256i*)(void*)(o + 33), pad);
    return (OPJ_UINT32)(o - dst) + 1;
}

//************************************************************************/
/** @brief Fetches 128 bits from a destuffed MagSgn buffer
  *
  *  Returns the 128 bits starting at bit position pos of the buffer
  *  produced by destuff_frwd.  The byte offset is clamped to limit so
  *  that positions past the end of the stream read as X without leaving
  *  the buffer.
  *
  *  @param [in]  dbuf is the destuffed MagSgn buffer
  *  @param [in]  limit is the clamp offset returned by destuff_frwd
  *  @param [in]  pos is the absolute bit position to fetch from
  */
OPJ_HT_FORCE_INLINE
__m128i dfetch(const OPJ_UINT8* dbuf, OPJ_UINT32 limit, OPJ_UINT32 pos)
{
    OPJ_UINT32 off = pos >> 3;
    const OPJ_UINT8* p;
    __m128i v, w, r, c;
    int k;

    off = off < limit ? off : limit;
    p = dbuf + off;
    v = _mm_loadu_si128((const __m128i*)(const void*)p);
    w = _mm_loadu_si128((const __m128i*)(const void*)(p + 8));
    k = (int)(pos & 7);
    r = _mm_srl_epi64(v, _mm_cvtsi32_si128(k));
    c = _mm_sll_epi64(w, _mm_cvtsi32_si128(64 - k));
    return _mm_or_si128(r, c);
}

//************************************************************************/
/** @brief Fetches at least 57 bits from a destuffed bitstream
  *
  *  Scalar counterpart of dfetch; returns the bits starting at bit
  *  position pos of a buffer produced by destuff_frwd or destuff_rev, in
  *  the lower bits of the result.  Bits above 64 - (pos & 7) are garbage.
  *
  *  @param [in]  dbuf is the destuffed bitstream buffer
  *  @param [in]  limit is the clamp offset returned by the destuffer
  *  @param [in]  pos is the absolute bit position to fetch from
  */
OPJ_HT_FORCE_INLINE
OPJ_UINT64 dfetch64(const OPJ_UINT8* dbuf, OPJ_UINT32 limit, OPJ_UINT32 pos)
{
    OPJ_UINT32 off = pos >> 3;
    OPJ_UINT64 v;
    off = off < limit ? off : limit;
    memcpy(&v, dbuf + off, 8);
    return v >> (pos & 7);
}

//************************************************************************/
/** @brief Branchless refill of a register-resident bit window
  *
  *  Loads 8 bytes from a destuffed buffer and inserts whole bytes above
  *  the bits remaining in the window, leaving 56 to 63 valid bits.  The
  *  read offset is clamped to limit so that, when consumption overruns
  *  the stream (truncated code-blocks), reads come from the zero padding
  *  -- the bytes there are exactly the fill the stream would produce.
  *
  *  @param [in,out]  val is the bit window; its low bits are valid
  *  @param [in,out]  bits is the number of valid bits in val
  *  @param [in,out]  off is the read offset in the destuffed buffer
  *  @param [in]      dbuf is the destuffed bitstream buffer
  *  @param [in]      limit is the clamp offset returned by destuff_vlc
  */
OPJ_HT_FORCE_INLINE
void drefill(OPJ_UINT64* val, OPJ_UINT32* bits, OPJ_UINT32* off,
             const OPJ_UINT8* dbuf, OPJ_UINT32 limit)
{
    OPJ_UINT64 v;
    OPJ_UINT32 o = *off < limit ? *off : limit;
    memcpy(&v, dbuf + o, 8);
    *val |= v << *bits;
    *off += (63 - *bits) >> 3;
    *bits |= 56;
}

//************************************************************************/
/** @brief Consumes bits from a window refilled by drefill
  *
  *  @param [in,out]  val is the bit window
  *  @param [in,out]  bits is the number of valid bits in val
  *  @param [in]      num_bits is the number of bits to consume
  */
OPJ_HT_FORCE_INLINE
void dconsume(OPJ_UINT64* val, OPJ_UINT32* bits, OPJ_UINT32 num_bits)
{
    *val >>= num_bits;
    *bits -= num_bits;
}

//************************************************************************/
/** @brief De-stuffs a backward-growing bitstream into a flat buffer
  *
  *  This replicates the bit production of the rev_struct bitstream
  *  reader of the scalar decoder, for both its VLC and MRP variants:
  *  reading backward from p, a byte contributes only 7 bits (its MSB --
  *  guaranteed 0 by the encoder -- is OR'ed with the following bit) when
  *  the byte after it in memory is greater than 0x8F and its own low 7
  *  bits are all 1s.  Bits beyond the end of the data read as 0s,
  *  matching both readers.  The first byte is processed with the
  *  caller-provided unstuff state; after that the state always equals
  *  (p[1] > 0x8F), so the fast path can detect stuffing with a pairwise
  *  byte comparison within the segment.
  *
  *  Writes at most cap + 1 destuffed bytes followed by 64 bytes of
  *  zero padding; dst must have room for cap + 65 bytes.
  *
  *  @param [in]  p is a pointer to the last byte of the segment, where
  *               backward reading starts
  *  @param [in]  size is the number of bytes in the segment
  *  @param [in]  unstuff is the initial unstuffing state
  *  @param [in]  acc holds bits produced by initialization, low nb bits
  *  @param [in]  nb is the number of valid bits in acc; less than 8
  *  @param [in]  dst is the output buffer, of at least cap + 65 bytes
  *  @param [in]  cap is the maximum number of destuffed bytes to write
  *  @return clamp offset for dfetch64/drefill; bytes at or beyond this
  *          offset hold no stream bits (they read as 0s)
  */
static OPJ_UINT32 destuff_rev(const OPJ_UINT8* p, int size, OPJ_BOOL unstuff,
                              OPJ_UINT64 acc, OPJ_UINT32 nb,
                              OPJ_UINT8* dst, OPJ_UINT32 cap)
{
    OPJ_UINT8* o = dst;
    OPJ_UINT8* o_end = dst + cap;
    __m256i z;

    // process the first byte with the caller-provided unstuff state;
    // afterwards the state is implied by the byte at p[1], which the
    // fast path checks vectorially (and which stays inside the segment)
    if (size > 0 && o < o_end) {
        OPJ_UINT32 d = *p--;
        --size;
        acc |= (OPJ_UINT64)d << nb;
        nb += 8 - ((unstuff && ((d & 0x7F) == 0x7F)) ? 1u : 0u);
        unstuff = d > 0x8F;
        if (nb >= 8) {
            *o++ = (OPJ_UINT8)acc;
            acc >>= 8;
            nb -= 8;
        }
    }

    // fast path; 16 source bytes at a time when none needs unstuffing
    while (size >= 16 && o + 24 <= o_end) {
        __m128i v = _mm_loadu_si128((const __m128i*)(const void*)(p - 15));
        __m128i nx = _mm_loadu_si128((const __m128i*)(const void*)(p - 14));
        __m128i is7f, le8f, stuff, r;
        OPJ_UINT64 v0, v1, w0, w1;

        is7f = _mm_cmpeq_epi8(_mm_and_si128(v, _mm_set1_epi8(0x7F)),
                              _mm_set1_epi8(0x7F));
        // le8f is 0xFF where the byte after (in memory) is <= 0x8F
        le8f = _mm_cmpeq_epi8(_mm_subs_epu8(nx, _mm_set1_epi8((char)0x8F)),
                              _mm_setzero_si128());
        stuff = _mm_andnot_si128(le8f, is7f);
        if (!_mm_testz_si128(stuff, stuff)) {
            // process these 16 bytes one at a time
            int i;
            for (i = 0; i < 16; ++i) {
                OPJ_UINT32 d = *p--;
                acc |= (OPJ_UINT64)d << nb;
                nb += 8 - ((unstuff && ((d & 0x7F) == 0x7F)) ? 1u : 0u);
                unstuff = d > 0x8F;
                if (nb >= 8) {
                    *o++ = (OPJ_UINT8)acc;
                    acc >>= 8;
                    nb -= 8;
                }
            }
            size -= 16;
            continue;
        }
        r = _mm_shuffle_epi8(v, _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7,
                                             8, 9, 10, 11, 12, 13, 14, 15));
#if defined(__x86_64__) || defined(_M_X64)
        v0 = (OPJ_UINT64)_mm_cvtsi128_si64(r);
        v1 = (OPJ_UINT64)_mm_extract_epi64(r, 1);
#else // 32-bit x86 lacks the 64-bit extract intrinsics
        v0 = (OPJ_UINT64)(OPJ_UINT32)_mm_cvtsi128_si32(r)
             | ((OPJ_UINT64)(OPJ_UINT32)_mm_extract_epi32(r, 1) << 32);
        v1 = (OPJ_UINT64)(OPJ_UINT32)_mm_extract_epi32(r, 2)
             | ((OPJ_UINT64)(OPJ_UINT32)_mm_extract_epi32(r, 3) << 32);
#endif
        w0 = acc | (v0 << nb);
        w1 = (v1 << nb) | (nb ? (v0 >> (64 - nb)) : 0);
        memcpy(o, &w0, 8);
        memcpy(o + 8, &w1, 8);
        acc = nb ? (v1 >> (64 - nb)) : 0;
        o += 16;
        p -= 16;
        size -= 16;
        unstuff = p[1] > 0x8F;
    }
    // tail; one byte at a time
    while (size > 0 && o < o_end) {
        OPJ_UINT32 d = *p--;
        --size;
        acc |= (OPJ_UINT64)d << nb;
        nb += 8 - ((unstuff && ((d & 0x7F) == 0x7F)) ? 1u : 0u);
        unstuff = d > 0x8F;
        if (nb >= 8) {
            *o++ = (OPJ_UINT8)acc;
            acc >>= 8;
            nb -= 8;
        }
    }
    // the bits above nb are already 0 = fill; pad with zero bytes
    *o = (OPJ_UINT8)acc;
    z = _mm256_setzero_si256();
    _mm256_storeu_si256((__m256i*)(void*)(o + 1), z);
    _mm256_storeu_si256((__m256i*)(void*)(o + 33), z);
    return (OPJ_UINT32)(o - dst) + 1;
}

//************************************************************************/
/** @brief De-stuffs the VLC segment into a flat buffer
  *
  *  Performs the same initialization the scalar rev_init does -- the
  *  first byte contributes only its upper 4 bits (3 bits if its lower 3
  *  bits are all 1s) -- then de-stuffs the remaining scup - 2 bytes
  *  backward; see destuff_rev.
  */
static OPJ_UINT32 destuff_vlc(const OPJ_UINT8* data, int lcup, int scup,
                              OPJ_UINT8* dst, OPJ_UINT32 cap)
{
    const OPJ_UINT8* p = data + lcup - 2;
    OPJ_UINT32 d = *p; // first byte, only the upper 4 bits are used
    OPJ_UINT64 acc = d >> 4;
    OPJ_UINT32 nb = 4 - ((acc & 7) == 7); // check standard
    OPJ_BOOL unstuff = (d | 0xF) > 0x8F;
    return destuff_rev(p - 1, scup - 2, unstuff, acc, nb, dst, cap);
}

//************************************************************************/
/** @brief De-stuffs the MRP segment into a flat buffer
  *
  *  Performs the same initialization the scalar rev_init_mrp does --
  *  reading starts at the last byte of the SPP+MRP segments with the
  *  unstuff state set -- then de-stuffs backward; see destuff_rev.
  */
static OPJ_UINT32 destuff_mrp(const OPJ_UINT8* data, int lcup, int len2,
                              OPJ_UINT8* dst, OPJ_UINT32 cap)
{
    return destuff_rev(data + lcup + len2 - 1, len2, OPJ_TRUE, 0, 0,
                       dst, cap);
}

//************************************************************************/
/** @brief decodes two consecutive quads (one octet), using 32 bit data
  *
  *  @param inf_u_q  decoded VLC code, with interleaved u values
  *  @param U_q      U values
  *  @param dbuf     de-stuffed MagSgn buffer
  *  @param limit    clamp offset of dbuf
  *  @param pos      bit position in dbuf, advanced by the bits consumed
  *  @param p        bitplane at which we are decoding
  *  @param vn       used for handling E values (stores v_n values)
  *  @return the decoded samples of the two quads
  */
OPJ_HT_FORCE_INLINE
__m256i decode_two_quad32_avx2(__m256i inf_u_q, __m256i U_q,
                               const OPJ_UINT8* dbuf, OPJ_UINT32 limit,
                               OPJ_UINT32* pos, OPJ_UINT32 p, __m128i* vn)
{
    __m256i row = _mm256_setzero_si256();

    // we keeps e_k, e_1, and rho in w2
    __m256i flags = _mm256_and_si256(inf_u_q,
                                     _mm256_set_epi32(0x8880, 0x4440, 0x2220, 0x1110,
                                             0x8880, 0x4440, 0x2220, 0x1110));
    __m256i insig = _mm256_cmpeq_epi32(flags, _mm256_setzero_si256());

    //are all insignificant?
    if ((OPJ_UINT32)_mm256_movemask_epi8(insig) != (OPJ_UINT32)0xFFFFFFFF) {
        __m256i m_n, w0, inc_sum, ex_sum, ms_vec, byte_idx, bit_idx;
        __m256i d0, d1, aa, bit_shift, shift, ones, twos, U_q_m1, tvn;
        __m128i ms_vec0, ms_vec1, a;
        OPJ_UINT32 total_mn1, total_mn2;

        flags = _mm256_mullo_epi16(flags, _mm256_set_epi16(1, 1, 2, 2, 4, 4, 8, 8,
                                   1, 1, 2, 2, 4, 4, 8, 8));

        // U_q holds U_q for this quad
        // flags has e_k, e_1, and rho such that e_k is sitting in the
        // 0x8000, e_1 in 0x800, and rho in 0x80

        // next e_k and m_n
        w0 = _mm256_srli_epi32(flags, 15); // e_k
        m_n = _mm256_sub_epi32(U_q, w0);
        m_n = _mm256_andnot_si256(insig, m_n);

        // find cumulative sums
        // to find at which bit in ms_vec the sample starts
        inc_sum = m_n; // inclusive scan
        inc_sum = _mm256_add_epi32(inc_sum, _mm256_bslli_epi128(inc_sum, 4));
        inc_sum = _mm256_add_epi32(inc_sum, _mm256_bslli_epi128(inc_sum, 8));
        total_mn1 = (OPJ_UINT32)_mm256_extract_epi16(inc_sum, 6);
        total_mn2 = (OPJ_UINT32)_mm256_extract_epi16(inc_sum, 14);

        ms_vec0 = dfetch(dbuf, limit, *pos);
        ms_vec1 = dfetch(dbuf, limit, *pos + total_mn1);
        *pos += total_mn1 + total_mn2;

        ms_vec = _mm256_inserti128_si256(_mm256_castsi128_si256(ms_vec0),
                                         ms_vec1, 0x1);

        ex_sum = _mm256_bslli_epi128(inc_sum, 4); // exclusive scan

        // find the starting byte and starting bit
        byte_idx = _mm256_srli_epi32(ex_sum, 3);
        bit_idx = _mm256_and_si256(ex_sum, _mm256_set1_epi32(7));
        byte_idx = _mm256_shuffle_epi8(byte_idx,
                                       _mm256_set_epi32(0x0C0C0C0C, 0x08080808, 0x04040404, 0x00000000,
                                               0x0C0C0C0C, 0x08080808, 0x04040404, 0x00000000));
        byte_idx = _mm256_add_epi32(byte_idx, _mm256_set1_epi32(0x03020100));
        d0 = _mm256_shuffle_epi8(ms_vec, byte_idx);
        byte_idx = _mm256_add_epi32(byte_idx, _mm256_set1_epi32(0x01010101));
        d1 = _mm256_shuffle_epi8(ms_vec, byte_idx);

        // shift samples values to correct location
        bit_idx = _mm256_or_si256(bit_idx, _mm256_slli_epi32(bit_idx, 16));

        a = _mm_set_epi8(1, 3, 7, 15, 31, 63, 127, -1,
                         1, 3, 7, 15, 31, 63, 127, -1);
        aa = _mm256_inserti128_si256(_mm256_castsi128_si256(a), a, 0x1);

        bit_shift = _mm256_shuffle_epi8(aa, bit_idx);
        bit_shift = _mm256_add_epi16(bit_shift, _mm256_set1_epi16(0x0101));
        d0 = _mm256_mullo_epi16(d0, bit_shift);
        d0 = _mm256_srli_epi16(d0, 8); // we should have 8 bits in the LSB
        d1 = _mm256_mullo_epi16(d1, bit_shift);
        d1 = _mm256_and_si256(d1,
                              _mm256_set1_epi32((OPJ_INT32)0xFF00FF00)); // 8 in MSB
        d0 = _mm256_or_si256(d0, d1);

        // find location of e_k and mask
        ones = _mm256_set1_epi32(1);
        twos = _mm256_set1_epi32(2);
        U_q_m1 = _mm256_sub_epi32(U_q, ones);
        U_q_m1 = _mm256_and_si256(U_q_m1,
                                  _mm256_set_epi32(0, 0, 0, 0x1F, 0, 0, 0, 0x1F));
        U_q_m1 = _mm256_shuffle_epi32(U_q_m1, 0);
        w0 = _mm256_sub_epi32(twos, w0);
        shift = _mm256_sllv_epi32(w0, U_q_m1); // U_q_m1 must be no more than 31
        ms_vec = _mm256_and_si256(d0, _mm256_sub_epi32(shift, ones));

        // next e_1
        w0 = _mm256_and_si256(flags, _mm256_set1_epi32(0x800));
        w0 = _mm256_cmpeq_epi32(w0, _mm256_setzero_si256());
        w0 = _mm256_andnot_si256(w0, shift);  // e_1 in correct position
        ms_vec = _mm256_or_si256(ms_vec, w0); // e_1
        w0 = _mm256_slli_epi32(ms_vec, 31);   // sign
        ms_vec = _mm256_or_si256(ms_vec, ones); // bin center
        tvn = ms_vec;
        ms_vec = _mm256_add_epi32(ms_vec, twos);// + 2
        ms_vec = _mm256_sll_epi32(ms_vec, _mm_cvtsi32_si128((int)p - 1));
        ms_vec = _mm256_or_si256(ms_vec, w0); // sign
        row = _mm256_andnot_si256(insig, ms_vec); // significant only

        ms_vec = _mm256_andnot_si256(insig, tvn); // significant only

        tvn = _mm256_shuffle_epi8(ms_vec,
                                  _mm256_set_epi32(-1, 0x0F0E0D0C, 0x07060504, -1,
                                          -1, -1, 0x0F0E0D0C, 0x07060504));

        *vn = _mm_or_si128(*vn, _mm256_castsi256_si128(tvn));
        *vn = _mm_or_si128(*vn, _mm256_extracti128_si256(tvn, 0x1));
    }
    return row;
}

//************************************************************************/
/** @brief decodes four consecutive quads (two octets), using 16 bit data
  *
  *  @param inf_u_q  decoded VLC code, with interleaved u values
  *  @param U_q      U values
  *  @param dbuf     de-stuffed MagSgn buffer
  *  @param limit    clamp offset of dbuf
  *  @param pos      bit position in dbuf, advanced by the bits consumed
  *  @param p        bitplane at which we are decoding (16-bit domain)
  *  @param vn       used for handling E values (stores v_n values)
  *  @return the decoded samples of the four quads, 16 bits each
  */
OPJ_HT_FORCE_INLINE
__m256i decode_four_quad16(const __m128i inf_u_q, __m128i U_q,
                           const OPJ_UINT8* dbuf, OPJ_UINT32 limit,
                           OPJ_UINT32* pos, OPJ_UINT32 p, __m128i* vn)
{
    __m256i w0;     // workers
    __m256i insig;  // lanes hold FF's if samples are insignificant
    __m256i flags;  // lanes hold e_k, e_1, and rho

    __m256i row = _mm256_setzero_si256();
    __m128i ddd = _mm_shuffle_epi8(inf_u_q,
                                   _mm_set_epi16(0x0d0c, 0x0d0c, 0x0908, 0x908,
                                           0x0504, 0x0504, 0x0100, 0x0100));
    w0 = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(ddd),
                                     _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
    // we keeps e_k, e_1, and rho in w2
    flags = _mm256_and_si256(w0,
                             _mm256_set_epi16((OPJ_INT16)0x8880, 0x4440, 0x2220, 0x1110,
                                     (OPJ_INT16)0x8880, 0x4440, 0x2220, 0x1110,
                                     (OPJ_INT16)0x8880, 0x4440, 0x2220, 0x1110,
                                     (OPJ_INT16)0x8880, 0x4440, 0x2220, 0x1110));
    insig = _mm256_cmpeq_epi16(flags, _mm256_setzero_si256());
    //are all insignificant?
    if ((OPJ_UINT32)_mm256_movemask_epi8(insig) != (OPJ_UINT32)0xFFFFFFFF) {
        __m256i U_q_avx, m_n, inc_sum, ex_sum, ms_vec, byte_idx, bit_idx;
        __m256i d0, d1, bit_shift, shift, ones, twos, kq, idx, pow2_tbl, pow2;
        __m256i tvn, shuf1, shuf2;
        __m128i ms_vec0, ms_vec1;
        OPJ_UINT32 total_mn1, total_mn2;

        ddd = _mm_or_si128(_mm_bslli_si128(U_q, 2), U_q);
        U_q_avx = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(ddd),
                                              _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
        flags = _mm256_mullo_epi16(flags, _mm256_set_epi16(1, 2, 4, 8, 1, 2, 4, 8,
                                   1, 2, 4, 8, 1, 2, 4, 8));

        // U_q holds U_q for this quad
        // flags has e_k, e_1, and rho such that e_k is sitting in the
        // 0x8000, e_1 in 0x800, and rho in 0x80

        // next e_k and m_n
        w0 = _mm256_srli_epi16(flags, 15); // e_k
        m_n = _mm256_sub_epi16(U_q_avx, w0);
        m_n = _mm256_andnot_si256(insig, m_n);

        // find cumulative sums
        // to find at which bit in ms_vec the sample starts
        inc_sum = m_n; // inclusive scan
        inc_sum = _mm256_add_epi16(inc_sum, _mm256_bslli_epi128(inc_sum, 2));
        inc_sum = _mm256_add_epi16(inc_sum, _mm256_bslli_epi128(inc_sum, 4));
        inc_sum = _mm256_add_epi16(inc_sum, _mm256_bslli_epi128(inc_sum, 8));
        total_mn1 = (OPJ_UINT32)_mm256_extract_epi16(inc_sum, 7);
        total_mn2 = (OPJ_UINT32)_mm256_extract_epi16(inc_sum, 15);
        ex_sum = _mm256_bslli_epi128(inc_sum, 2); // exclusive scan

        ms_vec0 = dfetch(dbuf, limit, *pos);
        ms_vec1 = dfetch(dbuf, limit, *pos + total_mn1);
        *pos += total_mn1 + total_mn2;

        ms_vec = _mm256_inserti128_si256(_mm256_castsi128_si256(ms_vec0),
                                         ms_vec1, 0x1);

        // find the starting byte and starting bit
        byte_idx = _mm256_srli_epi16(ex_sum, 3);
        bit_idx = _mm256_and_si256(ex_sum, _mm256_set1_epi16(7));
        byte_idx = _mm256_shuffle_epi8(byte_idx,
                                       _mm256_set_epi16(0x0E0E, 0x0C0C, 0x0A0A, 0x0808,
                                               0x0606, 0x0404, 0x0202, 0x0000,
                                               0x0E0E, 0x0C0C, 0x0A0A, 0x0808,
                                               0x0606, 0x0404, 0x0202, 0x0000));
        byte_idx = _mm256_add_epi16(byte_idx, _mm256_set1_epi16(0x0100));
        d0 = _mm256_shuffle_epi8(ms_vec, byte_idx);
        byte_idx = _mm256_add_epi16(byte_idx, _mm256_set1_epi16(0x0101));
        d1 = _mm256_shuffle_epi8(ms_vec, byte_idx);

        // shift samples values to correct location
        bit_shift = _mm256_shuffle_epi8(
                        _mm256_set_epi8(1, 3, 7, 15, 31, 63, 127, -1,
                                        1, 3, 7, 15, 31, 63, 127, -1,
                                        1, 3, 7, 15, 31, 63, 127, -1,
                                        1, 3, 7, 15, 31, 63, 127, -1), bit_idx);
        bit_shift = _mm256_add_epi16(bit_shift, _mm256_set1_epi16(0x0101));
        d0 = _mm256_mullo_epi16(d0, bit_shift);
        d0 = _mm256_srli_epi16(d0, 8); // we should have 8 bits in the LSB
        d1 = _mm256_mullo_epi16(d1, bit_shift);
        d1 = _mm256_and_si256(d1, _mm256_set1_epi16((OPJ_INT16)0xFF00)); // 8 in MSB
        d0 = _mm256_or_si256(d0, d1);

        // find location of e_k and mask
        ones = _mm256_set1_epi16(1);
        twos = _mm256_set1_epi16(2);
        // shift = (2 - e_k) << (U_q - 1); AVX2 has no _mm256_sllv_epi16,
        // so the variable shift is emulated with a pshufb power-of-two
        // lookup and a 16-bit multiply.  U_q - 1 <= 12 in this path;
        // for U_q == 0 the lookup indices have their MSB set, so pshufb
        // returns 0
        kq = _mm256_sub_epi16(U_q_avx, ones);
        idx = _mm256_or_si256(kq,
                              _mm256_slli_epi16(_mm256_sub_epi16(kq,
                                                _mm256_set1_epi16(8)), 8));
        pow2_tbl = _mm256_setr_epi8(
                       1, 2, 4, 8, 16, 32, 64, (char)128, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 2, 4, 8, 16, 32, 64, (char)128, 0, 0, 0, 0, 0, 0, 0, 0);
        pow2 = _mm256_shuffle_epi8(pow2_tbl, idx);
        w0 = _mm256_sub_epi16(twos, w0);
        shift = _mm256_mullo_epi16(w0, pow2);
        ms_vec = _mm256_and_si256(d0, _mm256_sub_epi16(shift, ones));

        // next e_1
        w0 = _mm256_and_si256(flags, _mm256_set1_epi16(0x800));
        w0 = _mm256_cmpeq_epi16(w0, _mm256_setzero_si256());
        w0 = _mm256_andnot_si256(w0, shift);  // e_1 in correct position
        ms_vec = _mm256_or_si256(ms_vec, w0); // e_1
        w0 = _mm256_slli_epi16(ms_vec, 15);   // sign
        ms_vec = _mm256_or_si256(ms_vec, ones); // bin center
        tvn = ms_vec;
        ms_vec = _mm256_add_epi16(ms_vec, twos);// + 2
        ms_vec = _mm256_sll_epi16(ms_vec, _mm_cvtsi32_si128((int)p - 1));
        ms_vec = _mm256_or_si256(ms_vec, w0); // sign
        row = _mm256_andnot_si256(insig, ms_vec); // significant only

        ms_vec = _mm256_andnot_si256(insig, tvn); // significant only

        shuf1 = _mm256_shuffle_epi8(ms_vec,
                                    _mm256_set_epi16(-1, -1, -1, -1, 0x0706, 0x0302, -1, -1,
                                            -1, -1, -1, -1, -1, -1, 0x0706, 0x0302));
        shuf2 = _mm256_shuffle_epi8(ms_vec,
                                    _mm256_set_epi16(-1, -1, -1, 0x0F0E, 0x0B0A, -1, -1, -1,
                                            -1, -1, -1, -1, -1, 0x0F0E, 0x0B0A, -1));
        ms_vec = _mm256_or_si256(shuf1, shuf2);

        *vn = _mm_or_si128(*vn, _mm256_castsi256_si128(ms_vec));
        *vn = _mm_or_si128(*vn, _mm256_extracti128_si256(ms_vec, 0x1));
    }
    return row;
}

//************************************************************************/
/** @brief Counts the leading zeros of eight 32-bit lanes
  *
  *  Uses the exponent of a float conversion, keeping only the 8 MSBs so
  *  that the value is not rounded up to the next power of two.
  */
OPJ_HT_FORCE_INLINE
__m256i avx2_lzcnt_epi32(__m256i v)
{
    // prevent value from being rounded up to the next power of two
    v = _mm256_andnot_si256(_mm256_srli_epi32(v, 8), v);  // keep 8 MSB

    v = _mm256_castps_si256(_mm256_cvtepi32_ps(v)); // convert an integer to float
    v = _mm256_srli_epi32(v, 23);                   // shift down the exponent
    v = _mm256_subs_epu16(_mm256_set1_epi32(158), v);  // undo bias
    v = _mm256_min_epi16(v, _mm256_set1_epi32(32));    // clamp at 32

    return v;
}

//************************************************************************/
/** @brief Step-2 MagSgn decode for the 16-bit (4-quad) path.
  *
  *  @param p is the bitplane in the 32-bit domain; samples are written
  *           in the upper 16 bits of each 32-bit sample
  *  @param bound is the largest acceptable U_q (zero_bplanes + 1)
  *  @return OPJ_HT_SIMD_OK or an U_q error
  */
OPJ_HT_NO_INLINE
opj_ht_simd_result_t decode_cb_step2_16bit(opj_ht_dec_simd_ws_t* ws,
        const OPJ_UINT8* coded_data, OPJ_UINT32 width, OPJ_UINT32 height,
        OPJ_UINT32 stride, OPJ_UINT32 sstr, OPJ_UINT32 p, OPJ_UINT32 bound,
        int lcup, int scup)
{
    OPJ_UINT16* scratch = ws->scratch;
    OPJ_UINT32* decoded_data = ws->out;
    OPJ_UINT16* v_n_scratch = ws->vn16;
    OPJ_UINT32* v_n_scratch_32 = ws->vn32;
    OPJ_UINT8* dbuf = ws->ms_buf;
    OPJ_UINT32 limit, pos = 0;
    OPJ_UINT32 x, y;

    // reduce bitplane by 16 because we now have 16 bits instead of 32
    p -= 16;

    // the loops below read up to 17 entries past (width >> 1)
    memset(v_n_scratch, 0, ((width >> 1) + 24) * sizeof(OPJ_UINT16));

    // maximum consumable MagSgn bits: 4096 samples x (bound <= 13) bits
    limit = destuff_frwd(coded_data, lcup - scup, dbuf,
                         OPJ_HT_SIMD_MS_CAP, 0xFF);

    {
        OPJ_UINT16 *sp = scratch;
        OPJ_UINT16 *vp = v_n_scratch;
        OPJ_UINT32 *dp = decoded_data;
        const __m128i sse_bound = _mm_set1_epi32((int)bound);
        vp[0] = 2; // for easy calculation of emax

        for (x = 0; x < width; x += 8, sp += 8, vp += 4, dp += 8) {
            __m128i inf_u_q, U_q, w, vn;
            __m256i row, w0, w1;

            ////process four quads
            inf_u_q = _mm_loadu_si128((__m128i*)(void*)sp);
            U_q = _mm_srli_epi32(inf_u_q, 16);
            w = _mm_cmpgt_epi32(U_q, sse_bound);
            if (!_mm_testz_si128(w, w)) {
                return OPJ_HT_SIMD_ERR_UQ_INITIAL;
            }

            vn = _mm_set1_epi16(2);
            row = decode_four_quad16(inf_u_q, U_q, dbuf, limit, &pos, p, &vn);

            w = _mm_cvtsi32_si128(*(unsigned short const*)(vp));
            _mm_storeu_si128((__m128i*)(void*)vp, _mm_or_si128(w, vn));

            w0 = _mm256_shuffle_epi8(row,
                                     _mm256_set_epi16(0x0D0C, -1, 0x0908, -1, 0x0504, -1, 0x0100, -1,
                                             0x0D0C, -1, 0x0908, -1, 0x0504, -1, 0x0100, -1));
            w1 = _mm256_shuffle_epi8(row,
                                     _mm256_set_epi16(0x0F0E, -1, 0x0B0A, -1, 0x0706, -1, 0x0302, -1,
                                             0x0F0E, -1, 0x0B0A, -1, 0x0706, -1, 0x0302, -1));

            _mm256_storeu_si256((__m256i*)(void*)dp, w0);
            _mm256_storeu_si256((__m256i*)(void*)(dp + stride), w1);
        }
    }

    for (y = 2; y < height; y += 2) {
        OPJ_UINT16 *vp;
        OPJ_UINT32 *vp_32;
        OPJ_UINT16 *sp;
        OPJ_UINT32 *dp;
        {
            // perform 15 - count_leading_zeros(*vp) here
            const __m256i avx_bound = _mm256_set1_epi32((int)bound);
            const __m256i avx_31 = _mm256_set1_epi32(31);
            const __m256i avx_f0 = _mm256_set1_epi32(0xF0);
            const __m256i avx_1 = _mm256_set1_epi32(1);
            const __m256i avx_0 = _mm256_setzero_si256();

            vp = v_n_scratch;
            vp_32 = v_n_scratch_32;
            sp = scratch + (y >> 1) * sstr;

            for (x = 0; x <= width; x += 16, vp += 8, sp += 16, vp_32 += 8) {
                __m128i v, v_p1;
                __m256i v_avx, inf_u_q, gamma, w0;

                v = _mm_loadu_si128((__m128i*)(void*)vp);
                v_p1 = _mm_loadu_si128((__m128i*)(void*)(vp + 1));
                v = _mm_or_si128(v, v_p1);

                v_avx = _mm256_cvtepu16_epi32(v);
                v_avx = avx2_lzcnt_epi32(v_avx);
                v_avx = _mm256_sub_epi32(avx_31, v_avx);

                inf_u_q = _mm256_loadu_si256((__m256i*)(void*)sp);
                gamma = _mm256_and_si256(inf_u_q, avx_f0);
                w0 = _mm256_sub_epi32(gamma, avx_1);
                gamma = _mm256_and_si256(gamma, w0);
                gamma = _mm256_cmpeq_epi32(gamma, avx_0);

                v_avx = _mm256_andnot_si256(gamma, v_avx);
                v_avx = _mm256_max_epi32(v_avx, avx_1);

                inf_u_q = _mm256_srli_epi32(inf_u_q, 16);
                v_avx = _mm256_add_epi32(inf_u_q, v_avx);

                w0 = _mm256_cmpgt_epi32(v_avx, avx_bound);
                if (!_mm256_testz_si256(w0, w0)) {
                    return OPJ_HT_SIMD_ERR_UQ;
                }

                _mm256_storeu_si256((__m256i*)(void*)vp_32, v_avx);
            }
        }

        vp = v_n_scratch;
        vp_32 = v_n_scratch_32;
        sp = scratch + (y >> 1) * sstr;
        dp = decoded_data + y * stride;
        vp[0] = 2; // for easy calculation of emax

        for (x = 0; x < width; x += 8, sp += 8, vp += 4, dp += 8, vp_32 += 4) {
            __m128i inf_u_q, U_q, vn, w;
            __m256i row, w0, w1;

            ////process four quads
            inf_u_q = _mm_loadu_si128((__m128i*)(void*)sp);
            U_q = _mm_loadu_si128((__m128i*)(void*)vp_32);

            vn = _mm_set1_epi16(2);
            row = decode_four_quad16(inf_u_q, U_q, dbuf, limit, &pos, p, &vn);

            w = _mm_cvtsi32_si128(*(unsigned short const*)(vp));
            _mm_storeu_si128((__m128i*)(void*)vp, _mm_or_si128(w, vn));

            w0 = _mm256_shuffle_epi8(row,
                                     _mm256_set_epi16(0x0D0C, -1, 0x0908, -1, 0x0504, -1, 0x0100, -1,
                                             0x0D0C, -1, 0x0908, -1, 0x0504, -1, 0x0100, -1));
            w1 = _mm256_shuffle_epi8(row,
                                     _mm256_set_epi16(0x0F0E, -1, 0x0B0A, -1, 0x0706, -1, 0x0302, -1,
                                             0x0F0E, -1, 0x0B0A, -1, 0x0706, -1, 0x0302, -1));

            _mm256_storeu_si256((__m256i*)(void*)dp, w0);
            _mm256_storeu_si256((__m256i*)(void*)(dp + stride), w1);
        }
    }
    return OPJ_HT_SIMD_OK;
}

//************************************************************************/
/** @brief Step-2 MagSgn decode for the 32-bit (2-quad) path.
  *
  *  @param p is the bitplane of the cleanup pass
  *  @param bound is the largest acceptable U_q (zero_bplanes + 1)
  *  @return OPJ_HT_SIMD_OK or an U_q error
  */
OPJ_HT_NO_INLINE
opj_ht_simd_result_t decode_cb_step2_32bit(opj_ht_dec_simd_ws_t* ws,
        const OPJ_UINT8* coded_data, OPJ_UINT32 width, OPJ_UINT32 height,
        OPJ_UINT32 stride, OPJ_UINT32 sstr, OPJ_UINT32 p, OPJ_UINT32 bound,
        int lcup, int scup)
{
    const OPJ_UINT32 v_n_size = OPJ_HT_SIMD_VN_SIZE;
    OPJ_UINT16* scratch = ws->scratch;
    OPJ_UINT32* decoded_data = ws->out;
    OPJ_UINT32* v_n_scratch = ws->vn32;
    OPJ_UINT8* dbuf = ws->ms_buf;
    OPJ_UINT32 limit, pos = 0;
    OPJ_UINT32 x, y;
    const __m256i avx_bound = _mm256_set1_epi32((int)bound);

    // the loops below read up to 17 entries past (width >> 1)
    memset(v_n_scratch, 0, ((width >> 1) + 24) * sizeof(OPJ_UINT32));

    // maximum consumable MagSgn bits: 4096 samples x (bound <= 31) bits
    limit = destuff_frwd(coded_data, lcup - scup, dbuf,
                         OPJ_HT_SIMD_MS_CAP, 0xFF);

    {
        OPJ_UINT16 *sp = scratch;
        OPJ_UINT32 *vp = v_n_scratch;
        OPJ_UINT32 *dp = decoded_data;
        vp[0] = 2; // for easy calculation of emax

        for (x = 0; x < width; x += 4, sp += 4, vp += 2, dp += 4) {
            __m128i vn, w0;
            __m256i inf_u_q, U_q, w, row;

            vn = _mm_set1_epi32(2);

            inf_u_q = _mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(void*)sp));
            inf_u_q = _mm256_permutevar8x32_epi32(inf_u_q,
                                                  _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1));

            U_q = _mm256_srli_epi32(inf_u_q, 16);
            w = _mm256_cmpgt_epi32(U_q, avx_bound);
            if (!_mm256_testz_si256(w, w)) {
                return OPJ_HT_SIMD_ERR_UQ_INITIAL;
            }

            row = decode_two_quad32_avx2(inf_u_q, U_q, dbuf, limit, &pos, p, &vn);
            row = _mm256_permutevar8x32_epi32(row,
                                              _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
            _mm_store_si128((__m128i*)(void*)dp, _mm256_castsi256_si128(row));
            _mm_store_si128((__m128i*)(void*)(dp + stride),
                            _mm256_extracti128_si256(row, 0x1));

            w0 = _mm_cvtsi32_si128(*(int const*)(void*)vp);
            w0 = _mm_or_si128(w0, vn);
            _mm_storeu_si128((__m128i*)(void*)vp, w0);
        }
    }

    for (y = 2; y < height; y += 2) {
        OPJ_UINT32 *vp;
        OPJ_UINT16 *sp;
        OPJ_UINT32 *dp;
        {
            // perform 31 - count_leading_zeros(*vp) here
            const __m256i avx_31 = _mm256_set1_epi32(31);
            const __m256i avx_f0 = _mm256_set1_epi32(0xF0);
            const __m256i avx_1 = _mm256_set1_epi32(1);
            const __m256i avx_0 = _mm256_setzero_si256();

            vp = v_n_scratch;
            sp = scratch + (y >> 1) * sstr;

            for (x = 0; x <= width; x += 16, vp += 8, sp += 16) {
                __m256i v, v_p1, inf_u_q, gamma, w0;

                v = _mm256_loadu_si256((__m256i*)(void*)vp);
                v_p1 = _mm256_loadu_si256((__m256i*)(void*)(vp + 1));
                v = _mm256_or_si256(v, v_p1);
                v = avx2_lzcnt_epi32(v);
                v = _mm256_sub_epi32(avx_31, v);

                inf_u_q = _mm256_loadu_si256((__m256i*)(void*)sp);
                gamma = _mm256_and_si256(inf_u_q, avx_f0);
                w0 = _mm256_sub_epi32(gamma, avx_1);
                gamma = _mm256_and_si256(gamma, w0);
                gamma = _mm256_cmpeq_epi32(gamma, avx_0);

                v = _mm256_andnot_si256(gamma, v);
                v = _mm256_max_epi32(v, avx_1);

                inf_u_q = _mm256_srli_epi32(inf_u_q, 16);
                v = _mm256_add_epi32(inf_u_q, v);

                w0 = _mm256_cmpgt_epi32(v, avx_bound);
                if (!_mm256_testz_si256(w0, w0)) {
                    return OPJ_HT_SIMD_ERR_UQ;
                }

                _mm256_storeu_si256((__m256i*)(void*)(vp + v_n_size), v);
            }
        }

        vp = v_n_scratch;
        sp = scratch + (y >> 1) * sstr;
        dp = decoded_data + y * stride;
        vp[0] = 2; // for easy calculation of emax

        for (x = 0; x < width; x += 4, sp += 4, vp += 2, dp += 4) {
            __m128i vn, w0;
            __m256i inf_u_q, U_q, row;

            //process two quads
            vn = _mm_set1_epi32(2);

            inf_u_q = _mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(void*)sp));
            inf_u_q = _mm256_permutevar8x32_epi32(inf_u_q,
                                                  _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1));

            U_q = _mm256_castsi128_si256(
                      _mm_loadl_epi64((__m128i*)(void*)(vp + v_n_size)));
            U_q = _mm256_permutevar8x32_epi32(U_q,
                                              _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1));

            row = decode_two_quad32_avx2(inf_u_q, U_q, dbuf, limit, &pos, p, &vn);
            row = _mm256_permutevar8x32_epi32(row,
                                              _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
            _mm_store_si128((__m128i*)(void*)dp, _mm256_castsi256_si128(row));
            _mm_store_si128((__m128i*)(void*)(dp + stride),
                            _mm256_extracti128_si256(row, 0x1));

            w0 = _mm_cvtsi32_si128(*(int const*)(void*)vp);
            w0 = _mm_or_si128(w0, vn);
            _mm_storeu_si128((__m128i*)(void*)vp, w0);
        }
    }
    return OPJ_HT_SIMD_OK;
}

//************************************************************************/
/** @brief Spreads the 16 bits of v into 16 bytes, each 0xFF or 0
  */
OPJ_HT_FORCE_INLINE
__m128i spread_bits16(OPJ_UINT32 v)
{
    __m128i r = _mm_set1_epi16((OPJ_INT16)(OPJ_UINT16)v);
    r = _mm_shuffle_epi8(r, _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0));
    r = _mm_and_si128(r, _mm_set1_epi64x((OPJ_INT64)0x8040201008040201));
    return _mm_cmpeq_epi8(r, _mm_set1_epi64x((OPJ_INT64)0x8040201008040201));
}

//************************************************************************/
/** @brief Significance-Propagation and Magnitude-Refinement passes.
  *
  *  Only runs when num_passes > 1.  Operates on the padded sample
  *  buffer; p is the bitplane of the cleanup pass in that buffer.
  */
OPJ_HT_NO_INLINE
void decode_cb_spp_mrp(opj_ht_dec_simd_ws_t* ws,
                       const OPJ_UINT8* coded_data, OPJ_UINT32 width,
                       OPJ_UINT32 height, OPJ_UINT32 stride, OPJ_UINT32 sstr,
                       OPJ_UINT32 p, OPJ_UINT32 num_passes,
                       OPJ_UINT32 lengths1, OPJ_UINT32 lengths2,
                       OPJ_BOOL stripe_causal)
{
    // We use scratch again, we can divide it into multiple regions
    // sigma holds all the significant samples, and it cannot
    // be modified after it is set.  it will be used during the
    // Magnitude Refinement Pass
    OPJ_UINT16* const scratch = ws->scratch;
    OPJ_UINT16* const sigma = scratch;
    OPJ_UINT32* decoded_data = ws->out;
    OPJ_UINT32 mstr;
    OPJ_UINT32 x, y;

    mstr = (width + 3u) >> 2;         // divide by 4, since each
    // ui16 contains 4 columns
    mstr = ((mstr + 2u) + 7u) & ~7u; // multiples of 8

    // We re-arrange quad significance, where each 4 consecutive
    // bits represent one quad, into column significance, where,
    // each 4 consequtive bits represent one column of 4 rows
    {
        const __m128i mask_3 = _mm_set1_epi32(0x30);
        const __m128i mask_C = _mm_set1_epi32(0xC0);
        const __m128i shuffle_mask = _mm_set_epi32(-1, -1, -1, 0x0C080400);
        OPJ_UINT16* dp;
        for (y = 0; y < height; y += 4) {
            OPJ_UINT16* sp = scratch + (y >> 1) * sstr;
            dp = sigma + (y >> 2) * mstr;
            for (x = 0; x < width; x += 8, sp += 8, dp += 2) {
                __m128i s0, s1, u3, uC, t0, t1, r;
                OPJ_UINT32 t;

                s0 = _mm_loadu_si128((__m128i*)(void*)(sp));
                u3 = _mm_and_si128(s0, mask_3);
                u3 = _mm_srli_epi32(u3, 4);
                uC = _mm_and_si128(s0, mask_C);
                uC = _mm_srli_epi32(uC, 2);
                t0 = _mm_or_si128(u3, uC);

                s1 = _mm_loadu_si128((__m128i*)(void*)(sp + sstr));
                u3 = _mm_and_si128(s1, mask_3);
                u3 = _mm_srli_epi32(u3, 2);
                uC = _mm_and_si128(s1, mask_C);
                t1 = _mm_or_si128(u3, uC);

                r = _mm_or_si128(t0, t1);
                r = _mm_shuffle_epi8(r, shuffle_mask);

                t = (OPJ_UINT32)_mm_extract_epi32(r, 0);
                memcpy(dp, &t, 4);
            }
            dp[0] = 0; // set an extra entry on the right with 0
        }
        {
            // reset one row after the codeblock
            __m128i zero = _mm_setzero_si128();
            dp = sigma + (y >> 2) * mstr;
            for (x = 0; x < width; x += 32, dp += 8) {
                _mm_storeu_si128((__m128i*)(void*)dp, zero);
            }
            dp[0] = 0; // set an extra entry on the right with 0
        }
    }

    // We perform Significance Propagation Pass here
    {
        // This stores significance information of the previous
        // 4 rows.  Significance information in this array includes
        // all signicant samples in bitplane p - 1; that is,
        // significant samples for bitplane p (discovered during the
        // cleanup pass and stored in sigma) and samples that have recently
        // became significant (during the SPP) in bitplane p-1.
        // We store enough for the widest row, containing 1024 columns,
        // which is equivalent to 256 of ui16, since each stores 4 columns.
        // We add an extra 8 entries, just in case we need more
        OPJ_UINT16 prev_row_sig[256 + 8];
        OPJ_UINT8* spp_buf = ws->spp_buf;
        OPJ_UINT32 spp_limit;
        OPJ_UINT32 spp_pos = 0;

        memset(prev_row_sig, 0, sizeof(prev_row_sig));

        // maximum consumable SPP bits: 4096 samples x 2 bits (one
        // significance bit and one sign bit per sample)
        spp_limit = destuff_frwd(coded_data + lengths1, (int)lengths2,
                                 spp_buf, OPJ_HT_SIMD_SPP_CAP, 0);

        for (y = 0; y < height; y += 4) {
            OPJ_UINT32 pattern = 0xFFFFu; // a pattern needed samples
            // prev holds sign. info. for the previous quad, together
            // with the rows on top of it and below it.
            OPJ_UINT32 prev = 0;
            OPJ_UINT16 *prev_sig = prev_row_sig;
            OPJ_UINT16 *cur_sig = sigma + (y >> 2) * mstr;
            OPJ_UINT32 *dpp = decoded_data + y * stride;

            if (height - y < 4) {
                pattern = 0x7777u;
                if (height - y < 3) {
                    pattern = 0x3333u;
                    if (height - y < 2) {
                        pattern = 0x1111u;
                    }
                }
            }

            for (x = 0; x < width; x += 4, dpp += 4, ++cur_sig, ++prev_sig) {
                OPJ_INT32 s;
                OPJ_UINT32 ps, ns, cs, u, mbr, t, new_sig;

                // only rows and columns inside the stripe are included
                s = (OPJ_INT32)x + 4 - (OPJ_INT32)width;
                s = s > 0 ? s : 0;
                pattern = pattern >> (s * 4);

                // We first find locations that need to be tested (potential
                // SPP members); these location will end up in mbr
                // In each iteration, we produce 16 bits because cwd can have
                // up to 16 bits of significance information, followed by the
                // corresponding 16 bits of sign information; therefore, it is
                // sufficient to fetch 32 bit data per loop.

                // Althougth we are interested in 16 bits only, we load 32 bits.
                // For the 16 bits we are producing, we need the next 4 bits --
                // We need data for at least 5 columns out of 8.
                // Therefore loading 32 bits is easier than loading 16 bits
                // twice.
                memcpy(&ps, prev_sig, 4);
                memcpy(&ns, cur_sig + mstr, 4);
                u = (ps & 0x88888888) >> 3; // the row on top
                if (!stripe_causal) {
                    u |= (ns & 0x11111111) << 3;    // the row below
                }

                memcpy(&cs, cur_sig, 4);
                // vertical integration
                mbr = cs;                      // this sig. info.
                mbr |= (cs & 0x77777777) << 1; //above neighbors
                mbr |= (cs & 0xEEEEEEEE) >> 1; //below neighbors
                mbr |= u;
                // horizontal integration
                t = mbr;
                mbr |= t << 4;      // neighbors on the left
                mbr |= t >> 4;      // neighbors on the right
                mbr |= prev >> 12;  // significance of previous group

                // remove outside samples, and already significant samples
                mbr &= pattern;
                mbr &= ~cs;

                // find samples that become significant during the SPP
                new_sig = mbr;
                if (new_sig) {
                    OPJ_UINT64 cwd = dfetch64(spp_buf, spp_limit, spp_pos);
                    OPJ_UINT32 cnt = 0;
                    OPJ_UINT32 col_mask = 0xFu;
                    OPJ_UINT32 inv_sig = ~cs & pattern;
                    OPJ_UINT32 i;

                    for (i = 0; i < 16; i += 4, col_mask <<= 4) {
                        OPJ_UINT32 sample_mask;
                        if ((col_mask & new_sig) == 0) {
                            continue;
                        }

                        //scan one column
                        sample_mask = 0x1111u & col_mask;
                        if (new_sig & sample_mask) {
                            new_sig &= ~sample_mask;
                            if (cwd & 1) {
                                OPJ_UINT32 tt = 0x33u << i;
                                new_sig |= tt & inv_sig;
                            }
                            cwd >>= 1;
                            ++cnt;
                        }

                        sample_mask <<= 1;
                        if (new_sig & sample_mask) {
                            new_sig &= ~sample_mask;
                            if (cwd & 1) {
                                OPJ_UINT32 tt = 0x76u << i;
                                new_sig |= tt & inv_sig;
                            }
                            cwd >>= 1;
                            ++cnt;
                        }

                        sample_mask <<= 1;
                        if (new_sig & sample_mask) {
                            new_sig &= ~sample_mask;
                            if (cwd & 1) {
                                OPJ_UINT32 tt = 0xECu << i;
                                new_sig |= tt & inv_sig;
                            }
                            cwd >>= 1;
                            ++cnt;
                        }

                        sample_mask <<= 1;
                        if (new_sig & sample_mask) {
                            new_sig &= ~sample_mask;
                            if (cwd & 1) {
                                OPJ_UINT32 tt = 0xC8u << i;
                                new_sig |= tt & inv_sig;
                            }
                            cwd >>= 1;
                            ++cnt;
                        }
                    }

                    if (new_sig) {
                        // the sign bits sit right after the cnt consumed bits
                        // of cwd; no reassembly is needed
                        __m128i new_sig_vec, inc_sum, ex_sum, cwd_vec, v, m, val;
                        OPJ_UINT32 *dp;
                        int c;

                        // Spread new_sig, such that each bit is in one byte with a
                        // value of 0 if new_sig bit is 0, and 0xFF if new_sig is 1
                        new_sig_vec = spread_bits16(new_sig);

                        // find cumulative sums
                        // to find which bit in cwd we should extract
                        inc_sum = new_sig_vec; // inclusive scan
                        inc_sum = _mm_abs_epi8(inc_sum); // cvrt to 0 or 1
                        inc_sum = _mm_add_epi8(inc_sum, _mm_bslli_si128(inc_sum, 1));
                        inc_sum = _mm_add_epi8(inc_sum, _mm_bslli_si128(inc_sum, 2));
                        inc_sum = _mm_add_epi8(inc_sum, _mm_bslli_si128(inc_sum, 4));
                        inc_sum = _mm_add_epi8(inc_sum, _mm_bslli_si128(inc_sum, 8));
                        cnt += (OPJ_UINT32)_mm_extract_epi16(inc_sum, 7) >> 8;
                        // exclusive scan
                        ex_sum = _mm_bslli_si128(inc_sum, 1);

                        // Spread cwd, such that each bit is in one byte
                        // with a value of 0 or 1.
                        cwd_vec = spread_bits16((OPJ_UINT32)cwd);
                        cwd_vec = _mm_abs_epi8(cwd_vec);

                        // Obtain bit from cwd_vec correspondig to ex_sum
                        // Basically, collect needed bits from cwd_vec
                        v = _mm_shuffle_epi8(cwd_vec, ex_sum);

                        // load data and set spp coefficients
                        m = _mm_set_epi8(-1, -1, -1, 12, -1, -1, -1, 8,
                                         -1, -1, -1, 4, -1, -1, -1, 0);
                        val = _mm_set1_epi32(3 << (p - 2));
                        dp = dpp;
                        for (c = 0; c < 4; ++ c) {
                            __m128i s0, s0_ns, s0_val;
                            // load coefficients
                            s0 = _mm_load_si128((__m128i*)(void*)dp);

                            // epi32 is -1 only for coefficient that
                            // are changed during the SPP
                            s0_ns = _mm_shuffle_epi8(new_sig_vec, m);
                            s0_ns = _mm_cmpeq_epi32(s0_ns, _mm_set1_epi32(0xFF));

                            // obtain sign for coefficients in SPP
                            s0_val = _mm_shuffle_epi8(v, m);
                            s0_val = _mm_slli_epi32(s0_val, 31);
                            s0_val = _mm_or_si128(s0_val, val);
                            s0_val = _mm_and_si128(s0_val, s0_ns);

                            // update vector
                            s0 = _mm_or_si128(s0, s0_val);
                            // store coefficients
                            _mm_store_si128((__m128i*)(void*)dp, s0);
                            // prepare for next row
                            dp += stride;
                            m = _mm_add_epi32(m, _mm_set1_epi32(1));
                        }
                    }
                    spp_pos += cnt;
                }

                new_sig |= cs;
                *prev_sig = (OPJ_UINT16)(new_sig);

                // vertical integration for the new sig. info.
                t = new_sig;
                new_sig |= (t & 0x7777) << 1; //above neighbors
                new_sig |= (t & 0xEEEE) >> 1; //below neighbors
                // add sig. info. from the row on top and below
                prev = new_sig | u;
                // we need only the bits in 0xF000
                prev &= 0xF000;
            }
        }
    }

    // We perform Magnitude Refinement Pass here
    if (num_passes > 2) {
        // de-stuff the MRP segment; consumption is at most 1 bit per
        // sample = 512 bytes, so 1024 bytes always suffice
        OPJ_UINT8* mrp_buf = ws->mrp_buf;
        OPJ_UINT32 mrp_limit = destuff_mrp(coded_data, (int)lengths1,
                                           (int)lengths2, mrp_buf,
                                           OPJ_HT_SIMD_MRP_CAP);
        OPJ_UINT32 mrp_pos = 0;

        for (y = 0; y < height; y += 4) {
            OPJ_UINT16 *cur_sig = sigma + (y >> 2) * mstr;
            OPJ_UINT32 *dpp = decoded_data + y * stride;
            OPJ_UINT32 i;
            for (i = 0; i < width; i += 4, dpp += 4) {
                //Process one entry from sigma array at a time
                // Each nibble (4 bits) in the sigma array represents 4 rows,
                OPJ_UINT64 cwd = dfetch64(mrp_buf, mrp_limit, mrp_pos);
                OPJ_UINT16 sig = *cur_sig++; // 16 bit that will be processed now
                OPJ_UINT32 total_bits = 0;
                if (sig) { // if any of the 32 bits are set
                    __m128i sig_vec, inc_sum, ex_sum, cwd_vec, m;
                    OPJ_UINT32 *dp;
                    int c;

                    // We work on 4 rows, with 4 samples each, since
                    // data is 32 bit (4 bytes)

                    // spread the 16 bits in sig to 0 or 1 bytes in sig_vec
                    sig_vec = spread_bits16(sig);
                    sig_vec = _mm_abs_epi8(sig_vec);

                    // find cumulative sums
                    // to find which bit in cwd we should extract
                    inc_sum = sig_vec; // inclusive scan
                    inc_sum = _mm_add_epi8(inc_sum, _mm_bslli_si128(inc_sum, 1));
                    inc_sum = _mm_add_epi8(inc_sum, _mm_bslli_si128(inc_sum, 2));
                    inc_sum = _mm_add_epi8(inc_sum, _mm_bslli_si128(inc_sum, 4));
                    inc_sum = _mm_add_epi8(inc_sum, _mm_bslli_si128(inc_sum, 8));
                    total_bits = (OPJ_UINT32)_mm_extract_epi16(inc_sum, 7) >> 8;
                    ex_sum = _mm_bslli_si128(inc_sum, 1); // exclusive scan

                    // Spread the 16 bits in cwd to inverted 0 or 1 bytes in
                    // cwd_vec. Then, convert these to a form suitable
                    // for coefficient modifications; in particular, a value
                    // of 0 is presented as binary 11, and a value of 1 is
                    // represented as binary 01
                    cwd_vec = spread_bits16((OPJ_UINT32)cwd);
                    cwd_vec = _mm_add_epi8(cwd_vec, _mm_set1_epi8(1));
                    cwd_vec = _mm_add_epi8(cwd_vec, cwd_vec);
                    cwd_vec = _mm_or_si128(cwd_vec, _mm_set1_epi8(1));

                    // load data and insert the mrp bit
                    m = _mm_set_epi8(-1, -1, -1, 12, -1, -1, -1, 8,
                                     -1, -1, -1, 4, -1, -1, -1, 0);
                    dp = dpp;
                    for (c = 0; c < 4; ++c) {
                        __m128i s0, s0_sig, s0_idx, s0_val;
                        // load coefficients
                        s0 = _mm_load_si128((__m128i*)(void*)dp);
                        // find significant samples in this row
                        s0_sig = _mm_shuffle_epi8(sig_vec, m);
                        s0_sig = _mm_cmpeq_epi8(s0_sig, _mm_setzero_si128());
                        // get MRP bit index, and MRP pattern
                        s0_idx = _mm_shuffle_epi8(ex_sum, m);
                        s0_val = _mm_shuffle_epi8(cwd_vec, s0_idx);
                        // keep data from significant samples only
                        s0_val = _mm_andnot_si128(s0_sig, s0_val);
                        // move mrp bits to correct position, and employ
                        s0_val = _mm_sll_epi32(s0_val, _mm_cvtsi32_si128((int)p - 2));
                        s0 = _mm_xor_si128(s0, s0_val);
                        // store coefficients
                        _mm_store_si128((__m128i*)(void*)dp, s0);
                        // prepare for next row
                        dp += stride;
                        m = _mm_add_epi32(m, _mm_set1_epi32(1));
                    }
                }
                // consume data according to the number of bits set
                mrp_pos += total_bits;
            }
        }
    }
}

//************************************************************************/
/** @brief Rejects quad pairs whose VLC codewords mark samples outside the
  *         code-block significant, as the scalar decoder does.
  *
  *  Only the last quad pair of a row (partial width) and the last quad
  *  row of an odd-height code-block can hold such samples.
  *
  *  @param sp is the first record of the quad row in scratch
  *  @param width is the code-block width
  *  @param last_half is true when only the upper row of the quads is
  *         inside the code-block
  *  @return OPJ_TRUE if the quad row is valid
  */
static INLINE
OPJ_BOOL check_quad_row(const OPJ_UINT16* sp, OPJ_UINT32 width,
                        OPJ_BOOL last_half)
{
    OPJ_UINT32 x = last_half ? 0 : ((width - 1) & ~3u);
    for (; x < width; x += 4) {
        OPJ_UINT32 t0 = sp[x], t1 = sp[x + 2];
        OPJ_UINT32 locs = 0xFF;
        if (x + 4 > width) {
            locs >>= (x + 4 - width) << 1;    // limits width
        }
        if (last_half) {
            locs &= 0x55;    // limits height
        }
        if ((((t0 & 0xF0) >> 4) | (t1 & 0xF0)) & ~locs) {
            return OPJ_FALSE;
        }
    }
    return OPJ_TRUE;
}

//************************************************************************/
/** @brief Step-1: decode the VLC and MEL segments into the scratch buffer.
  *
  *  @return OPJ_HT_SIMD_OK, OPJ_HT_SIMD_ERR_MEL or OPJ_HT_SIMD_ERR_OUTSIDE
  */
OPJ_HT_NO_INLINE
opj_ht_simd_result_t decode_cb_step1_vlc(opj_ht_dec_simd_ws_t* ws,
        const OPJ_UINT8* coded_data, int lcup, int scup, OPJ_UINT32 width,
        OPJ_UINT32 height, OPJ_UINT32 sstr)
{
    OPJ_UINT16* scratch = ws->scratch;
    const OPJ_UINT16* uvlc_tbl0 = ws->uvlc_tbl0;
    const OPJ_UINT16* uvlc_tbl1 = ws->uvlc_tbl1;
    const OPJ_UINT32* uvlc_tbl1_wide = ws->uvlc_tbl1_wide;
    OPJ_UINT8* vlc_buf = ws->vlc_buf;
    dec_mel_t mel;
    OPJ_UINT32 vlc_limit, vlc_off = 0, vlc_bits = 0;
    OPJ_UINT64 vlc_val = 0;
    OPJ_UINT32 c_q = 0;
    OPJ_UINT16 *sp = scratch;
    OPJ_UINT32 x, y;
    int run;

    // init structures
    if (!mel_init(&mel, coded_data, lcup, scup)) {
        return OPJ_HT_SIMD_ERR_MEL;
    }

    // de-stuff the VLC segment; its size is at most scup - 1 < 4095
    // bytes (scup is a 12-bit value), and consumption per quad pair is
    // at most 7 + 7 + 30 bits, so 4096 bytes always suffice
    vlc_limit = destuff_vlc(coded_data, lcup, scup, vlc_buf,
                            OPJ_HT_SIMD_VLC_CAP);

    run = mel_get_run(&mel); // decode runs of events from MEL bitstrm
    // data represented as runs of 0 events
    // See mel_decode description

    //initial quad row
    for (x = 0; x < width; sp += 4) {
        OPJ_UINT16 t0, t1, u_q;
        OPJ_UINT32 uvlc_mode, uvlc_entry, len, tmp;

        // decode VLC
        /////////////

        // first quad
        drefill(&vlc_val, &vlc_bits, &vlc_off, vlc_buf, vlc_limit);

        //decode VLC using the context c_q and the head of VLC bitstream
        t0 = vlc_tbl0[c_q + (vlc_val & 0x7F)];

        // if context is zero, use one MEL event
        if (c_q == 0) { //zero context
            run -= 2; //subtract 2, since events number if multiplied by 2

            // Is the run terminated in 1? if so, use decoded VLC code,
            // otherwise, discard decoded data, since we will decoded again
            // using a different context
            t0 = (run == -1) ? t0 : 0;

            // is run -1 or -2? this means a run has been consumed
            if (run < 0) {
                run = mel_get_run(&mel);    // get another run
            }
        }
        sp[0] = t0;
        x += 2;

        // prepare context for the next quad; eqn. 1 in ITU T.814
        c_q = ((t0 & 0x10U) << 3) | ((t0 & 0xE0U) << 2);

        //remove data from vlc stream (0 bits are removed if vlc is not used)
        dconsume(&vlc_val, &vlc_bits, t0 & 0x7u);

        //second quad

        //decode VLC using the context c_q and the head of VLC bitstream
        t1 = vlc_tbl0[c_q + (vlc_val & 0x7F)];

        // if context is zero, use one MEL event
        if (c_q == 0 && x < width) { //zero context
            run -= 2; //subtract 2, since events number if multiplied by 2

            // if event is 0, discard decoded t1
            t1 = (run == -1) ? t1 : 0;

            if (run < 0) { // have we consumed all events in a run
                run = mel_get_run(&mel);    // if yes, then get another run
            }
        }
        t1 = x < width ? t1 : 0;
        sp[2] = t1;
        x += 2;

        //prepare context for the next quad, eqn. 1 in ITU T.814
        c_q = ((t1 & 0x10U) << 3) | ((t1 & 0xE0U) << 2);

        //remove data from vlc stream, if qinf is not used, cwdlen is 0
        dconsume(&vlc_val, &vlc_bits, t1 & 0x7u);

        // decode u
        /////////////
        // uvlc_mode is made up of u_offset bits from the quad pair
        uvlc_mode = ((t0 & 0x8U) << 3) | ((t1 & 0x8U) << 4);
        if (uvlc_mode == 0xc0) { // if both u_offset are set, get an event from
            // the MEL run of events
            run -= 2; //subtract 2, since events number if multiplied by 2

            uvlc_mode += (run == -1) ? 0x40 : 0; // increment uvlc_mode by
            // is 0x40

            if (run < 0) { //if run is consumed (run is -1 or -2), get another run
                run = mel_get_run(&mel);
            }
        }

        //decode uvlc_mode to get u for both quads
        uvlc_entry = uvlc_tbl0[uvlc_mode + (vlc_val & 0x3F)];
        //remove total prefix length
        dconsume(&vlc_val, &vlc_bits, uvlc_entry & 0x7u);
        uvlc_entry >>= 3;
        //extract suffixes for quad 0 and 1
        len = uvlc_entry & 0xF;           //suffix length for 2 quads
        tmp = (OPJ_UINT32)vlc_val & ((1u << len) - 1); //suffix value, 2 quads
        dconsume(&vlc_val, &vlc_bits, len);
        uvlc_entry >>= 4;
        // quad 0 length
        len = uvlc_entry & 0x7; // quad 0 suffix length
        uvlc_entry >>= 3;
        u_q = (OPJ_UINT16)(1 + (uvlc_entry & 7) + (tmp & ~(0xFFU << len))); //kap. 1
        sp[1] = u_q;
        u_q = (OPJ_UINT16)(1 + (uvlc_entry >> 3) + (tmp >> len));  //kappa == 1
        sp[3] = u_q;
    }
    sp[0] = sp[1] = 0;

    if (!check_quad_row(scratch, width, height == 1)) {
        return OPJ_HT_SIMD_ERR_OUTSIDE;
    }

    //non initial quad rows
    for (y = 2; y < height; y += 2) {
        c_q = 0;                                // context
        sp = scratch + (y >> 1) * sstr;         // this row of quads

        for (x = 0; x < width; sp += 4) {
            OPJ_UINT16 t0, t1;
            OPJ_UINT32 uvlc_mode, uvlc_entry, total_bits;

            // decode VLC
            /////////////

            // sigma_q (n, ne, nf)
            c_q |= ((sp[0 - (OPJ_INT32)sstr] & 0xA0U) << 2);
            c_q |= ((sp[2 - (OPJ_INT32)sstr] & 0x20U) << 4);

            // first quad
            drefill(&vlc_val, &vlc_bits, &vlc_off, vlc_buf, vlc_limit);

            //decode VLC using the context c_q and the head of VLC bitstream
            t0 = vlc_tbl1[c_q + (vlc_val & 0x7F)];

            // if context is zero, use one MEL event
            if (c_q == 0) { //zero context
                run -= 2; //subtract 2, since events number is multiplied by 2

                // Is the run terminated in 1? if so, use decoded VLC code,
                // otherwise, discard decoded data, since we will decoded again
                // using a different context
                t0 = (run == -1) ? t0 : 0;

                // is run -1 or -2? this means a run has been consumed
                if (run < 0) {
                    run = mel_get_run(&mel);    // get another run
                }
            }
            sp[0] = t0;
            x += 2;

            // prepare context for the next quad; eqn. 2 in ITU T.814
            // sigma_q (w, sw)
            c_q = ((t0 & 0x40U) << 2) | ((t0 & 0x80U) << 1);
            // sigma_q (nw)
            c_q |= sp[0 - (OPJ_INT32)sstr] & 0x80;
            // sigma_q (n, ne, nf)
            c_q |= ((sp[2 - (OPJ_INT32)sstr] & 0xA0U) << 2);
            c_q |= ((sp[4 - (OPJ_INT32)sstr] & 0x20U) << 4);

            //remove data from vlc stream (0 bits are removed if vlc is unused)
            dconsume(&vlc_val, &vlc_bits, t0 & 0x7u);

            //second quad

            //decode VLC using the context c_q and the head of VLC bitstream
            t1 = vlc_tbl1[c_q + (vlc_val & 0x7F)];

            // if context is zero, use one MEL event
            if (c_q == 0 && x < width) { //zero context
                run -= 2; //subtract 2, since events number if multiplied by 2

                // if event is 0, discard decoded t1
                t1 = (run == -1) ? t1 : 0;

                if (run < 0) { // have we consumed all events in a run
                    run = mel_get_run(&mel);    // if yes, then get another run
                }
            }
            t1 = x < width ? t1 : 0;
            sp[2] = t1;
            x += 2;

            // partial c_q, will be completed when we process the next quad
            // sigma_q (w, sw)
            c_q = ((t1 & 0x40U) << 2) | ((t1 & 0x80U) << 1);
            // sigma_q (nw)
            c_q |= sp[2 - (OPJ_INT32)sstr] & 0x80;

            //remove data from vlc stream, if qinf is not used, cwdlen is 0
            dconsume(&vlc_val, &vlc_bits, t1 & 0x7u);

            // decode u using wide UVLC table
            /////////////
            uvlc_mode = (((t0 >> 3) & 1u) | (((t1 >> 3) & 1u) << 1));
            uvlc_entry = uvlc_tbl1_wide[(uvlc_mode << 10) | (vlc_val & 0x3FF)];
            total_bits = uvlc_entry & 0x1F;
            if (total_bits < 0x1F) {
                sp[1] = (OPJ_UINT16)((uvlc_entry >> 5) & 0xFF);
                sp[3] = (OPJ_UINT16)((uvlc_entry >> 13) & 0xFF);
                dconsume(&vlc_val, &vlc_bits, total_bits);
            } else {
                OPJ_UINT32 len, tmp;
                uvlc_mode = ((t0 & 0x8U) << 3) | ((t1 & 0x8U) << 4);
                uvlc_entry = uvlc_tbl1[uvlc_mode + (vlc_val & 0x3F)];
                dconsume(&vlc_val, &vlc_bits, uvlc_entry & 0x7u);
                uvlc_entry >>= 3;
                len = uvlc_entry & 0xF;
                tmp = (OPJ_UINT32)vlc_val & ((1u << len) - 1);
                dconsume(&vlc_val, &vlc_bits, len);
                uvlc_entry >>= 4;
                len = uvlc_entry & 0x7;
                uvlc_entry >>= 3;
                sp[1] = (OPJ_UINT16)((uvlc_entry & 7) + (tmp & ~(0xFFU << len)));
                sp[3] = (OPJ_UINT16)((uvlc_entry >> 3) + (tmp >> len));
            }
        }
        sp[0] = sp[1] = 0;

        if (!check_quad_row(scratch + (y >> 1) * sstr, width, y + 2 > height)) {
            return OPJ_HT_SIMD_ERR_OUTSIDE;
        }
    }
    return OPJ_HT_SIMD_OK;
}

//************************************************************************/
/** @brief Builds the UVLC decoding tables of the SIMD decoders
  *
  *  uvlc_tbl0 (initial quad row) and uvlc_tbl1 (other rows) are indexed
  *  by the 6 LSBs of the VLC bitstream and, above them, the u_off bits of
  *  the two quads (+ 4 * the MEL event in the initial row when both are
  *  set). Each entry holds, from the LSB: total prefix length (3 bits),
  *  total suffix length (4 bits), quad 0 suffix length (3 bits), quad 0
  *  prefix value (3 bits) and quad 1 prefix value (3 bits).
  *
  *  uvlc_tbl1_wide is indexed by mode (2 bits) * 1024 + 10 bits of the
  *  VLC bitstream; each entry holds the total length (bits 0-4), u_q0
  *  (bits 5-12) and u_q1 (bits 13-20), or 0x1F when the codewords are
  *  longer than 10 bits and uvlc_tbl1 must be used.
  *
  *  Ported from OpenJPH's uvlc_init_tables and uvlc_init_wide_table.
  */
void opj_ht_dec_avx2_init(opj_ht_dec_simd_ws_t* ws)
{
    // table stores possible decoding three bits from vlc
    // there are 8 entries for xx1, x10, 100, 000, where x means do not
    // care. Table value is made up of
    // 2 bits in the LSB for prefix length
    // 3 bits for suffix length
    // 3 bits in the MSB for prefix value (u_pfx in Table 3 of ITU T.814)
    static const OPJ_UINT8 dec[8] = { // the index is the prefix codeword
        3 | (5 << 2) | (5 << 5), //000 == 000, prefix codeword "000"
        1 | (0 << 2) | (1 << 5), //001 == xx1, prefix codeword "1"
        2 | (0 << 2) | (2 << 5), //010 == x10, prefix codeword "01"
        1 | (0 << 2) | (1 << 5), //011 == xx1, prefix codeword "1"
        3 | (1 << 2) | (3 << 5), //100 == 100, prefix codeword "001"
        1 | (0 << 2) | (1 << 5), //101 == xx1, prefix codeword "1"
        2 | (0 << 2) | (2 << 5), //110 == x10, prefix codeword "01"
        1 | (0 << 2) | (1 << 5)  //111 == xx1, prefix codeword "1"
    };
    OPJ_UINT32 i;

    for (i = 0; i < 256 + 64; ++i) {
        OPJ_UINT32 mode = i >> 6;
        OPJ_UINT32 vlc = i & 0x3F;
        OPJ_UINT32 total_prefix = 0, total_suffix = 0, u0_suffix_len = 0;
        OPJ_UINT32 u0 = 0, u1 = 0;

        if (mode == 0) { // both u_off are 0
        } else if (mode <= 2) { // u_off are either 01 or 10
            OPJ_UINT32 d = dec[vlc & 0x7];   //look at the least significant 3 bits
            total_prefix = d & 0x3;
            total_suffix = (d >> 2) & 0x7;
            u0_suffix_len = (mode == 1) ? total_suffix : 0;
            u0 = (mode == 1) ? (d >> 5) : 0;
            u1 = (mode == 1) ? 0 : (d >> 5);
        } else if (mode == 3) { // both u_off are 1, and MEL event is 0
            OPJ_UINT32 d0 = dec[vlc & 0x7];  // LSBs of VLC are prefix codeword
            OPJ_UINT32 d1;
            vlc >>= d0 & 0x3;          // Consume bits
            d1 = dec[vlc & 0x7];  // LSBs of VLC are prefix codeword
            if ((d0 & 0x3) == 3) {
                total_prefix = (d0 & 0x3) + 1;
                u0_suffix_len = (d0 >> 2) & 0x7;
                total_suffix = u0_suffix_len;
                u0 = d0 >> 5;
                u1 = (vlc & 1) + 1;
            } else {
                total_prefix = (d0 & 0x3) + (d1 & 0x3);
                u0_suffix_len = (d0 >> 2) & 0x7;
                total_suffix = u0_suffix_len + ((d1 >> 2) & 0x7);
                u0 = d0 >> 5;
                u1 = d1 >> 5;
            }
        } else { // mode == 4, both u_off are 1, and MEL event is 1
            OPJ_UINT32 d0 = dec[vlc & 0x7];  // LSBs of VLC are prefix codeword
            OPJ_UINT32 d1;
            vlc >>= d0 & 0x3;          // Consume bits
            d1 = dec[vlc & 0x7];  // LSBs of VLC are prefix codeword
            total_prefix = (d0 & 0x3) + (d1 & 0x3);
            u0_suffix_len = (d0 >> 2) & 0x7;
            total_suffix = u0_suffix_len + ((d1 >> 2) & 0x7);
            u0 = (d0 >> 5) + 2;
            u1 = (d1 >> 5) + 2;
        }
        ws->uvlc_tbl0[i] = (OPJ_UINT16)(total_prefix | (total_suffix << 3) |
                                        (u0_suffix_len << 7) | (u0 << 10) |
                                        (u1 << 13));
    }

    for (i = 0; i < 256; ++i) {
        OPJ_UINT32 mode = i >> 6;
        OPJ_UINT32 vlc = i & 0x3F;
        OPJ_UINT32 total_prefix = 0, total_suffix = 0, u0_suffix_len = 0;
        OPJ_UINT32 u0 = 0, u1 = 0;

        if (mode == 0) { // both u_off are 0
        } else if (mode <= 2) { // u_off are either 01 or 10
            OPJ_UINT32 d = dec[vlc & 0x7];   // look at the 3 LSB bits
            total_prefix = d & 0x3;
            total_suffix = (d >> 2) & 0x7;
            u0_suffix_len = (mode == 1) ? total_suffix : 0;
            u0 = (mode == 1) ? (d >> 5) : 0;
            u1 = (mode == 1) ? 0 : (d >> 5);
        } else { // mode == 3, both u_off are 1
            OPJ_UINT32 d0 = dec[vlc & 0x7];  // LSBs of VLC are prefix codeword
            OPJ_UINT32 d1;
            vlc >>= d0 & 0x3;          // Consume bits
            d1 = dec[vlc & 0x7];  // LSBs of VLC are prefix codeword
            total_prefix = (d0 & 0x3) + (d1 & 0x3);
            u0_suffix_len = (d0 >> 2) & 0x7;
            total_suffix = u0_suffix_len + ((d1 >> 2) & 0x7);
            u0 = d0 >> 5;
            u1 = d1 >> 5;
        }
        ws->uvlc_tbl1[i] = (OPJ_UINT16)(total_prefix | (total_suffix << 3) |
                                        (u0_suffix_len << 7) | (u0 << 10) |
                                        (u1 << 13));
    }

    for (i = 0; i < 4096; ++i) {
        OPJ_UINT32 mode = i >> 10;      // 2 bits
        OPJ_UINT32 vlc = i & 0x3FF;     // 10 bits
        OPJ_UINT32 d0, p0_len, s0_len, u0_pfx, d1, p1_len, s1_len, u1_pfx;
        OPJ_UINT32 total_prefix, total, suffix_bits, s0_val, s1_val;

        if (mode == 0) {
            ws->uvlc_tbl1_wide[i] = 0;
            continue;
        }
        if (mode <= 2) { // single UVLC (one u_off set)
            OPJ_UINT32 d = dec[vlc & 0x7];
            OPJ_UINT32 prefix_len = d & 0x3;
            OPJ_UINT32 suffix_len = (d >> 2) & 0x7;
            OPJ_UINT32 u_val = (d >> 5) +
                               ((vlc >> prefix_len) & ((1u << suffix_len) - 1));
            OPJ_UINT32 u_q0 = (mode == 1) ? u_val : 0;
            OPJ_UINT32 u_q1 = (mode == 2) ? u_val : 0;
            ws->uvlc_tbl1_wide[i] = (prefix_len + suffix_len) | (u_q0 << 5) |
                                    (u_q1 << 13);
            continue;
        }

        // mode == 3: both u_off set
        // Bitstream layout: [prefix0][prefix1][suffix0][suffix1]
        d0 = dec[vlc & 0x7];
        p0_len = d0 & 0x3;
        s0_len = (d0 >> 2) & 0x7;
        u0_pfx = d0 >> 5;
        d1 = dec[(vlc >> p0_len) & 0x7];  // consume prefix0
        p1_len = d1 & 0x3;
        s1_len = (d1 >> 2) & 0x7;
        u1_pfx = d1 >> 5;

        total_prefix = p0_len + p1_len;
        total = total_prefix + s0_len + s1_len;
        if (total > 10) {
            ws->uvlc_tbl1_wide[i] = 0x1F; // fall back to uvlc_tbl1
            continue;
        }

        // suffixes follow both prefixes in the bitstream
        suffix_bits = vlc >> total_prefix;
        s0_val = suffix_bits & ((1u << s0_len) - 1);
        s1_val = (suffix_bits >> s0_len) & ((1u << s1_len) - 1);
        ws->uvlc_tbl1_wide[i] = total | ((u0_pfx + s0_val) << 5) |
                                ((u1_pfx + s1_val) << 13);
    }
}

//************************************************************************/
/** @brief Converts the decoded sign-magnitude samples to two's complement
  *         and writes them, with a stride of width, to dst.
  *
  *  @param shift is the number of bits the magnitudes are shifted down
  */
OPJ_HT_NO_INLINE
void write_samples(const OPJ_UINT32* src, OPJ_UINT32 stride,
                   OPJ_INT32* dst, OPJ_UINT32 width, OPJ_UINT32 height,
                   OPJ_UINT32 shift)
{
    const __m256i mag_mask = _mm256_set1_epi32(0x7FFFFFFF);
    const __m128i cnt = _mm_cvtsi32_si128((int)shift);
    const __m256i tail = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)(width & 7)),
                                            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    OPJ_UINT32 x, y;

    for (y = 0; y < height; ++y, src += stride, dst += width) {
        for (x = 0; x + 8 <= width; x += 8) {
            __m256i v = _mm256_load_si256((const __m256i*)(const void*)(src + x));
            __m256i sgn = _mm256_srai_epi32(v, 31);
            __m256i mag = _mm256_srl_epi32(_mm256_and_si256(v, mag_mask), cnt);
            // -mag = (mag ^ -1) - (-1)
            _mm256_storeu_si256((__m256i*)(void*)(dst + x),
                                _mm256_sub_epi32(_mm256_xor_si256(mag, sgn), sgn));
        }
        if (x < width) {
            __m256i v = _mm256_load_si256((const __m256i*)(const void*)(src + x));
            __m256i sgn = _mm256_srai_epi32(v, 31);
            __m256i mag = _mm256_srl_epi32(_mm256_and_si256(v, mag_mask), cnt);
            _mm256_maskstore_epi32((int*)(void*)(dst + x), tail,
                                   _mm256_sub_epi32(_mm256_xor_si256(mag, sgn), sgn));
        }
    }
}

//************************************************************************/
opj_ht_simd_result_t opj_ht_dec_cblk_avx2(opj_ht_dec_simd_ws_t* ws,
        const opj_ht_simd_cblk_t* cb, OPJ_INT32* dst)
{
    const OPJ_UINT32 width = cb->width;
    const OPJ_UINT32 height = cb->height;
    // The temporary storage scratch holds two types of data in an
    // interleaved fashion. The interleaving allows us to use one
    // memory pointer.
    // We have one entry for a decoded VLC code, and one entry for UVLC.
    // Entries are 16 bits each, corresponding to one quad,
    // but since we want to use XMM registers of the SSE family
    // of SIMD; we allocated 16 bytes or more per quad row; that is,
    // the width is no smaller than 16 bytes (or 8 entries), and the
    // height is 512 quads
    // Each VLC entry contains, in the following order, starting
    // from MSB
    // e_k (4bits), e_1 (4bits), rho (4bits), useless for step 2 (4bits)
    // Each entry in UVLC contains u_q
    // One extra row to handle the case of SPP propagating downwards
    // when codeblock width is 4
    const OPJ_UINT32 sstr = ((width + 2u) + 7u) & ~7u; // multiples of 8
    // stride of the decoded samples: 8 samples (the 16-bit path writes
    // 8 at a time) and 32 bytes, so aligned loads work on every row
    const OPJ_UINT32 stride = (width + 7u) & ~7u;
    const OPJ_UINT32 quad_rows = (height + 1u) >> 1;
    const OPJ_UINT32 bound = cb->zero_bplanes + 1u;
    OPJ_UINT32 p, shift;
    opj_ht_simd_result_t res;

    memset(ws->scratch, 0,
           ((size_t)(quad_rows + 2u) * sstr + 16u) * sizeof(OPJ_UINT16));

    // step 1: decode VLC and MEL segments into scratch
    res = decode_cb_step1_vlc(ws, cb->coded_data, cb->lcup, cb->scup,
                              width, height, sstr);
    if (res != OPJ_HT_SIMD_OK) {
        return res;
    }

    // step 2: decode the MagSgn segment.
    // The 16-bit path handles four quads per iteration; it computes
    // samples at bitplane 29 - zero_bplanes, whose magnitudes (at most
    // bound + 2 bits) fit in 15 bits when zero_bplanes <= 12, and shifts
    // them down by 28 - Mb at the end, which is exact when Mb <= 28.
    // The 32-bit path computes samples at the scalar decoder's bitplane
    // with the scalar decoder's 32-bit arithmetic.
    if (cb->zero_bplanes <= 12 && cb->Mb <= 28) {
        p = 29u - cb->zero_bplanes;
        shift = 28u - cb->Mb;
        res = decode_cb_step2_16bit(ws, cb->coded_data, width, height,
                                    stride, sstr, p, bound, cb->lcup, cb->scup);
    } else {
        p = cb->numbps;
        shift = 0;
        res = decode_cb_step2_32bit(ws, cb->coded_data, width, height,
                                    stride, sstr, p, bound, cb->lcup, cb->scup);
    }
    if (res != OPJ_HT_SIMD_OK) {
        return res;
    }

    if (cb->num_passes > 1) {
        // the passes read and write whole stripes of 4 rows; clear the
        // rows step 2 did not write
        OPJ_UINT32 h2 = (height + 1u) & ~1u, h4 = (height + 3u) & ~3u;
        if (h4 > h2) {
            memset(ws->out + (size_t)h2 * stride, 0,
                   (size_t)(h4 - h2) * stride * sizeof(OPJ_UINT32));
        }
        decode_cb_spp_mrp(ws, cb->coded_data, width, height, stride, sstr, p,
                          cb->num_passes, cb->lengths1, cb->lengths2,
                          cb->stripe_causal);
    }

    write_samples(ws->out, stride, dst, width, height, shift);
    return OPJ_HT_SIMD_OK;
}
