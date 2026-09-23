//***************************************************************************/
// This software is released under the 2-Clause BSD license, included
// below.
//
// Copyright (c) 2019, Aous Naman
// Copyright (c) 2019, Kakadu Software Pty Ltd, Australia
// Copyright (c) 2019, The University of New South Wales, Australia
// Copyright (c) 2026, Diamond Light Source
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
// File: ht_enc.c
// It is a C port of the generic 32-bit HTJ2K block encoder of OpenJPH
// (src/core/coding/ojph_block_encoder.cpp, OpenJPH 0.32.0), whose author
// is Aous Naman.
//***************************************************************************/

//***************************************************************************/
/** @file ht_enc.c
 *  @brief implements the HTJ2K block encoder (cleanup pass only)
 */

#include <assert.h>
#include <string.h>
#include "opj_includes.h"

#include "t1_ht_enc_luts.h"

/////////////////////////////////////////////////////////////////////////////
// compiler detection
/////////////////////////////////////////////////////////////////////////////
#ifdef _MSC_VER
#define OPJ_COMPILER_MSVC
#elif (defined __GNUC__)
#define OPJ_COMPILER_GNUC
#endif

//************************************************************************/
/** @brief Counts the number of leading zeros
  *
  *   @param [in]  val is the value for which leading zero count is sought;
  *                it must not be zero
  */
#ifdef OPJ_COMPILER_MSVC
#pragma intrinsic(_BitScanReverse)
#endif
static INLINE
OPJ_UINT32 opj_ht_enc_count_leading_zeros(OPJ_UINT32 val)
{
#ifdef OPJ_COMPILER_MSVC
    unsigned long result = 0;
    _BitScanReverse(&result, val);
    return 31U ^ (OPJ_UINT32)result;
#elif (defined OPJ_COMPILER_GNUC)
    return (OPJ_UINT32)__builtin_clz(val);
#else
    OPJ_UINT32 n = 0;
    while ((val & 0x80000000U) == 0) {
        val <<= 1;
        ++n;
    }
    return n;
#endif
}

//************************************************************************/
/** @defgroup ht_enc_uvlc Access to the fields of uvlc_enc_tbl entries
  *  @{
  */
#define OPJ_HT_UVLC_PRE(e)      ((int)((e) & 0x7))
#define OPJ_HT_UVLC_PRE_LEN(e)  ((int)(((e) >> 3) & 0x7))
#define OPJ_HT_UVLC_SUF(e)      ((int)(((e) >> 6) & 0x1F))
#define OPJ_HT_UVLC_SUF_LEN(e)  ((int)(((e) >> 11) & 0x7))
/** @} */

//************************************************************************/
/** @defgroup ht_enc_sizes Sizes of the MEL and VLC buffers
  *  @{
  *  A code-block has at most 4096 samples and 1024 columns, which makes at
  *  most 1024 quads. Every byte of the three bitstreams carries at least
  *  7 bits, the 8th being possibly lost to bit-stuffing.
  *
  *  MEL: one symbol per quad whose context is zero, plus at most one per
  *  pair of quads in the initial row (at most 256 pairs); a symbol costs
  *  at most 6 bits: (1024 + 256) * 6 bits / 7 = 1098 bytes, plus the
  *  terminating byte.
  *
  *  VLC: per quad, a codeword of at most 7 bits and at most 8 bits of
  *  UVLC code (a pair of quads uses at most 2 prefixes of 3 bits and 2
  *  suffixes of 5 bits): (1024 * 15 + 4) bits / 7 = 2195 bytes, plus the
  *  initial byte and the terminating byte.
  *
  *  Scup, the combined length of both, can therefore not exceed its limit
  *  of 4079 bytes; this is still checked.
  */
#define OPJ_HT_MEL_SIZE 1104
#define OPJ_HT_VLC_SIZE 2200
/** @} */

//************************************************************************/
/** @brief MEL encoder state */
typedef struct opj_ht_mel_enc {
    OPJ_UINT8* buf;       /**< data buffer */
    OPJ_UINT32 pos;       /**< position of the next write within buf */
    OPJ_UINT32 buf_size;  /**< size of buf, which must not be exceeded */
    int remaining_bits;   /**< number of empty bits in tmp */
    int tmp;              /**< temporary storage of coded bits */
    int run;              /**< number of 0 run */
    int k;                /**< state */
    int threshold;        /**< threshold where one bit must be coded */
    OPJ_BOOL overflow;    /**< set if buf was too small */
} opj_ht_mel_enc_t;

//************************************************************************/
/** @brief Initialises the MEL encoder
  *
  *   @param [out] melp is the MEL encoder state
  *   @param [in]  buffer_size is the size of data
  *   @param [in]  data is the buffer receiving the MEL bitstream
  */
static INLINE
void opj_ht_mel_init(opj_ht_mel_enc_t* melp, OPJ_UINT32 buffer_size,
                     OPJ_UINT8* data)
{
    melp->buf = data;
    melp->pos = 0;
    melp->buf_size = buffer_size;
    melp->remaining_bits = 8;
    melp->tmp = 0;
    melp->run = 0;
    melp->k = 0;
    melp->threshold = 1; // this is 1 << mel_exp[melp->k];
    melp->overflow = OPJ_FALSE;
}

//************************************************************************/
/** @brief Emits one bit into the MEL bitstream
  *
  *   @param [in, out] melp is the MEL encoder state
  *   @param [in]      v is the bit, 0 or 1
  */
static INLINE
void opj_ht_mel_emit_bit(opj_ht_mel_enc_t* melp, int v)
{
    assert(v == 0 || v == 1);
    melp->tmp = (melp->tmp << 1) + v;
    melp->remaining_bits--;
    if (melp->remaining_bits == 0) {
        if (melp->pos < melp->buf_size) {
            melp->buf[melp->pos++] = (OPJ_UINT8)melp->tmp;
        } else {
            melp->overflow = OPJ_TRUE;
        }
        melp->remaining_bits = (melp->tmp == 0xFF ? 7 : 8);
        melp->tmp = 0;
    }
}

//************************************************************************/
/** @brief Encodes one MEL symbol
  *
  *   @param [in, out] melp is the MEL encoder state
  *   @param [in]      bit is the symbol
  */
static INLINE
void opj_ht_mel_encode(opj_ht_mel_enc_t* melp, OPJ_BOOL bit)
{
    //MEL exponent
    static const int mel_exp[13] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5};

    if (!bit) {
        ++melp->run;
        if (melp->run >= melp->threshold) {
            opj_ht_mel_emit_bit(melp, 1);
            melp->run = 0;
            melp->k = opj_int_min(12, melp->k + 1);
            melp->threshold = 1 << mel_exp[melp->k];
        }
    } else {
        int t;
        opj_ht_mel_emit_bit(melp, 0);
        t = mel_exp[melp->k];
        while (t > 0) {
            opj_ht_mel_emit_bit(melp, (melp->run >> --t) & 1);
        }
        melp->run = 0;
        melp->k = opj_int_max(0, melp->k - 1);
        melp->threshold = 1 << mel_exp[melp->k];
    }
}

//************************************************************************/
/** @brief VLC encoder state; the VLC bitstream is written backwards */
typedef struct opj_ht_vlc_enc {
    OPJ_UINT8* buf;       /**< points to the last byte of the buffer */
    OPJ_UINT32 pos;       /**< bytes written so far; the next is buf[-pos] */
    OPJ_UINT32 buf_size;  /**< size of the buffer, which must not be exceeded */
    int used_bits;        /**< number of occupied bits in tmp */
    int tmp;              /**< temporary storage of coded bits */
    int last_greater_than_8F; /**< 1 if the last byte is greater than 0x8F */
    OPJ_BOOL overflow;    /**< set if the buffer was too small */
} opj_ht_vlc_enc_t;

//************************************************************************/
/** @brief Initialises the VLC encoder
  *
  *   @param [out] vlcp is the VLC encoder state
  *   @param [in]  buffer_size is the size of data
  *   @param [in]  data is the buffer receiving the VLC bitstream, which is
  *                written from its end
  */
static INLINE
void opj_ht_vlc_init(opj_ht_vlc_enc_t* vlcp, OPJ_UINT32 buffer_size,
                     OPJ_UINT8* data)
{
    vlcp->buf = data + buffer_size - 1; //points to last byte
    vlcp->pos = 1;                      //locations will be all -pos
    vlcp->buf_size = buffer_size;

    vlcp->buf[0] = 0xFF;
    vlcp->used_bits = 4;
    vlcp->tmp = 0xF;
    vlcp->last_greater_than_8F = 1;
    vlcp->overflow = OPJ_FALSE;
}

//************************************************************************/
/** @brief Emits a codeword into the VLC bitstream
  *
  *   @param [in, out] vlcp is the VLC encoder state
  *   @param [in]      cwd is the codeword, emitted LSB first
  *   @param [in]      cwd_len is the number of bits in cwd
  */
static INLINE
void opj_ht_vlc_encode(opj_ht_vlc_enc_t* vlcp, int cwd, int cwd_len)
{
    while (cwd_len > 0) {
        int avail_bits, t;

        if (vlcp->pos >= vlcp->buf_size) {
            vlcp->overflow = OPJ_TRUE;
            return;
        }

        avail_bits = 8 - vlcp->last_greater_than_8F - vlcp->used_bits;
        t = opj_int_min(avail_bits, cwd_len);
        vlcp->tmp |= (cwd & ((1 << t) - 1)) << vlcp->used_bits;
        vlcp->used_bits += t;
        avail_bits -= t;
        cwd_len -= t;
        cwd >>= t;
        if (avail_bits == 0) {
            if (vlcp->last_greater_than_8F && vlcp->tmp != 0x7F) {
                vlcp->last_greater_than_8F = 0;
                continue; //one empty bit remaining
            }
            *(vlcp->buf - vlcp->pos) = (OPJ_UINT8)(vlcp->tmp);
            vlcp->pos++;
            vlcp->last_greater_than_8F = vlcp->tmp > 0x8F;
            vlcp->tmp = 0;
            vlcp->used_bits = 0;
        }
    }
}

//************************************************************************/
/** @brief Terminates the MEL and VLC bitstreams, fusing their last bytes
  *         when possible
  *
  *   @param [in, out] melp is the MEL encoder state
  *   @param [in, out] vlcp is the VLC encoder state
  */
static INLINE
void opj_ht_terminate_mel_vlc(opj_ht_mel_enc_t* melp, opj_ht_vlc_enc_t* vlcp)
{
    int mel_mask, vlc_mask, fuse;

    if (melp->run > 0) {
        opj_ht_mel_emit_bit(melp, 1);
    }

    melp->tmp = melp->tmp << melp->remaining_bits;
    mel_mask = (0xFF << melp->remaining_bits) & 0xFF;
    vlc_mask = 0xFF >> (8 - vlcp->used_bits);
    if ((mel_mask | vlc_mask) == 0) {
        return;    //last mel byte cannot be 0xFF, since then
    }              //melp->remaining_bits would be < 8
    if (melp->pos >= melp->buf_size) {
        melp->overflow = OPJ_TRUE;
        return;
    }
    fuse = melp->tmp | vlcp->tmp;
    if ((((fuse ^ melp->tmp) & mel_mask)
            | ((fuse ^ vlcp->tmp) & vlc_mask)) == 0
            && (fuse != 0xFF) && vlcp->pos > 1) {
        melp->buf[melp->pos++] = (OPJ_UINT8)fuse;
    } else {
        if (vlcp->pos >= vlcp->buf_size) {
            vlcp->overflow = OPJ_TRUE;
            return;
        }
        melp->buf[melp->pos++] = (OPJ_UINT8)melp->tmp; //melp->tmp cannot be 0xFF
        *(vlcp->buf - vlcp->pos) = (OPJ_UINT8)vlcp->tmp;
        vlcp->pos++;
    }
}

//************************************************************************/
/** @brief MagSgn encoder state */
typedef struct opj_ht_ms_enc {
    OPJ_UINT8* buf;       /**< data buffer */
    OPJ_UINT32 pos;       /**< position of the next write within buf */
    OPJ_UINT32 buf_size;  /**< size of buf, which must not be exceeded */
    int max_bits;         /**< maximum number of bits that can be stored in tmp */
    int used_bits;        /**< number of occupied bits in tmp */
    OPJ_UINT32 tmp;       /**< temporary storage of coded bits */
    OPJ_BOOL overflow;    /**< set if buf was too small */
} opj_ht_ms_enc_t;

//************************************************************************/
/** @brief Initialises the MagSgn encoder
  *
  *   @param [out] msp is the MagSgn encoder state
  *   @param [in]  buffer_size is the size of data
  *   @param [in]  data is the buffer receiving the MagSgn bitstream
  */
static INLINE
void opj_ht_ms_init(opj_ht_ms_enc_t* msp, OPJ_UINT32 buffer_size,
                    OPJ_UINT8* data)
{
    msp->buf = data;
    msp->pos = 0;
    msp->buf_size = buffer_size;
    msp->max_bits = 8;
    msp->used_bits = 0;
    msp->tmp = 0;
    msp->overflow = OPJ_FALSE;
}

//************************************************************************/
/** @brief Emits bits into the MagSgn bitstream
  *
  *   @param [in, out] msp is the MagSgn encoder state
  *   @param [in]      cwd holds the bits, emitted LSB first
  *   @param [in]      cwd_len is the number of bits in cwd, at most 31
  */
static INLINE
void opj_ht_ms_encode(opj_ht_ms_enc_t* msp, OPJ_UINT32 cwd, int cwd_len)
{
    while (cwd_len > 0) {
        int t;

        if (msp->pos >= msp->buf_size) {
            msp->overflow = OPJ_TRUE;
            return;
        }
        t = opj_int_min(msp->max_bits - msp->used_bits, cwd_len);
        msp->tmp |= (cwd & ((1U << t) - 1)) << msp->used_bits;
        msp->used_bits += t;
        cwd >>= t;
        cwd_len -= t;
        if (msp->used_bits >= msp->max_bits) {
            msp->buf[msp->pos++] = (OPJ_UINT8)msp->tmp;
            msp->max_bits = (msp->tmp == 0xFF) ? 7 : 8;
            msp->tmp = 0;
            msp->used_bits = 0;
        }
    }
}

//************************************************************************/
/** @brief Terminates the MagSgn bitstream
  *
  *   @param [in, out] msp is the MagSgn encoder state
  */
static INLINE
void opj_ht_ms_terminate(opj_ht_ms_enc_t* msp)
{
    if (msp->used_bits) {
        int t = msp->max_bits - msp->used_bits; //unused bits
        msp->tmp |= (0xFFU & ((1U << t) - 1)) << msp->used_bits;
        msp->used_bits += t;
        if (msp->tmp != 0xFF) {
            if (msp->pos >= msp->buf_size) {
                msp->overflow = OPJ_TRUE;
                return;
            }
            msp->buf[msp->pos++] = (OPJ_UINT8)msp->tmp;
        }
    } else if (msp->max_bits == 7) {
        msp->pos--;
    }
}

//************************************************************************/
/** @brief Examines one sample of a quad
  *
  *   If the sample is significant, sets bit in rho, stores its exponent
  *   E = number of bits of (2 mu - 1) in e_q, updates e_qmax, and stores
  *   2 (mu - 1) + sign in s.
  *
  *   @param [in]      t is the sample: sign in bit 31, magnitude mu below
  *   @param [in]      bit is the bit of rho for this sample
  *   @param [in, out] rho is the significance pattern of the quad
  *   @param [out]     e_q is the exponent of the sample
  *   @param [in, out] e_qmax is the largest exponent in the quad
  *   @param [out]     s is the value coded in the MagSgn bitstream
  */
static INLINE
void opj_ht_enc_sample(OPJ_UINT32 t, int bit, int* rho, int* e_q,
                       int* e_qmax, OPJ_UINT32* s)
{
    OPJ_UINT32 val = t + t; //multiply by 2 and get rid of sign: 2 \mu
    if (val) {
        *rho += bit;
        *e_q = 32 - (int)opj_ht_enc_count_leading_zeros(--val); //2\mu - 1
        *e_qmax = opj_int_max(*e_qmax, *e_q);
        *s = --val + (t >> 31); //v_n = 2(\mu-1) + s_n
    }
}

//************************************************************************/
/** @brief Emits the MagSgn bits of the four samples of a quad
  *
  *   @param [in, out] msp is the MagSgn encoder state
  *   @param [in]      rho is the significance pattern of the quad
  *   @param [in]      U_q is the exponent bound of the quad
  *   @param [in]      tuple is the VLC table entry, whose 4 LSBs are e_k
  *   @param [in]      s are the MagSgn values of the four samples
  */
static INLINE
void opj_ht_enc_quad_magsgn(opj_ht_ms_enc_t* msp, int rho, int U_q,
                            int tuple, const OPJ_UINT32* s)
{
    int m;

    m = (rho & 1) ? U_q - (tuple & 1) : 0;
    opj_ht_ms_encode(msp, s[0] & ((1U << m) - 1), m);
    m = (rho & 2) ? U_q - ((tuple & 2) >> 1) : 0;
    opj_ht_ms_encode(msp, s[1] & ((1U << m) - 1), m);
    m = (rho & 4) ? U_q - ((tuple & 4) >> 2) : 0;
    opj_ht_ms_encode(msp, s[2] & ((1U << m) - 1), m);
    m = (rho & 8) ? U_q - ((tuple & 8) >> 3) : 0;
    opj_ht_ms_encode(msp, s[3] & ((1U << m) - 1), m);
}

//************************************************************************/
/** @brief Computes the EMB pattern of a quad: which samples have the
  *         quad's largest exponent
  */
static INLINE
int opj_ht_enc_eps(const int* e_q, int e_qmax)
{
    int eps = 0;
    eps |= (e_q[0] == e_qmax);
    eps |= (e_q[1] == e_qmax) << 1;
    eps |= (e_q[2] == e_qmax) << 2;
    eps |= (e_q[3] == e_qmax) << 3;
    return eps;
}

//************************************************************************/
/** @brief Makes sure the code-block data buffer holds at least size bytes
  *
  *  The buffer keeps the layout set up by the tile coder: one byte is
  *  reserved before cblk->data.
  */
static OPJ_BOOL opj_ht_enc_reserve(opj_tcd_cblk_enc_t* cblk, OPJ_UINT32 size)
{
    OPJ_BYTE* p;

    if (size <= cblk->data_size) {
        return OPJ_TRUE;
    }
    if (cblk->data) {
        p = (OPJ_BYTE*)opj_realloc(cblk->data - 1, (size_t)size + 1);
    } else {
        p = (OPJ_BYTE*)opj_malloc((size_t)size + 1);
        if (p) {
            p[0] = 0;
        }
    }
    if (p == NULL) {
        return OPJ_FALSE;
    }
    cblk->data = p + 1;
    cblk->data_size = size;
    return OPJ_TRUE;
}

//************************************************************************/
/** @brief Encodes one code-block with the HT block coder, as a single
  *         HT cleanup pass
  *
  *  @param [in, out] t1 holds the samples in t1->data, t1->w columns by
  *                   t1->h rows, row after row: bit 31 of each sample is its
  *                   sign and bits 30-0 its (quantised) magnitude
  *  @param [in, out] cblk receives the coded data in cblk->data and the
  *                   single coding pass in cblk->passes[0]
  *  @param [in]      max_mag is a bitwise OR of all magnitudes in t1->data;
  *                   it must be nonzero and smaller than 2^30
  *
  *  The magnitudes are coded in full (the cleanup pass covers bit-plane 0),
  *  so the number of missing MSBs of the code-block must be signalled as
  *  Mb - 1. cblk->numbps is set to 1 accordingly.
  *
  *  @return OPJ_FALSE if memory could not be allocated, or if the coded data
  *          do not fit the HT codestream syntax (which is not expected)
  */
OPJ_BOOL opj_t1_ht_encode_cblk(opj_t1_t *t1, opj_tcd_cblk_enc_t* cblk,
                               OPJ_UINT32 max_mag)
{
    OPJ_UINT8 mel_vlc_buf[OPJ_HT_MEL_SIZE + OPJ_HT_VLC_SIZE];
    opj_ht_mel_enc_t mel;
    opj_ht_vlc_enc_t vlc;
    opj_ht_ms_enc_t ms;
    const OPJ_UINT32 width = t1->w;
    const OPJ_UINT32 height = t1->h;
    const OPJ_UINT32 stride = t1->w;
    const OPJ_UINT32* buf = (const OPJ_UINT32*)t1->data;
    OPJ_UINT32 ms_size, num_bytes, length, e_bits;

    //e_val: E values for a line (these are the highest set bit)
    //cx_val: is the context values
    //Each byte stores the info for the 2 sample. For E, it is maximum
    // of the two samples, while for cx, it is the OR of these two samples.
    //The maximum is between the pixel at the bottom left of one quad
    // and the bottom right of the earlier quad. The same is true for cx.
    //For 1024 pixels, we need 512 bytes, plus one for the non-existing
    // earlier quad, and one for beyond the end; entries past the end of a
    // line are read, but only for a quad that does not exist, so they are
    // all cleared to keep the result deterministic
    OPJ_UINT8 e_val[516];
    OPJ_UINT8 cx_val[516];
    OPJ_UINT8* lep;
    OPJ_UINT8* lcxp;

    int e_qmax[2] = {0, 0}, e_q[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int rho[2] = {0, 0};
    int c_q0 = 0, c_q1;
    OPJ_UINT32 s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    OPJ_UINT32 x, y;
    const OPJ_UINT32* sp;
    int max_e;

    assert(width <= 1024 && height <= 1024 && width * height <= 4096);
    assert(max_mag != 0 && max_mag < (1U << 30));

    // Bound on the MagSgn bitstream: each sample needs at most e_bits bits
    // (the largest exponent) and each byte carries at least 7 bits; plus
    // one byte for the termination
    e_bits = 33U - opj_ht_enc_count_leading_zeros(max_mag);
    ms_size = (width * height * e_bits + 6U) / 7U + 1U;
    if (!opj_ht_enc_reserve(cblk, ms_size + OPJ_HT_MEL_SIZE
                            + OPJ_HT_VLC_SIZE)) {
        return OPJ_FALSE;
    }

    opj_ht_mel_init(&mel, OPJ_HT_MEL_SIZE, mel_vlc_buf);
    opj_ht_vlc_init(&vlc, OPJ_HT_VLC_SIZE, mel_vlc_buf + OPJ_HT_MEL_SIZE);
    opj_ht_ms_init(&ms, ms_size, cblk->data);

    memset(e_val, 0, sizeof(e_val));
    memset(cx_val, 0, sizeof(cx_val));
    lep = e_val;
    lcxp = cx_val;

    //initial row of quads
    sp = buf;
    for (x = 0; x < width; x += 4) {
        int Uq0, u_q0, u_q1, eps0, tuple0;

        //prepare two quads
        opj_ht_enc_sample(sp[0], 1, &rho[0], &e_q[0], &e_qmax[0], &s[0]);
        opj_ht_enc_sample(height > 1 ? sp[stride] : 0, 2, &rho[0], &e_q[1],
                          &e_qmax[0], &s[1]);
        ++sp;

        if (x + 1 < width) {
            opj_ht_enc_sample(sp[0], 4, &rho[0], &e_q[2], &e_qmax[0], &s[2]);
            opj_ht_enc_sample(height > 1 ? sp[stride] : 0, 8, &rho[0], &e_q[3],
                              &e_qmax[0], &s[3]);
            ++sp;
        }

        Uq0 = opj_int_max(e_qmax[0], 1); //kappa_q = 1
        u_q0 = Uq0 - 1;
        u_q1 = 0; //kappa_q = 1

        eps0 = u_q0 > 0 ? opj_ht_enc_eps(e_q, e_qmax[0]) : 0;
        lep[0] = (OPJ_UINT8)opj_int_max(lep[0], e_q[1]);
        lep++;
        lep[0] = (OPJ_UINT8)e_q[3];
        lcxp[0] = (OPJ_UINT8)(lcxp[0] | ((rho[0] & 2) >> 1));
        lcxp++;
        lcxp[0] = (OPJ_UINT8)((rho[0] & 8) >> 3);

        tuple0 = vlc_enc_tbl0[(c_q0 << 8) + (rho[0] << 4) + eps0];
        opj_ht_vlc_encode(&vlc, tuple0 >> 8, (tuple0 >> 4) & 7);

        if (c_q0 == 0) {
            opj_ht_mel_encode(&mel, rho[0] != 0);
        }

        opj_ht_enc_quad_magsgn(&ms, rho[0], Uq0, tuple0, s);

        if (x + 2 < width) {
            int Uq1, eps1, tuple1;

            opj_ht_enc_sample(sp[0], 1, &rho[1], &e_q[4], &e_qmax[1], &s[4]);
            opj_ht_enc_sample(height > 1 ? sp[stride] : 0, 2, &rho[1], &e_q[5],
                              &e_qmax[1], &s[5]);
            ++sp;

            if (x + 3 < width) {
                opj_ht_enc_sample(sp[0], 4, &rho[1], &e_q[6], &e_qmax[1], &s[6]);
                opj_ht_enc_sample(height > 1 ? sp[stride] : 0, 8, &rho[1],
                                  &e_q[7], &e_qmax[1], &s[7]);
                ++sp;
            }

            c_q1 = (rho[0] >> 1) | (rho[0] & 1);
            Uq1 = opj_int_max(e_qmax[1], 1); //kappa_q = 1
            u_q1 = Uq1 - 1; //kappa_q = 1

            eps1 = u_q1 > 0 ? opj_ht_enc_eps(e_q + 4, e_qmax[1]) : 0;
            lep[0] = (OPJ_UINT8)opj_int_max(lep[0], e_q[5]);
            lep++;
            lep[0] = (OPJ_UINT8)e_q[7];
            lcxp[0] = (OPJ_UINT8)(lcxp[0] | ((rho[1] & 2) >> 1));
            lcxp++;
            lcxp[0] = (OPJ_UINT8)((rho[1] & 8) >> 3);
            tuple1 = vlc_enc_tbl0[(c_q1 << 8) + (rho[1] << 4) + eps1];
            opj_ht_vlc_encode(&vlc, tuple1 >> 8, (tuple1 >> 4) & 7);

            if (c_q1 == 0) {
                opj_ht_mel_encode(&mel, rho[1] != 0);
            }

            opj_ht_enc_quad_magsgn(&ms, rho[1], Uq1, tuple1, s + 4);
        }

        if (u_q0 > 0 && u_q1 > 0) {
            opj_ht_mel_encode(&mel, opj_int_min(u_q0, u_q1) > 2);
        }

        if (u_q0 > 2 && u_q1 > 2) {
            const int e0 = uvlc_enc_tbl[u_q0 - 2], e1 = uvlc_enc_tbl[u_q1 - 2];
            opj_ht_vlc_encode(&vlc, OPJ_HT_UVLC_PRE(e0), OPJ_HT_UVLC_PRE_LEN(e0));
            opj_ht_vlc_encode(&vlc, OPJ_HT_UVLC_PRE(e1), OPJ_HT_UVLC_PRE_LEN(e1));
            opj_ht_vlc_encode(&vlc, OPJ_HT_UVLC_SUF(e0), OPJ_HT_UVLC_SUF_LEN(e0));
            opj_ht_vlc_encode(&vlc, OPJ_HT_UVLC_SUF(e1), OPJ_HT_UVLC_SUF_LEN(e1));
        } else if (u_q0 > 2 && u_q1 > 0) {
            const int e0 = uvlc_enc_tbl[u_q0];
            opj_ht_vlc_encode(&vlc, OPJ_HT_UVLC_PRE(e0), OPJ_HT_UVLC_PRE_LEN(e0));
            opj_ht_vlc_encode(&vlc, u_q1 - 1, 1);
            opj_ht_vlc_encode(&vlc, OPJ_HT_UVLC_SUF(e0), OPJ_HT_UVLC_SUF_LEN(e0));
        } else {
            const int e0 = uvlc_enc_tbl[u_q0], e1 = uvlc_enc_tbl[u_q1];
            opj_ht_vlc_encode(&vlc, OPJ_HT_UVLC_PRE(e0), OPJ_HT_UVLC_PRE_LEN(e0));
            opj_ht_vlc_encode(&vlc, OPJ_HT_UVLC_PRE(e1), OPJ_HT_UVLC_PRE_LEN(e1));
            opj_ht_vlc_encode(&vlc, OPJ_HT_UVLC_SUF(e0), OPJ_HT_UVLC_SUF_LEN(e0));
            opj_ht_vlc_encode(&vlc, OPJ_HT_UVLC_SUF(e1), OPJ_HT_UVLC_SUF_LEN(e1));
        }

        //prepare for next iteration
        c_q0 = (rho[1] >> 1) | (rho[1] & 1);
        s[0] = s[1] = s[2] = s[3] = s[4] = s[5] = s[6] = s[7] = 0;
        e_q[0] = e_q[1] = e_q[2] = e_q[3] = e_q[4] = e_q[5] = e_q[6] = e_q[7] = 0;
        rho[0] = rho[1] = 0;
        e_qmax[0] = e_qmax[1] = 0;
    }

    lep[1] = 0;

    //non-initial rows of quads
    for (y = 2; y < height; y += 2) {
        lep = e_val;
        max_e = opj_int_max(lep[0], lep[1]) - 1;
        lep[0] = 0;
        lcxp = cx_val;
        c_q0 = lcxp[0] + (lcxp[1] << 2);
        lcxp[0] = 0;

        sp = buf + y * stride;
        for (x = 0; x < width; x += 4) {
            int kappa, Uq0, u_q0, u_q1, eps0, tuple0;
            int e0, e1;

            //prepare two quads
            opj_ht_enc_sample(sp[0], 1, &rho[0], &e_q[0], &e_qmax[0], &s[0]);
            opj_ht_enc_sample(y + 1 < height ? sp[stride] : 0, 2, &rho[0],
                              &e_q[1], &e_qmax[0], &s[1]);
            ++sp;

            if (x + 1 < width) {
                opj_ht_enc_sample(sp[0], 4, &rho[0], &e_q[2], &e_qmax[0], &s[2]);
                opj_ht_enc_sample(y + 1 < height ? sp[stride] : 0, 8, &rho[0],
                                  &e_q[3], &e_qmax[0], &s[3]);
                ++sp;
            }

            kappa = (rho[0] & (rho[0] - 1)) ? opj_int_max(1, max_e) : 1;
            Uq0 = opj_int_max(e_qmax[0], kappa);
            u_q0 = Uq0 - kappa;
            u_q1 = 0;

            eps0 = u_q0 > 0 ? opj_ht_enc_eps(e_q, e_qmax[0]) : 0;
            lep[0] = (OPJ_UINT8)opj_int_max(lep[0], e_q[1]);
            lep++;
            max_e = opj_int_max(lep[0], lep[1]) - 1;
            lep[0] = (OPJ_UINT8)e_q[3];
            lcxp[0] = (OPJ_UINT8)(lcxp[0] | ((rho[0] & 2) >> 1));
            lcxp++;
            c_q1 = lcxp[0] + (lcxp[1] << 2);
            lcxp[0] = (OPJ_UINT8)((rho[0] & 8) >> 3);
            tuple0 = vlc_enc_tbl1[(c_q0 << 8) + (rho[0] << 4) + eps0];
            opj_ht_vlc_encode(&vlc, tuple0 >> 8, (tuple0 >> 4) & 7);

            if (c_q0 == 0) {
                opj_ht_mel_encode(&mel, rho[0] != 0);
            }

            opj_ht_enc_quad_magsgn(&ms, rho[0], Uq0, tuple0, s);

            if (x + 2 < width) {
                int Uq1, eps1, tuple1;

                opj_ht_enc_sample(sp[0], 1, &rho[1], &e_q[4], &e_qmax[1], &s[4]);
                opj_ht_enc_sample(y + 1 < height ? sp[stride] : 0, 2, &rho[1],
                                  &e_q[5], &e_qmax[1], &s[5]);
                ++sp;

                if (x + 3 < width) {
                    opj_ht_enc_sample(sp[0], 4, &rho[1], &e_q[6], &e_qmax[1],
                                      &s[6]);
                    opj_ht_enc_sample(y + 1 < height ? sp[stride] : 0, 8, &rho[1],
                                      &e_q[7], &e_qmax[1], &s[7]);
                    ++sp;
                }

                kappa = (rho[1] & (rho[1] - 1)) ? opj_int_max(1, max_e) : 1;
                c_q1 |= ((rho[0] & 4) >> 1) | ((rho[0] & 8) >> 2);
                Uq1 = opj_int_max(e_qmax[1], kappa);
                u_q1 = Uq1 - kappa;

                eps1 = u_q1 > 0 ? opj_ht_enc_eps(e_q + 4, e_qmax[1]) : 0;
                lep[0] = (OPJ_UINT8)opj_int_max(lep[0], e_q[5]);
                lep++;
                max_e = opj_int_max(lep[0], lep[1]) - 1;
                lep[0] = (OPJ_UINT8)e_q[7];
                lcxp[0] = (OPJ_UINT8)(lcxp[0] | ((rho[1] & 2) >> 1));
                lcxp++;
                c_q0 = lcxp[0] + (lcxp[1] << 2);
                lcxp[0] = (OPJ_UINT8)((rho[1] & 8) >> 3);
                tuple1 = vlc_enc_tbl1[(c_q1 << 8) + (rho[1] << 4) + eps1];
                opj_ht_vlc_encode(&vlc, tuple1 >> 8, (tuple1 >> 4) & 7);

                if (c_q1 == 0) {
                    opj_ht_mel_encode(&mel, rho[1] != 0);
                }

                opj_ht_enc_quad_magsgn(&ms, rho[1], Uq1, tuple1, s + 4);
            }

            e0 = uvlc_enc_tbl[u_q0];
            e1 = uvlc_enc_tbl[u_q1];
            opj_ht_vlc_encode(&vlc, OPJ_HT_UVLC_PRE(e0), OPJ_HT_UVLC_PRE_LEN(e0));
            opj_ht_vlc_encode(&vlc, OPJ_HT_UVLC_PRE(e1), OPJ_HT_UVLC_PRE_LEN(e1));
            opj_ht_vlc_encode(&vlc, OPJ_HT_UVLC_SUF(e0), OPJ_HT_UVLC_SUF_LEN(e0));
            opj_ht_vlc_encode(&vlc, OPJ_HT_UVLC_SUF(e1), OPJ_HT_UVLC_SUF_LEN(e1));

            //prepare for next iteration
            c_q0 |= ((rho[1] & 4) >> 1) | ((rho[1] & 8) >> 2);
            s[0] = s[1] = s[2] = s[3] = s[4] = s[5] = s[6] = s[7] = 0;
            e_q[0] = e_q[1] = e_q[2] = e_q[3] = e_q[4] = e_q[5] = e_q[6] = e_q[7] = 0;
            rho[0] = rho[1] = 0;
            e_qmax[0] = e_qmax[1] = 0;
        }
    }

    opj_ht_terminate_mel_vlc(&mel, &vlc);
    opj_ht_ms_terminate(&ms);

    num_bytes = mel.pos + vlc.pos; // Scup
    if (mel.overflow || vlc.overflow || ms.overflow || num_bytes < 2
            || num_bytes > 4079) {
        return OPJ_FALSE;
    }

    // the cleanup segment is MagSgn, then MEL, then VLC; MagSgn is already
    // in place
    length = ms.pos + num_bytes;
    memcpy(cblk->data + ms.pos, mel.buf, mel.pos);
    memcpy(cblk->data + ms.pos + mel.pos, vlc.buf - vlc.pos + 1, vlc.pos);

    // put in the interface locator word
    cblk->data[length - 1] = (OPJ_BYTE)(num_bytes >> 4);
    cblk->data[length - 2] = (OPJ_BYTE)((cblk->data[length - 2] & 0xF0)
                                        | (num_bytes & 0xF));

    cblk->numbps = 1;
    cblk->totalpasses = 1;
    cblk->passes[0].rate = length;
    cblk->passes[0].len = length;
    cblk->passes[0].term = 1;

    return OPJ_TRUE;
}
