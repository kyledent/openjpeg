/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OPJ_HT_DEC_SIMD_H
#define OPJ_HT_DEC_SIMD_H

/**
@file ht_dec_simd.h
@brief Internal interface between the HTJ2K block decoder front end
(ht_dec.c) and its SIMD back ends (ht_dec_avx2.c).

The front end validates the code-block exactly as the scalar decoder does
and then hands the three HT segments to a SIMD back end, which decodes them
into a padded, aligned sign-magnitude buffer of the workspace and writes the
final two's complement samples to the Tier-1 buffer. The back end must
produce bit-exactly the samples the scalar decoder produces, and must reject
a code-block whenever the scalar decoder rejects it. When the back end
rejects a code-block, the front end decodes it again with the scalar
decoder, which reports the error with its own message.
*/

/** Number of 16-bit quad records in the workspace scratch buffer: one
    quad row needs at most 1024 + 9 entries and a code-block has at most
    (4096 / w + 1) / 2 + 2 quad rows. 4800 covers every legal geometry. */
#define OPJ_HT_SIMD_SCRATCH_SIZE    4800
/** Samples in the decoded-sample buffer: stride rounded up to 8 columns,
    height rounded up to 4 rows, for any w * h <= 4096, w, h <= 1024 */
#define OPJ_HT_SIMD_OUT_SIZE        8192
/** Maximum number of de-stuffed MagSgn bytes the decoder can consume:
    4096 samples of at most 32 bits each */
#define OPJ_HT_SIMD_MS_CAP          (4096 * 32 / 8)
/** VLC segment (at most 4079 bytes) */
#define OPJ_HT_SIMD_VLC_CAP         4096
/** SPP segment: at most one significance and one sign bit per sample */
#define OPJ_HT_SIMD_SPP_CAP         (4096 * 2 / 8)
/** MRP segment: at most one bit per sample */
#define OPJ_HT_SIMD_MRP_CAP         1024
/** De-stuffing writes the data, one extra byte and 64 bytes of padding;
    fetches read up to 24 bytes past the clamp offset */
#define OPJ_HT_SIMD_PAD             72
/** Entries of the v_n (exponent) scratch rows */
#define OPJ_HT_SIMD_VN_SIZE         (512 + 16)

/** Workspace of the SIMD HT block decoders. One per opj_t1_t, allocated
    (32-byte aligned) on first use; nothing in it is shared between
    threads. */
typedef struct opj_ht_dec_simd_ws {
    /** decoded samples, sign-magnitude, stride of (w + 7) & ~7 */
    OPJ_UINT32 out[OPJ_HT_SIMD_OUT_SIZE];
    /** v_n scratch: two rows of 32-bit entries */
    OPJ_UINT32 vn32[2 * OPJ_HT_SIMD_VN_SIZE];
    /** v_n scratch: one row of 16-bit entries (16-bit path) */
    OPJ_UINT16 vn16[OPJ_HT_SIMD_VN_SIZE + 16];
    /** quad records: decoded VLC codeword and u_q, interleaved */
    OPJ_UINT16 scratch[OPJ_HT_SIMD_SCRATCH_SIZE];
    /** wide UVLC table for non-initial quad rows (see opj_ht_dec_avx2_init) */
    OPJ_UINT32 uvlc_tbl1_wide[4096];
    /** UVLC table for the initial quad row */
    OPJ_UINT16 uvlc_tbl0[256 + 64];
    /** UVLC table for non-initial quad rows */
    OPJ_UINT16 uvlc_tbl1[256];
    /** de-stuffed MagSgn segment */
    OPJ_UINT8 ms_buf[OPJ_HT_SIMD_MS_CAP + OPJ_HT_SIMD_PAD];
    /** de-stuffed VLC segment */
    OPJ_UINT8 vlc_buf[OPJ_HT_SIMD_VLC_CAP + OPJ_HT_SIMD_PAD];
    /** de-stuffed SPP segment */
    OPJ_UINT8 spp_buf[OPJ_HT_SIMD_SPP_CAP + OPJ_HT_SIMD_PAD];
    /** de-stuffed MRP segment */
    OPJ_UINT8 mrp_buf[OPJ_HT_SIMD_MRP_CAP + OPJ_HT_SIMD_PAD];
} opj_ht_dec_simd_ws_t;

/** Result codes of a SIMD back end, one per check of the scalar decoder
    that the back end performs itself */
typedef enum {
    OPJ_HT_SIMD_OK = 0,
    /** mel_init found a stuffing violation */
    OPJ_HT_SIMD_ERR_MEL,
    /** U_q larger than zero bitplanes + 1 in the initial quad row */
    OPJ_HT_SIMD_ERR_UQ_INITIAL,
    /** U_q larger than zero bitplanes + 1 in a non-initial quad row */
    OPJ_HT_SIMD_ERR_UQ,
    /** a VLC codeword marks a sample outside the code-block significant */
    OPJ_HT_SIMD_ERR_OUTSIDE
} opj_ht_simd_result_t;

/** Parameters of one code-block, already validated by the front end */
typedef struct opj_ht_simd_cblk {
    const OPJ_UINT8* coded_data; /**< cleanup segment, then SPP/MRP segment */
    OPJ_UINT32 zero_bplanes;     /**< missing MSBs (P) */
    OPJ_UINT32 numbps;           /**< Mb + 1 - P, >= 1 */
    OPJ_UINT32 Mb;               /**< Mb, <= 30 */
    OPJ_UINT32 num_passes;       /**< 1, 2 or 3 */
    OPJ_UINT32 lengths1;         /**< cleanup segment length (>= 2) */
    OPJ_UINT32 lengths2;         /**< SPP + MRP segment length */
    int lcup;                    /**< lengths1 */
    int scup;                    /**< MEL + VLC length, 2..min(lcup, 4079) */
    OPJ_UINT32 width;            /**< 1..1024 */
    OPJ_UINT32 height;           /**< 1..1024, width * height <= 4096 */
    OPJ_BOOL stripe_causal;      /**< vertically causal context */
} opj_ht_simd_cblk_t;

/**
Initialise the decoding tables of a workspace for the AVX2 decoder.
Only call when the CPU supports AVX2.
@param ws       workspace
*/
void opj_ht_dec_avx2_init(opj_ht_dec_simd_ws_t* ws);

/**
Decode one HT code-block with AVX2 and write the samples, as two's
complement integers with a stride of width, to dst.
Only call when the CPU supports AVX2.
@param ws       workspace, with the UVLC tables initialised
@param cb       the validated code-block parameters
@param dst      destination, width * height samples
@return OPJ_HT_SIMD_OK or an error code (dst is then undefined)
*/
opj_ht_simd_result_t opj_ht_dec_cblk_avx2(opj_ht_dec_simd_ws_t* ws,
        const opj_ht_simd_cblk_t* cb, OPJ_INT32* dst);

#endif /* OPJ_HT_DEC_SIMD_H */
