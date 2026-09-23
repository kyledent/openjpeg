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

/* Differential test of the HT block decoder implementations: random
 * code-blocks (random bytes with a plausible Scup, random geometry, number
 * of passes, missing MSBs, Mb and vertical causality) are decoded with the
 * scalar decoder forced and with the default implementation (the SIMD one
 * when the CPU supports it), and the return values and samples must be
 * identical. Most random code-blocks are rejected; those that are accepted
 * exercise the cleanup, SigProp and MagRef passes on arbitrary bits.
 *
 * usage: test_ht_dec [iterations] [seed]
 */

#define OPJ_SKIP_POISON
#include "opj_includes.h"

#include <stdio.h>

OPJ_BOOL opj_t1_ht_decode_cblk(opj_t1_t *t1,
                               opj_tcd_cblk_dec_t* cblk,
                               OPJ_UINT32 orient,
                               OPJ_UINT32 roishift,
                               OPJ_UINT32 cblksty,
                               opj_event_mgr_t *p_manager,
                               opj_mutex_t* p_manager_mutex,
                               OPJ_BOOL check_pterm);

static OPJ_UINT64 rng_state = 0x243F6A8885A308D3ULL;

static OPJ_UINT32 rnd(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (OPJ_UINT32)(rng_state >> 11);
}

static OPJ_UINT32 rndn(OPJ_UINT32 n)
{
    return n ? rnd() % n : 0;
}

static void quiet_callback(const char *msg, void *client_data)
{
    (void)msg;
    (void)client_data;
}

int main(int argc, char** argv)
{
    long iters = argc > 1 ? atol(argv[1]) : 20000;
    long it, accepted = 0, rejected = 0, multi_pass = 0, failures = 0;
    opj_t1_t *t1_scalar = opj_t1_create(OPJ_FALSE);
    opj_t1_t *t1_default = opj_t1_create(OPJ_FALSE);
    opj_event_mgr_t mgr;
    OPJ_BYTE* data = (OPJ_BYTE*)opj_malloc(8192);

    if (!t1_scalar || !t1_default || !data) {
        return 1;
    }
    if (argc > 2) {
        rng_state ^= (OPJ_UINT64)atol(argv[2]) * 0x9E3779B97F4A7C15ULL;
    }
    memset(&mgr, 0, sizeof(mgr));
    mgr.error_handler = quiet_callback;
    mgr.warning_handler = quiet_callback;
    mgr.info_handler = quiet_callback;
    t1_scalar->ht_dec_impl = OPJ_HT_DEC_IMPL_SCALAR;

    for (it = 0; it < iters; ++it) {
        opj_tcd_cblk_dec_t cblk;
        opj_tcd_seg_t segs[2];
        opj_tcd_seg_data_chunk_t chunk;
        OPJ_UINT32 i, w, h, P, Mb, num_passes, len1, len2, scup;
        OPJ_UINT32 cblksty = J2K_CCP_CBLKSTY_HT;
        OPJ_BOOL r_scalar, r_default;

        /* a legal code-block geometry: w <= 2^a, h <= 2^b, a + b <= 12 */
        do {
            OPJ_UINT32 a = 2 + rndn(9), b = 2 + rndn(9);
            if (a + b > 12) {
                continue;
            }
            w = 1 + rndn(1u << a);
            h = 1 + rndn(1u << b);
            break;
        } while (1);
        /* many missing MSBs make random VLC data more often acceptable */
        P = rndn(2) ? rndn(31) : rndn(14);
        Mb = P + rndn(31 - P);
        if (Mb == 0) {
            Mb = 1;
        }
        num_passes = 1 + rndn(3);
        len1 = 2 + rndn(rndn(2) ? 48 : 2000);
        len2 = num_passes > 1 ? 1 + rndn(rndn(2) ? 32 : 1024) : 0;
        for (i = 0; i < len1 + len2; ++i) {
            data[i] = (OPJ_BYTE)rnd();
        }
        /* a plausible Scup, 2 <= Scup <= min(Lcup, 4079) */
        scup = 2 + rndn((len1 - 1 < 4078 ? len1 - 1 : 4078));
        data[len1 - 1] = (OPJ_BYTE)(scup >> 4);
        data[len1 - 2] = (OPJ_BYTE)((data[len1 - 2] & 0xF0) | (scup & 0xF));
        if (rndn(2)) {
            cblksty |= J2K_CCP_CBLKSTY_VSC;
        }

        memset(&cblk, 0, sizeof(cblk));
        memset(segs, 0, sizeof(segs));
        cblk.x1 = (OPJ_INT32)w;
        cblk.y1 = (OPJ_INT32)h;
        cblk.Mb = Mb;
        cblk.numbps = Mb + 1 - P;
        cblk.segs = segs;
        cblk.numsegs = num_passes > 1 ? 2 : 1;
        cblk.real_num_segs = cblk.numsegs;
        segs[0].real_num_passes = 1;
        segs[0].len = len1;
        segs[1].real_num_passes = num_passes - 1;
        segs[1].len = len2;
        chunk.data = data;
        chunk.len = len1 + len2;
        cblk.chunks = &chunk;
        cblk.numchunks = 1;

        r_scalar = opj_t1_ht_decode_cblk(t1_scalar, &cblk, 0, 0, cblksty, &mgr,
                                         NULL, OPJ_FALSE);
        r_default = opj_t1_ht_decode_cblk(t1_default, &cblk, 0, 0, cblksty, &mgr,
                                          NULL, OPJ_FALSE);
        if (r_scalar != r_default) {
            fprintf(stderr, "iteration %ld: return values differ (%d vs %d)\n",
                    it, r_scalar, r_default);
            ++failures;
        } else if (r_scalar) {
            ++accepted;
            multi_pass += num_passes > 1;
            if (memcmp(t1_scalar->data, t1_default->data,
                       (size_t)w * h * sizeof(OPJ_INT32)) != 0) {
                fprintf(stderr, "iteration %ld: samples differ (%ux%u, P=%u, "
                        "Mb=%u, %u passes)\n", it, w, h, P, Mb, num_passes);
                ++failures;
            }
        } else {
            ++rejected;
        }
    }

    printf("%ld code-blocks: %ld accepted (%ld with SigProp/MagRef), "
           "%ld rejected, %ld failures; default implementation %u\n",
           iters, accepted, multi_pass, rejected, failures,
           t1_default->ht_dec_impl);
    opj_free(data);
    opj_t1_destroy(t1_scalar);
    opj_t1_destroy(t1_default);
    return failures ? 1 : 0;
}
