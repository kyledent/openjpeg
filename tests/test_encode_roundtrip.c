/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2026, Diamond Light Source
 * All rights reserved.
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

/*
 * In-memory lossless round trip: encode a deterministic noisy image with the
 * given options, decode it, and require every sample back exactly.
 *
 *   test_encode_roundtrip [-w W] [-h H] [-prec P] [-sgnd 0|1] [-mode M]
 *                         [-threads N] [-dec-threads N] [-res R] [-cblk C]
 *                         [-layers L] [-jp2] [-expect-reject]
 *
 * -threads / -dec-threads call opj_codec_set_threads() with the value given,
 * including 0 and 1, so both the inline and the one-worker paths are covered.
 * -expect-reject passes only if the encoder refuses the parameters.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openjpeg.h"

typedef struct {
    unsigned char *buf;
    size_t len, cap, pos;
} membuf;

static OPJ_SIZE_T mb_write(void *p, OPJ_SIZE_T n, void *u)
{
    membuf *m = (membuf *)u;
    if (m->pos + n > m->cap) {
        size_t nc = m->cap ? m->cap : 65536;
        unsigned char *nb;
        while (nc < m->pos + n) {
            nc *= 2;
        }
        nb = (unsigned char *)realloc(m->buf, nc);
        if (!nb) {
            return (OPJ_SIZE_T) - 1;
        }
        m->buf = nb;
        m->cap = nc;
    }
    memcpy(m->buf + m->pos, p, n);
    m->pos += n;
    if (m->pos > m->len) {
        m->len = m->pos;
    }
    return n;
}

static OPJ_SIZE_T mb_read(void *p, OPJ_SIZE_T n, void *u)
{
    membuf *m = (membuf *)u;
    if (m->pos >= m->len) {
        return (OPJ_SIZE_T) - 1;
    }
    if (n > m->len - m->pos) {
        n = m->len - m->pos;
    }
    memcpy(p, m->buf + m->pos, n);
    m->pos += n;
    return n;
}

static OPJ_OFF_T mb_skip(OPJ_OFF_T n, void *u)
{
    membuf *m = (membuf *)u;
    if (n < 0 && (size_t)(-n) > m->pos) {
        n = -(OPJ_OFF_T)m->pos;
    }
    m->pos += (size_t)n;
    return n;
}

static OPJ_BOOL mb_seek(OPJ_OFF_T n, void *u)
{
    membuf *m = (membuf *)u;
    if (n < 0) {
        return OPJ_FALSE;
    }
    m->pos = (size_t)n;
    return OPJ_TRUE;
}

static opj_stream_t *mb_stream(membuf *m, OPJ_BOOL input)
{
    opj_stream_t *s = opj_stream_create(1 << 16, input);
    opj_stream_set_user_data(s, m, NULL);
    opj_stream_set_user_data_length(s, input ? m->len : 0);
    opj_stream_set_read_function(s, mb_read);
    opj_stream_set_write_function(s, mb_write);
    opj_stream_set_skip_function(s, mb_skip);
    opj_stream_set_seek_function(s, mb_seek);
    return s;
}

static void msg(const char *t, void *u)
{
    fprintf(stderr, "[%s] %s", (const char *)u, t);
}

static void quiet(const char *t, void *u)
{
    (void)t;
    (void)u;
}

int main(int argc, char **argv)
{
    int w = 300, h = 200, prec = 12, sgnd = 0, mode = 0, threads = -1;
    int dec_threads = -1, res = 6, cblk = 64, layers = 1, jp2 = 0;
    int expect_reject = 0, i, ok, bad = 0;
    unsigned seed = 2463534242u;
    OPJ_INT32 *ref;
    opj_image_cmptparm_t cp;
    opj_image_t *img, *out = NULL;
    opj_cparameters_t p;
    opj_dparameters_t dp;
    opj_codec_t *c;
    opj_stream_t *s;
    membuf m;
    OPJ_INT32 lo, span;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            w = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-h") && i + 1 < argc) {
            h = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-prec") && i + 1 < argc) {
            prec = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-sgnd") && i + 1 < argc) {
            sgnd = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-mode") && i + 1 < argc) {
            mode = (int)strtol(argv[++i], NULL, 0);
        } else if (!strcmp(argv[i], "-threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-dec-threads") && i + 1 < argc) {
            dec_threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-res") && i + 1 < argc) {
            res = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-cblk") && i + 1 < argc) {
            cblk = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-layers") && i + 1 < argc) {
            layers = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-jp2")) {
            jp2 = 1;
        } else if (!strcmp(argv[i], "-expect-reject")) {
            expect_reject = 1;
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }

    memset(&cp, 0, sizeof(cp));
    cp.dx = cp.dy = 1;
    cp.w = (OPJ_UINT32)w;
    cp.h = (OPJ_UINT32)h;
    cp.prec = (OPJ_UINT32)prec;
    cp.sgnd = (OPJ_UINT32)sgnd;
    img = opj_image_create(1, &cp, OPJ_CLRSPC_GRAY);
    if (!img) {
        return 1;
    }
    img->x1 = (OPJ_UINT32)w;
    img->y1 = (OPJ_UINT32)h;

    /* A smooth ramp plus uniform noise: every bit-plane gets used, and the
     * wavelet has structure to find. The encoder takes ownership of the
     * image's sample buffer, so the reference is a separate copy. */
    ref = (OPJ_INT32 *)malloc((size_t)w * (size_t)h * sizeof(OPJ_INT32));
    if (!ref) {
        return 1;
    }
    span = (OPJ_INT32)1 << prec;
    lo = sgnd ? -(span / 2) : 0;
    for (i = 0; i < w * h; i++) {
        OPJ_INT32 ramp = (OPJ_INT32)((long long)(i % w + i / w) * (span / 4) /
                                     (w + h));
        OPJ_INT32 v;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        v = lo + ramp + (OPJ_INT32)(seed % (unsigned)(span / 2));
        if (v > lo + span - 1) {
            v = lo + span - 1;
        }
        ref[i] = img->comps[0].data[i] = v;
    }

    opj_set_default_encoder_parameters(&p);
    p.tcp_numlayers = layers;
    for (i = 0; i < layers; i++) {
        p.tcp_rates[i] = 0;
    }
    p.cp_disto_alloc = 1;
    p.numresolution = res;
    p.cblockw_init = cblk;
    p.cblockh_init = cblk;
    p.mode = mode;

    c = opj_create_compress(jp2 ? OPJ_CODEC_JP2 : OPJ_CODEC_J2K);
    opj_set_info_handler(c, quiet, NULL);
    opj_set_warning_handler(c, msg, (void *)"encoder warning");
    opj_set_error_handler(c, expect_reject ? quiet : msg, (void *)"encoder error");
    memset(&m, 0, sizeof(m));
    ok = opj_setup_encoder(c, &p, img);
    if (ok && threads >= 0) {
        ok = opj_codec_set_threads(c, threads);
    }
    if (ok) {
        s = mb_stream(&m, OPJ_FALSE);
        ok = opj_start_compress(c, img, s) && opj_encode(c, s) &&
             opj_end_compress(c, s);
        opj_stream_destroy(s);
    }
    opj_destroy_codec(c);
    opj_image_destroy(img);

    if (expect_reject) {
        free(ref);
        free(m.buf);
        if (ok) {
            fprintf(stderr, "mode 0x%x: encoder ACCEPTED parameters it should "
                    "have refused\n", mode);
            return 1;
        }
        printf("mode 0x%x: rejected, as expected\n", mode);
        return 0;
    }
    if (!ok) {
        fprintf(stderr, "encode failed\n");
        return 1;
    }

    opj_set_default_decoder_parameters(&dp);
    c = opj_create_decompress(jp2 ? OPJ_CODEC_JP2 : OPJ_CODEC_J2K);
    opj_set_info_handler(c, quiet, NULL);
    opj_set_warning_handler(c, msg, (void *)"decoder warning");
    opj_set_error_handler(c, msg, (void *)"decoder error");
    opj_setup_decoder(c, &dp);
    if (dec_threads >= 0) {
        opj_codec_set_threads(c, dec_threads);
    }
    m.pos = 0;
    s = mb_stream(&m, OPJ_TRUE);
    ok = opj_read_header(s, c, &out) && opj_decode(c, s, out) &&
         opj_end_decompress(c, s);
    opj_stream_destroy(s);
    opj_destroy_codec(c);
    if (!ok) {
        fprintf(stderr, "decode failed\n");
        opj_image_destroy(out);
        free(ref);
        free(m.buf);
        return 1;
    }
    if ((int)out->comps[0].w != w || (int)out->comps[0].h != h) {
        fprintf(stderr, "decoded %ux%u, expected %dx%d\n", out->comps[0].w,
                out->comps[0].h, w, h);
        bad = 1;
    } else {
        for (i = 0; i < w * h; i++) {
            bad += out->comps[0].data[i] != ref[i];
        }
    }
    printf("%dx%d prec %d%s mode 0x%x threads %d/%d: %zu bytes, %d samples differ\n",
           w, h, prec, sgnd ? "s" : "u", mode, threads, dec_threads, m.len, bad);
    opj_image_destroy(out);
    free(ref);
    free(m.buf);
    return bad ? 1 : 0;
}
