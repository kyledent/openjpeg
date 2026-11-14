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
 * In-memory round trip: encode a deterministic noisy image with the given
 * options, decode it, and require every sample back exactly.
 *
 *   test_encode_roundtrip [-w W] [-h H] [-prec P] [-sgnd 0|1] [-comps N]
 *                         [-mct 0|1] [-mode M] [-threads N] [-dec-threads N]
 *                         [-res R] [-cblk C] [-cblkw W] [-cblkh H] [-tile T]
 *                         [-prc P] [-layers L] [-rate R] [-guard G] [-roi]
 *                         [-irrev] [-qstep Q] [-qstep-check default|smaller]
 *                         [-jp2] [-checker K] [-expect-reject]
 *                         [-o FILE] [-i FILE] [-dump-src FILE]
 *
 * -threads / -dec-threads call opj_codec_set_threads() with the value given,
 * including 0 and 1, so both the inline and the one-worker paths are covered.
 * -expect-reject passes only if the encoder refuses the parameters.
 * -checker K replaces the image with a checkerboard of K x K squares of the
 * extreme sample values, whose wavelet coefficients exceed the nominal range
 * of their subband (e.g. for K = 2), so -guard 0 makes it unrepresentable.
 *
 * -irrev uses the irreversible 9/7 transform, which is not lossless: the
 * image decoded must then be identical to the one decoded from the same
 * parameters with mode 0, since both the MQ and the HT coder code the same
 * quantisation indices in full.
 *
 * -qstep Q sets the QSTEP option (the base quantisation step of the
 * irreversible transform, a fraction of the sample range), which is how HT
 * code-blocks are coded lossily; the decode must still equal mode 0's. With
 * -qstep-check default the codestream must be byte-identical to one encoded
 * without QSTEP (pass Q = 2^-prec); with -qstep-check smaller it must be
 * smaller than that one.
 *
 * The markers are checked too: Rsiz bit 14 and a CAP marker right after SIZ
 * must be present exactly when the mode has the HT bit (0x40), and, with
 * -jp2, the file brand must be "jph " instead of "jp2 ".
 *
 * For cross-checks with other implementations: -o saves the codestream,
 * -dump-src saves component 0 of the source image as raw little-endian
 * samples of (prec + 7) / 8 bytes, and -i decodes the given file instead of
 * encoding, comparing it with the image the other options describe.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openjpeg.h"

#define MAX_COMPS 4

typedef struct {
    unsigned char *buf;
    size_t len, cap, pos;
} membuf;

typedef struct {
    int w, h, prec, sgnd, comps, mct, mode, threads, dec_threads, res;
    int cblkw, cblkh, tile, prc, layers, guard, roi, irrev, jp2, checker;
    float rate;
    double qstep;
} options;

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

/* A smooth ramp plus uniform noise: every bit-plane gets used, and the
 * wavelet has structure to find. Component c gets its own noise. Or, with
 * -checker, a checkerboard of the extreme values. */
static opj_image_t *make_image(const options *o, OPJ_INT32 **ref)
{
    opj_image_cmptparm_t cp[MAX_COMPS];
    opj_image_t *img;
    long long span = 1LL << o->prec;
    long long lo = o->sgnd ? -(span / 2) : 0;
    int c, i;

    memset(cp, 0, sizeof(cp));
    for (c = 0; c < o->comps; c++) {
        cp[c].dx = cp[c].dy = 1;
        cp[c].w = (OPJ_UINT32)o->w;
        cp[c].h = (OPJ_UINT32)o->h;
        cp[c].prec = (OPJ_UINT32)o->prec;
        cp[c].sgnd = (OPJ_UINT32)o->sgnd;
    }
    img = opj_image_create((OPJ_UINT32)o->comps, cp,
                           o->comps == 3 ? OPJ_CLRSPC_SRGB : OPJ_CLRSPC_GRAY);
    if (!img) {
        return NULL;
    }
    img->x1 = (OPJ_UINT32)o->w;
    img->y1 = (OPJ_UINT32)o->h;
    for (c = 0; c < o->comps; c++) {
        unsigned seed = 2463534242u + 0x9E3779B9u * (unsigned)c;
        for (i = 0; i < o->w * o->h; i++) {
            long long ramp = (long long)(i % o->w + i / o->w) * (span / 4) /
                             (o->w + o->h);
            long long v;
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            v = lo + ramp + (long long)(seed % (unsigned long long)(span / 2));
            if (v > lo + span - 1) {
                v = lo + span - 1;
            }
            if (o->checker > 0) {
                v = ((i % o->w) / o->checker + (i / o->w) / o->checker) % 2 ?
                    lo + span - 1 : lo;
            }
            img->comps[c].data[i] = (OPJ_INT32)v;
            if (ref) {
                ref[c][i] = (OPJ_INT32)v;
            }
        }
    }
    return img;
}

/* Encodes the image the options describe with the given mode into m;
 * returns 1 on success. The encoder takes the image's sample buffers. */
static int encode(const options *o, int mode, membuf *m, OPJ_BOOL report)
{
    opj_image_t *img = make_image(o, NULL);
    opj_cparameters_t p;
    opj_codec_t *c;
    opj_stream_t *s;
    char guard_opt[32], qstep_opt[64];
    const char *extra[3];
    int i, n_extra = 0, ok;

    if (!img) {
        return 0;
    }
    opj_set_default_encoder_parameters(&p);
    p.tcp_numlayers = o->layers;
    for (i = 0; i < o->layers; i++) {
        p.tcp_rates[i] = 0;
    }
    p.tcp_rates[0] = o->rate;
    p.cp_disto_alloc = 1;
    p.numresolution = o->res;
    p.cblockw_init = o->cblkw;
    p.cblockh_init = o->cblkh;
    p.mode = mode;
    p.tcp_mct = (char)o->mct;
    p.irreversible = o->irrev;
    if (o->tile > 0) {
        p.tile_size_on = OPJ_TRUE;
        p.cp_tdx = o->tile;
        p.cp_tdy = o->tile;
    }
    if (o->prc > 0) {
        p.csty |= 0x01;
        p.res_spec = 1;
        p.prcw_init[0] = o->prc;
        p.prch_init[0] = o->prc;
    }
    if (o->roi) {
        p.roi_compno = 0;
        p.roi_shift = 2;
    }

    c = opj_create_compress(o->jp2 ? OPJ_CODEC_JP2 : OPJ_CODEC_J2K);
    opj_set_info_handler(c, quiet, NULL);
    opj_set_warning_handler(c, msg, (void *)"encoder warning");
    opj_set_error_handler(c, report ? msg : quiet, (void *)"encoder error");
    ok = opj_setup_encoder(c, &p, img);
    if (o->guard >= 0) {
        snprintf(guard_opt, sizeof(guard_opt), "GUARD_BITS=%d", o->guard);
        extra[n_extra++] = guard_opt;
    }
    if (o->qstep != 0.0) {
        snprintf(qstep_opt, sizeof(qstep_opt), "QSTEP=%.17g", o->qstep);
        extra[n_extra++] = qstep_opt;
    }
    extra[n_extra] = NULL;
    if (ok && n_extra > 0) {
        ok = opj_encoder_set_extra_options(c, extra);
    }
    if (ok && o->threads >= 0) {
        ok = opj_codec_set_threads(c, o->threads);
    }
    if (ok) {
        s = mb_stream(m, OPJ_FALSE);
        ok = opj_start_compress(c, img, s) && opj_encode(c, s) &&
             opj_end_compress(c, s);
        opj_stream_destroy(s);
    }
    opj_destroy_codec(c);
    opj_image_destroy(img);
    return ok;
}

static opj_image_t *decode(const options *o, membuf *m)
{
    opj_dparameters_t dp;
    opj_codec_t *c;
    opj_stream_t *s;
    opj_image_t *out = NULL;
    int ok;

    opj_set_default_decoder_parameters(&dp);
    c = opj_create_decompress(o->jp2 ? OPJ_CODEC_JP2 : OPJ_CODEC_J2K);
    opj_set_info_handler(c, quiet, NULL);
    opj_set_warning_handler(c, msg, (void *)"decoder warning");
    opj_set_error_handler(c, msg, (void *)"decoder error");
    opj_setup_decoder(c, &dp);
    if (o->dec_threads >= 0) {
        opj_codec_set_threads(c, o->dec_threads);
    }
    m->pos = 0;
    s = mb_stream(m, OPJ_TRUE);
    ok = opj_read_header(s, c, &out) && opj_decode(c, s, out) &&
         opj_end_decompress(c, s);
    opj_stream_destroy(s);
    opj_destroy_codec(c);
    if (!ok) {
        opj_image_destroy(out);
        return NULL;
    }
    return out;
}

static unsigned rd16(const unsigned char *b)
{
    return ((unsigned)b[0] << 8) | b[1];
}

/* Checks the file brand and the SIZ / CAP markers; returns 0 if they are
 * right for the code-block style used. */
static int check_markers(const options *o, const membuf *m)
{
    const int ht = (o->mode & 0x40) != 0;
    const unsigned char *b = m->buf;
    size_t i, siz_end;

    if (o->jp2) {
        /* the signature box (12 bytes), then the file type box: length,
         * "ftyp", brand, minor version, compatibility list */
        const char *brand = ht ? "jph " : "jp2 ";
        if (m->len < 32 || memcmp(b + 16, "ftyp", 4) != 0 ||
                memcmp(b + 20, brand, 4) != 0 || memcmp(b + 28, brand, 4) != 0) {
            fprintf(stderr, "file type box without the brand \"%s\"\n", brand);
            return 1;
        }
    }
    for (i = 0; i + 8 <= m->len; i++) {
        if (b[i] == 0xFF && b[i + 1] == 0x4F && b[i + 2] == 0xFF &&
                b[i + 3] == 0x51) {
            break;
        }
    }
    if (i + 8 > m->len) {
        fprintf(stderr, "no SOC + SIZ found\n");
        return 1;
    }
    if (((rd16(b + i + 6) & 0x4000) != 0) != ht) {
        fprintf(stderr, "Rsiz 0x%04x: bit 14 should be %d\n", rd16(b + i + 6), ht);
        return 1;
    }
    siz_end = i + 4 + rd16(b + i + 4);
    if (siz_end + 10 > m->len) {
        fprintf(stderr, "truncated main header\n");
        return 1;
    }
    if (ht) {
        if (rd16(b + siz_end) != 0xFF50 || rd16(b + siz_end + 2) != 8 ||
                rd16(b + siz_end + 4) != 0x0002 || rd16(b + siz_end + 6) != 0) {
            fprintf(stderr, "no CAP marker with Pcap^15 after SIZ\n");
            return 1;
        }
    } else if (rd16(b + siz_end) == 0xFF50) {
        fprintf(stderr, "unexpected CAP marker\n");
        return 1;
    }
    return 0;
}

static int write_file(const char *path, const void *data, size_t len)
{
    FILE *f = fopen(path, "wb");
    int ok = f && fwrite(data, 1, len, f) == len;
    if (f && fclose(f) != 0) {
        ok = 0;
    }
    if (!ok) {
        fprintf(stderr, "cannot write %s\n", path);
    }
    return ok;
}

static int read_file(const char *path, membuf *m)
{
    FILE *f = fopen(path, "rb");
    long n;
    int ok = 0;
    if (f && fseek(f, 0, SEEK_END) == 0 && (n = ftell(f)) > 0 &&
            fseek(f, 0, SEEK_SET) == 0) {
        m->buf = (unsigned char *)malloc((size_t)n);
        ok = m->buf && fread(m->buf, 1, (size_t)n, f) == (size_t)n;
        m->len = m->cap = (size_t)n;
    }
    if (f) {
        fclose(f);
    }
    if (!ok) {
        fprintf(stderr, "cannot read %s\n", path);
    }
    return ok;
}

/* Component 0 as raw little-endian samples of (prec + 7) / 8 bytes */
static int dump_source(const options *o, const OPJ_INT32 *ref, const char *path)
{
    const size_t bps = (size_t)(o->prec + 7) / 8, n = (size_t)o->w * (size_t)o->h;
    unsigned char *raw = (unsigned char *)malloc(n * bps);
    size_t i, k;
    int ok;
    if (!raw) {
        return 0;
    }
    for (i = 0; i < n; i++) {
        for (k = 0; k < bps; k++) {
            raw[i * bps + k] = (unsigned char)((OPJ_UINT32)ref[i] >> (8 * k));
        }
    }
    ok = write_file(path, raw, n * bps);
    free(raw);
    return ok;
}

/* Number of samples of out that differ from ref (or -1 if the geometry is
 * wrong); *maxerr receives the largest absolute difference. */
static long compare(const options *o, const opj_image_t *out,
                    OPJ_INT32 *const *ref, long long *maxerr)
{
    long bad = 0;
    int c, i;
    *maxerr = 0;
    if ((int)out->numcomps != o->comps) {
        fprintf(stderr, "decoded %u components, expected %d\n", out->numcomps,
                o->comps);
        return -1;
    }
    for (c = 0; c < o->comps; c++) {
        if ((int)out->comps[c].w != o->w || (int)out->comps[c].h != o->h) {
            fprintf(stderr, "decoded %ux%u, expected %dx%d\n", out->comps[c].w,
                    out->comps[c].h, o->w, o->h);
            return -1;
        }
        for (i = 0; i < o->w * o->h; i++) {
            long long d = (long long)out->comps[c].data[i] - ref[c][i];
            if (d != 0) {
                bad++;
                if (d < 0) {
                    d = -d;
                }
                if (d > *maxerr) {
                    *maxerr = d;
                }
            }
        }
    }
    return bad;
}

int main(int argc, char **argv)
{
    options o;
    int expect_reject = 0, i, c, ok, ret = 1;
    const char *out_path = NULL, *in_path = NULL, *src_path = NULL;
    const char *qstep_check = NULL;
    OPJ_INT32 *ref[MAX_COMPS] = { NULL, NULL, NULL, NULL };
    opj_image_t *img, *out = NULL;
    membuf m;
    long bad;
    long long maxerr = 0;

    memset(&o, 0, sizeof(o));
    o.w = 300;
    o.h = 200;
    o.prec = 12;
    o.comps = 1;
    o.threads = -1;
    o.dec_threads = -1;
    o.res = 6;
    o.cblkw = o.cblkh = 64;
    o.layers = 1;
    o.guard = -1;
    memset(&m, 0, sizeof(m));

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            o.w = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-h") && i + 1 < argc) {
            o.h = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-prec") && i + 1 < argc) {
            o.prec = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-sgnd") && i + 1 < argc) {
            o.sgnd = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-comps") && i + 1 < argc) {
            o.comps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-mct") && i + 1 < argc) {
            o.mct = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-mode") && i + 1 < argc) {
            o.mode = (int)strtol(argv[++i], NULL, 0);
        } else if (!strcmp(argv[i], "-threads") && i + 1 < argc) {
            o.threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-dec-threads") && i + 1 < argc) {
            o.dec_threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-res") && i + 1 < argc) {
            o.res = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-cblk") && i + 1 < argc) {
            o.cblkw = o.cblkh = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-cblkw") && i + 1 < argc) {
            o.cblkw = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-cblkh") && i + 1 < argc) {
            o.cblkh = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-tile") && i + 1 < argc) {
            o.tile = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-prc") && i + 1 < argc) {
            o.prc = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-layers") && i + 1 < argc) {
            o.layers = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-rate") && i + 1 < argc) {
            o.rate = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "-guard") && i + 1 < argc) {
            o.guard = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-roi")) {
            o.roi = 1;
        } else if (!strcmp(argv[i], "-irrev")) {
            o.irrev = 1;
        } else if (!strcmp(argv[i], "-qstep") && i + 1 < argc) {
            o.qstep = atof(argv[++i]);
        } else if (!strcmp(argv[i], "-qstep-check") && i + 1 < argc) {
            qstep_check = argv[++i];
        } else if (!strcmp(argv[i], "-jp2")) {
            o.jp2 = 1;
        } else if (!strcmp(argv[i], "-checker") && i + 1 < argc) {
            o.checker = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-expect-reject")) {
            expect_reject = 1;
        } else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            in_path = argv[++i];
        } else if (!strcmp(argv[i], "-dump-src") && i + 1 < argc) {
            src_path = argv[++i];
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    if (o.comps < 1 || o.comps > MAX_COMPS || o.prec < 1 || o.prec > 30 ||
            o.w < 1 || o.h < 1 || o.layers < 1 || o.layers > 100) {
        fprintf(stderr, "unsupported image or layer parameters\n");
        return 2;
    }

    for (c = 0; c < o.comps; c++) {
        ref[c] = (OPJ_INT32 *)malloc((size_t)o.w * (size_t)o.h * sizeof(OPJ_INT32));
        if (!ref[c]) {
            goto end;
        }
    }
    img = make_image(&o, ref);
    if (!img) {
        goto end;
    }
    opj_image_destroy(img);
    if (src_path && !dump_source(&o, ref[0], src_path)) {
        goto end;
    }

    if (in_path) {
        /* decode a file written by another encoder */
        if (!read_file(in_path, &m)) {
            goto end;
        }
    } else {
        ok = encode(&o, o.mode, &m, !expect_reject);
        if (expect_reject) {
            if (ok) {
                fprintf(stderr, "mode 0x%x: encoder ACCEPTED parameters it should "
                        "have refused\n", o.mode);
                goto end;
            }
            printf("mode 0x%x: rejected, as expected\n", o.mode);
            ret = 0;
            goto end;
        }
        if (!ok) {
            fprintf(stderr, "encode failed\n");
            goto end;
        }
        if (out_path && !write_file(out_path, m.buf, m.len)) {
            goto end;
        }
        if (check_markers(&o, &m)) {
            goto end;
        }
        if (qstep_check) {
            options plain = o;
            membuf ref_m;
            int same;
            memset(&ref_m, 0, sizeof(ref_m));
            plain.qstep = 0.0;
            if (!encode(&plain, o.mode, &ref_m, OPJ_TRUE)) {
                fprintf(stderr, "encode without QSTEP failed\n");
                free(ref_m.buf);
                goto end;
            }
            same = m.len == ref_m.len && !memcmp(m.buf, ref_m.buf, m.len);
            printf("QSTEP=%.17g: %zu bytes, without it %zu bytes%s\n", o.qstep, m.len,
                   ref_m.len, same ? " (identical)" : "");
            free(ref_m.buf);
            if (!strcmp(qstep_check, "default") ? !same :
                    !strcmp(qstep_check, "smaller") ? !(m.len < ref_m.len) : 1) {
                fprintf(stderr, "-qstep-check %s failed\n", qstep_check);
                goto end;
            }
        }
    }

    out = decode(&o, &m);
    if (!out) {
        fprintf(stderr, "decode failed\n");
        goto end;
    }
    bad = compare(&o, out, ref, &maxerr);
    if (bad < 0) {
        goto end;
    }
    if (o.irrev && !in_path) {
        /* lossy: compare with the decoded MQ coded codestream instead */
        membuf mq;
        opj_image_t *out_mq;
        long long maxerr_mq;
        memset(&mq, 0, sizeof(mq));
        if (!encode(&o, 0, &mq, OPJ_TRUE) || !(out_mq = decode(&o, &mq))) {
            fprintf(stderr, "MQ reference failed\n");
            free(mq.buf);
            goto end;
        }
        for (c = 0; c < o.comps; c++) {
            memcpy(ref[c], out_mq->comps[c].data,
                   (size_t)o.w * (size_t)o.h * sizeof(OPJ_INT32));
        }
        opj_image_destroy(out_mq);
        free(mq.buf);
        printf("lossy: largest error %lld on %ld samples; ", maxerr, bad);
        bad = compare(&o, out, ref, &maxerr_mq);
        if (bad < 0) {
            goto end;
        }
        printf("against mode 0: ");
    }
    printf("%dx%dx%d prec %d%s mode 0x%x threads %d/%d: %zu bytes, %ld samples differ\n",
           o.w, o.h, o.comps, o.prec, o.sgnd ? "s" : "u", o.mode, o.threads,
           o.dec_threads, m.len, bad);
    ret = bad ? 1 : 0;

end:
    opj_image_destroy(out);
    for (c = 0; c < MAX_COMPS; c++) {
        free(ref[c]);
    }
    free(m.buf);
    return ret;
}
