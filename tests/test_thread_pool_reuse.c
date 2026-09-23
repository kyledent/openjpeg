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
 * Thread pools are reused across codecs (opj_thread_pool_acquire/release).
 * This checks what reuse could break, on POSIX systems:
 *  - encoders and decoders alternating at the same thread count (worker
 *    thread local storage holds per-direction Tier-1 state), and several
 *    thread counts interleaved: every round trip must be exact;
 *  - repeated codecs must not accumulate threads (Linux: /proc/self/task);
 *  - after fork(), the child must still be able to encode and decode with
 *    threads (the parent's idle workers do not exist in the child).
 */
#define _DEFAULT_SOURCE
#include <dirent.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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
    m->pos += (size_t)n;
    return n;
}

static OPJ_BOOL mb_seek(OPJ_OFF_T n, void *u)
{
    ((membuf *)u)->pos = (size_t)n;
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

/* Encode a w x h 16-bit signed noise image with `threads`, decode it with
 * `threads`, return the number of differing samples (or -1 on failure). */
static int round_trip(int w, int h, int threads, unsigned seed, int mode)
{
    opj_image_cmptparm_t cp;
    opj_image_t *img, *out = NULL;
    opj_cparameters_t p;
    opj_dparameters_t dp;
    opj_codec_t *c;
    opj_stream_t *s;
    membuf m;
    OPJ_INT32 *ref;
    int i, ok, bad = 0;

    memset(&cp, 0, sizeof(cp));
    cp.dx = cp.dy = 1;
    cp.w = (OPJ_UINT32)w;
    cp.h = (OPJ_UINT32)h;
    cp.prec = 16;
    cp.sgnd = 1;
    img = opj_image_create(1, &cp, OPJ_CLRSPC_GRAY);
    ref = (OPJ_INT32 *)malloc((size_t)w * (size_t)h * sizeof(OPJ_INT32));
    if (!img || !ref) {
        return -1;
    }
    img->x1 = (OPJ_UINT32)w;
    img->y1 = (OPJ_UINT32)h;
    for (i = 0; i < w * h; i++) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        ref[i] = img->comps[0].data[i] = (OPJ_INT32)(seed % 60000u) - 30000;
    }

    opj_set_default_encoder_parameters(&p);
    p.tcp_numlayers = 1;
    p.tcp_rates[0] = 0;
    p.cp_disto_alloc = 1;
    p.mode = mode;
    c = opj_create_compress(OPJ_CODEC_J2K);
    opj_set_error_handler(c, msg, (void *)"encoder error");
    memset(&m, 0, sizeof(m));
    ok = opj_setup_encoder(c, &p, img) && opj_codec_set_threads(c, threads);
    if (ok) {
        s = mb_stream(&m, OPJ_FALSE);
        ok = opj_start_compress(c, img, s) && opj_encode(c, s) &&
             opj_end_compress(c, s);
        opj_stream_destroy(s);
    }
    opj_destroy_codec(c);
    opj_image_destroy(img);
    if (!ok) {
        free(ref);
        free(m.buf);
        return -1;
    }

    opj_set_default_decoder_parameters(&dp);
    c = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_error_handler(c, msg, (void *)"decoder error");
    opj_setup_decoder(c, &dp);
    opj_codec_set_threads(c, threads);
    m.pos = 0;
    s = mb_stream(&m, OPJ_TRUE);
    ok = opj_read_header(s, c, &out) && opj_decode(c, s, out) &&
         opj_end_decompress(c, s);
    opj_stream_destroy(s);
    opj_destroy_codec(c);
    if (ok) {
        for (i = 0; i < w * h; i++) {
            bad += out->comps[0].data[i] != ref[i];
        }
    }
    opj_image_destroy(out);
    free(ref);
    free(m.buf);
    return ok ? bad : -1;
}

static int count_threads(void)
{
    int n = 0;
    struct dirent *e;
    DIR *d = opendir("/proc/self/task");
    if (!d) {
        return -1;
    }
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] != '.') {
            n++;
        }
    }
    closedir(d);
    return n;
}

int main(void)
{
    static const int counts[] = {4, 4, 2, 4, 3, 2, 4, 3};
    int it, r, n_threads, peak = 0, failures = 0;
    pid_t pid;

    if (!opj_has_thread_support()) {
        printf("no thread support: nothing to test\n");
        return 0;
    }

    /* Alternating encoders/decoders, several pool sizes, BYPASS and plain */
    for (it = 0; it < 24; it++) {
        const int t = counts[it % 8];
        r = round_trip(384 + it, 256 + 3 * it, t, 12345u + (unsigned)it,
                       (it & 1) ? 1 : 0);
        if (r != 0) {
            fprintf(stderr, "iteration %d (threads %d): %d samples differ\n", it,
                    t, r);
            failures++;
        }
        n_threads = count_threads();
        if (n_threads > peak) {
            peak = n_threads;
        }
    }
    /* 3 pool sizes x 2 directions, at most 4 threads each, plus the main
     * thread: anything above that would mean pools are not being reused. */
    printf("thread count after the loop: %d (peak %d)\n", count_threads(), peak);
    if (peak > 1 + 2 * (4 + 3 + 2)) {
        fprintf(stderr, "too many threads: pools are accumulating\n");
        failures++;
    }

    /* A forked child must not be handed the parent's idle pools */
    pid = fork();
    if (pid == 0) {
        alarm(60); /* a deadlock kills the child instead of hanging the test */
        r = round_trip(300, 200, 4, 777u, 0);
        _exit(r == 0 ? 0 : 1);
    } else if (pid > 0) {
        int status = 0;
        if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) ||
                WEXITSTATUS(status) != 0) {
            fprintf(stderr, "forked child failed (status 0x%x)\n", status);
            failures++;
        } else {
            printf("forked child encoded and decoded with 4 threads\n");
        }
    } else {
        perror("fork");
        failures++;
    }
    /* and the parent keeps working after the fork */
    r = round_trip(300, 200, 4, 999u, 0);
    if (r != 0) {
        fprintf(stderr, "parent after fork: %d\n", r);
        failures++;
    }

    printf("%s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
