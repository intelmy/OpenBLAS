/***************************************************************************
Copyright (c) 2014, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

/* need a new enough GCC for avx512 support */
#if (( defined(__GNUC__)  && __GNUC__   >= 6 && defined(__AVX512CD__)) || (defined(__clang__) && __clang_major__ >= 6))

#define HAVE_SGEMV_N_SKYLAKE_KERNEL 1
#include "common.h"
#include <immintrin.h>

static int sgemv_kernel_small_n(BLASLONG m, BLASLONG n, float alpha, float *a, BLASLONG lda, float *x, float *y)
{
    __m512 matrixArray_0, matrixArray_1, matrixArray_2, matrixArray_3, matrixArray_4, matrixArray_5, matrixArray_6, matrixArray_7;
    __m512 accum512_0, accum512_1, accum512_2, accum512_3, accum512_4, accum512_5, accum512_6, accum512_7;
    __m512 xArray_0;
    __m512  ALPHAVECTOR = _mm512_set1_ps(alpha);
    BLASLONG tag_m_128x = m & (~127);
    BLASLONG tag_m_64x = m & (~63);
    BLASLONG tag_m_32x = m & (~31);
    BLASLONG tag_m_16x = m & (~15);

    for (BLASLONG idx_m = 0; idx_m < tag_m_128x; idx_m+=128) {
        accum512_0 = _mm512_setzero_ps();
        accum512_1 = _mm512_setzero_ps();
        accum512_2 = _mm512_setzero_ps();
        accum512_3 = _mm512_setzero_ps();
        accum512_4 = _mm512_setzero_ps();
        accum512_5 = _mm512_setzero_ps();
        accum512_6 = _mm512_setzero_ps();
        accum512_7 = _mm512_setzero_ps();

        for (BLASLONG idx_n = 0; idx_n < n; idx_n++) {
            xArray_0 = _mm512_set1_ps(x[idx_n]);

            matrixArray_0 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 0]);
            matrixArray_1 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 16]);
            matrixArray_2 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 32]);
            matrixArray_3 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 48]);
            matrixArray_4 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 64]);
            matrixArray_5 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 80]);
            matrixArray_6 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 96]);
            matrixArray_7 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 112]);

            accum512_0 = _mm512_fmadd_ps(matrixArray_0, xArray_0, accum512_0);
            accum512_1 = _mm512_fmadd_ps(matrixArray_1, xArray_0, accum512_1);
            accum512_2 = _mm512_fmadd_ps(matrixArray_2, xArray_0, accum512_2);
            accum512_3 = _mm512_fmadd_ps(matrixArray_3, xArray_0, accum512_3);
            accum512_4 = _mm512_fmadd_ps(matrixArray_4, xArray_0, accum512_4);
            accum512_5 = _mm512_fmadd_ps(matrixArray_5, xArray_0, accum512_5);
            accum512_6 = _mm512_fmadd_ps(matrixArray_6, xArray_0, accum512_6);
            accum512_7 = _mm512_fmadd_ps(matrixArray_7, xArray_0, accum512_7);
        }

        _mm512_storeu_ps(&y[idx_m + 0], _mm512_fmadd_ps(accum512_0, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 0])));
        _mm512_storeu_ps(&y[idx_m + 16], _mm512_fmadd_ps(accum512_1, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 16])));
        _mm512_storeu_ps(&y[idx_m + 32], _mm512_fmadd_ps(accum512_2, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 32])));
        _mm512_storeu_ps(&y[idx_m + 48], _mm512_fmadd_ps(accum512_3, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 48])));
        _mm512_storeu_ps(&y[idx_m + 64], _mm512_fmadd_ps(accum512_4, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 64])));
        _mm512_storeu_ps(&y[idx_m + 80], _mm512_fmadd_ps(accum512_5, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 80])));
        _mm512_storeu_ps(&y[idx_m + 96], _mm512_fmadd_ps(accum512_6, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 96])));
        _mm512_storeu_ps(&y[idx_m + 112], _mm512_fmadd_ps(accum512_7, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 112])));
    }
    for (BLASLONG idx_m = tag_m_128x; idx_m < tag_m_64x; idx_m+=64) {
        accum512_0 = _mm512_setzero_ps();
        accum512_1 = _mm512_setzero_ps();
        accum512_2 = _mm512_setzero_ps();
        accum512_3 = _mm512_setzero_ps();

        for (BLASLONG idx_n = 0; idx_n < n; idx_n++) {
            xArray_0 = _mm512_set1_ps(x[idx_n]);

            matrixArray_0 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 0]);
            matrixArray_1 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 16]);
            matrixArray_2 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 32]);
            matrixArray_3 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 48]);

            accum512_0 = _mm512_fmadd_ps(matrixArray_0, xArray_0, accum512_0);
            accum512_1 = _mm512_fmadd_ps(matrixArray_1, xArray_0, accum512_1);
            accum512_2 = _mm512_fmadd_ps(matrixArray_2, xArray_0, accum512_2);
            accum512_3 = _mm512_fmadd_ps(matrixArray_3, xArray_0, accum512_3);
        }

        _mm512_storeu_ps(&y[idx_m + 0], _mm512_fmadd_ps(accum512_0, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 0])));
        _mm512_storeu_ps(&y[idx_m + 16], _mm512_fmadd_ps(accum512_1, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 16])));
        _mm512_storeu_ps(&y[idx_m + 32], _mm512_fmadd_ps(accum512_2, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 32])));
        _mm512_storeu_ps(&y[idx_m + 48], _mm512_fmadd_ps(accum512_3, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 48])));
    }
    for (BLASLONG idx_m = tag_m_64x; idx_m < tag_m_32x; idx_m+=32) {
        accum512_0 = _mm512_setzero_ps();
        accum512_1 = _mm512_setzero_ps();

        for (BLASLONG idx_n = 0; idx_n < n; idx_n++) {
            xArray_0 = _mm512_set1_ps(x[idx_n]);

            matrixArray_0 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 0]);
            matrixArray_1 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 16]);

            accum512_0 = _mm512_fmadd_ps(matrixArray_0, xArray_0, accum512_0);
            accum512_1 = _mm512_fmadd_ps(matrixArray_1, xArray_0, accum512_1);
        }

        _mm512_storeu_ps(&y[idx_m + 0], _mm512_fmadd_ps(accum512_0, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 0])));
        _mm512_storeu_ps(&y[idx_m + 16], _mm512_fmadd_ps(accum512_1, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 16])));
    }    

    for (BLASLONG idx_m = tag_m_32x; idx_m < tag_m_16x; idx_m+=16) {
        accum512_0 = _mm512_setzero_ps();

        for (BLASLONG idx_n = 0; idx_n < n; idx_n++) {
            xArray_0 = _mm512_set1_ps(x[idx_n]);

            matrixArray_0 = _mm512_loadu_ps(&a[idx_n * lda + idx_m + 0]);

            accum512_0 = _mm512_fmadd_ps(matrixArray_0, xArray_0, accum512_0);
        }

        _mm512_storeu_ps(&y[idx_m + 0], _mm512_fmadd_ps(accum512_0, ALPHAVECTOR, _mm512_loadu_ps(&y[idx_m + 0])));
    }       

    if (tag_m_16x != m) {
        accum512_0 = _mm512_setzero_ps();

        unsigned short tail_mask_value = (((unsigned int)0xffff) >> (16-(m&15)));
        __mmask16 tail_mask = *((__mmask16*) &tail_mask_value);

        for(BLASLONG idx_n = 0; idx_n < n; idx_n++) {
            xArray_0 = _mm512_set1_ps(x[idx_n]);
            matrixArray_0 = _mm512_maskz_loadu_ps(tail_mask, &a[idx_n * lda + tag_m_16x]);

            accum512_0 = _mm512_fmadd_ps(matrixArray_0, xArray_0, accum512_0);
        }

        _mm512_mask_storeu_ps(&y[tag_m_16x], tail_mask, _mm512_fmadd_ps(accum512_0, ALPHAVECTOR, _mm512_maskz_loadu_ps(tail_mask, &y[tag_m_16x])));

    }
    return 0;
}

#endif
