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
#define _MM512_BROADCASTD_EPI32(addr, zmm)             \
                    __asm__ ("vpbroadcastd (%1), %0;"  \
                            : "=v" (zmm)               \
                            : "r"  (addr) )

#define _VFMADD231_PS_MUL_ADDR(ymm1, ymm2, addr)          \
                    __asm__ ("vfmadd231ps %0, %1, (%2);"  \
                            : "=v" (ymm1)                 \
                            : "v"  (ymm2), "r" (addr) )

#define _VFMADD213_PS_ADD_ADDR(ymm1, ymm2, addr)          \
                    __asm__ ("vfmadd213ps %0, %1, (%2);"  \
                            : "=v" (ymm1), "v" (ymm2)     \
                            : "r"  (addr) )               

// Define micro kernels for ALPHA not ONE && BETA as ONE scenarios
static int sgemv_kernel_n(BLASLONG m, BLASLONG n, float alpha, float *a, BLASLONG lda, float *x, float *y)
{
    //printf("----- enter into kernel\n");
    __m256 ma0, ma1, ma2, ma3, ma4, ma5, ma6, ma7;
    __m256 as0, as1, as2, as3, as4, as5, as6, as7;
    __m256 alphav = _mm256_set1_ps(alpha);
    __m256 xv;
    BLASLONG tag_m_64x = m & (~63);
    BLASLONG tag_m_32x = m & (~31);
    BLASLONG tag_m_16x = m & (~15);
    BLASLONG tag_m_8x = m & (~7);
    //unsigned char one_mask_uint = (unsigned char)0xff;
    __mmask8 one_mask = 0xff;

    for (BLASLONG idx_m = 0; idx_m < tag_m_64x; idx_m+=64) {
        //printf("enter into kernel 64\n");
        as0 = _mm256_setzero_ps();
        as1 = _mm256_setzero_ps();
        as2 = _mm256_setzero_ps();
        as3 = _mm256_setzero_ps();
        as4 = _mm256_setzero_ps();
        as5 = _mm256_setzero_ps();
        as6 = _mm256_setzero_ps();
        as7 = _mm256_setzero_ps();

        for (BLASLONG idx_n = 0; idx_n < n; idx_n++) {
            xv = _mm256_set1_ps(x[idx_n]);
            ma0 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m +0]);
            ma1 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m +8]);
            ma2 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m +16]);
            ma3 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m +24]);
            ma4 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m +32]);
            ma5 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m +40]);
            ma6 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m +48]);
            ma7 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m +56]);

            as0 = _mm256_maskz_fmadd_ps(one_mask, ma0, xv, as0);
            as1 = _mm256_maskz_fmadd_ps(one_mask, ma1, xv, as1);
            as2 = _mm256_maskz_fmadd_ps(one_mask, ma2, xv, as2);
            as3 = _mm256_maskz_fmadd_ps(one_mask, ma3, xv, as3);
            as4 = _mm256_maskz_fmadd_ps(one_mask, ma4, xv, as4);
            as5 = _mm256_maskz_fmadd_ps(one_mask, ma5, xv, as5);
            as6 = _mm256_maskz_fmadd_ps(one_mask, ma6, xv, as6);
            as7 = _mm256_maskz_fmadd_ps(one_mask, ma7, xv, as7);
                    
        }
        _mm256_mask_storeu_ps(&y[idx_m], one_mask, _mm256_maskz_fmadd_ps(one_mask, as0, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m])));
        _mm256_mask_storeu_ps(&y[idx_m + 8], one_mask, _mm256_maskz_fmadd_ps(one_mask, as1, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m + 8])));
        _mm256_mask_storeu_ps(&y[idx_m + 16], one_mask, _mm256_maskz_fmadd_ps(one_mask, as2, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m + 16])));
        _mm256_mask_storeu_ps(&y[idx_m + 24], one_mask, _mm256_maskz_fmadd_ps(one_mask, as3, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m + 24])));
        _mm256_mask_storeu_ps(&y[idx_m + 32], one_mask, _mm256_maskz_fmadd_ps(one_mask, as4, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m + 32])));
        _mm256_mask_storeu_ps(&y[idx_m + 40], one_mask, _mm256_maskz_fmadd_ps(one_mask, as5, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m + 40])));
        _mm256_mask_storeu_ps(&y[idx_m + 48], one_mask, _mm256_maskz_fmadd_ps(one_mask, as6, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m + 48])));
        _mm256_mask_storeu_ps(&y[idx_m + 56], one_mask, _mm256_maskz_fmadd_ps(one_mask, as7, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m + 56])));
    }

    if (tag_m_64x != m){
    for (BLASLONG idx_m = tag_m_64x; idx_m < tag_m_32x; idx_m+=32) {
    //for (BLASLONG idx_m = 0; idx_m < tag_m_32x; idx_m+=32) {
        as0 = _mm256_setzero_ps();
        as1 = _mm256_setzero_ps();
        as2 = _mm256_setzero_ps();
        as3 = _mm256_setzero_ps();

        for (BLASLONG idx_n = 0; idx_n < n; idx_n++) {
            xv = _mm256_set1_ps(x[idx_n]);
            ma0 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m +0]);
            ma1 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m +8]);
            ma2 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m +16]);
            ma3 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m +24]);

            as0 = _mm256_maskz_fmadd_ps(one_mask, ma0, xv, as0);
            as1 = _mm256_maskz_fmadd_ps(one_mask, ma1, xv, as1);
            as2 = _mm256_maskz_fmadd_ps(one_mask, ma2, xv, as2);
            as3 = _mm256_maskz_fmadd_ps(one_mask, ma3, xv, as3);
        }
        _mm256_mask_storeu_ps(&y[idx_m], one_mask, _mm256_maskz_fmadd_ps(one_mask, as0, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m])));
        _mm256_mask_storeu_ps(&y[idx_m + 8], one_mask, _mm256_maskz_fmadd_ps(one_mask, as1, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m + 8])));
        _mm256_mask_storeu_ps(&y[idx_m + 16], one_mask, _mm256_maskz_fmadd_ps(one_mask, as2, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m + 16])));
        _mm256_mask_storeu_ps(&y[idx_m + 24], one_mask, _mm256_maskz_fmadd_ps(one_mask, as3, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m + 24])));
 
    }    
 
    if (tag_m_32x != m ) {
            for (BLASLONG idx_m = tag_m_32x; idx_m < tag_m_16x; idx_m+=16) {
            as4 = _mm256_setzero_ps();
            as5 = _mm256_setzero_ps();
    
            for (BLASLONG idx_n = 0; idx_n < n; idx_n++) {
                xv = _mm256_set1_ps(x[idx_n]);
                ma4 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m +0]);
                ma5 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m +8]);
    
                as4 = _mm256_maskz_fmadd_ps(one_mask, ma4, xv, as4);
                as5 = _mm256_maskz_fmadd_ps(one_mask, ma5, xv, as5);
            }
            _mm256_mask_storeu_ps(&y[idx_m], one_mask, _mm256_maskz_fmadd_ps(one_mask, as4, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m])));
            _mm256_mask_storeu_ps(&y[idx_m + 8], one_mask, _mm256_maskz_fmadd_ps(one_mask, as5, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m + 8])));
        }
    
        if (tag_m_16x != m ) {
            for (BLASLONG idx_m = tag_m_16x; idx_m < tag_m_8x; idx_m+=8) {
                //printf("enter into kernel 8 \n");
                as6 = _mm256_setzero_ps();
    
                for (BLASLONG idx_n = 0; idx_n < n; idx_n++) {
                    xv = _mm256_set1_ps(x[idx_n]);
                    ma6 = _mm256_maskz_loadu_ps(one_mask, &a[idx_n * lda + idx_m]);
    
                    as6 = _mm256_maskz_fmadd_ps(one_mask, ma6, xv, as6);
                }
                _mm256_mask_storeu_ps(&y[idx_m], one_mask, _mm256_maskz_fmadd_ps(one_mask, as6, alphav, _mm256_maskz_loadu_ps(one_mask, &y[idx_m])));
            }
        
            if (tag_m_8x != m) {
                //printf("enter into kernel tail \n");
                as7 = _mm256_setzero_ps();
    
                unsigned char tail_mask_uint = (((unsigned char)0xff) >> (8-(m&7)));
                __mmask8 tail_mask = *((__mmask8*) &tail_mask_uint);
    
                for(BLASLONG idx_n = 0; idx_n < n; idx_n++) {
                    xv = _mm256_set1_ps(x[idx_n]);
                    ma7 = _mm256_maskz_loadu_ps(tail_mask, &a[idx_n * lda + tag_m_8x]);
    
                    as7 = _mm256_maskz_fmadd_ps(tail_mask, ma7, xv, as7);
                }
    
                _mm256_mask_storeu_ps(&y[tag_m_8x], tail_mask, _mm256_maskz_fmadd_ps(tail_mask, as7, alphav, _mm256_maskz_loadu_ps(tail_mask, &y[tag_m_8x])));
    
            }
        }
    }
    }
    //printf("-----end of misc\n");
    return 0;
}


#endif