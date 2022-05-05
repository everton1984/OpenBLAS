/***************************************************************************
Copyright (c) 2021, The OpenBLAS Project
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
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A00 PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

#include "common.h"
#include <arm_neon.h>

#define DECLR_ACC1(N,M) \
  float64x2_t acc##N##_##M = vdupq_n_f64(0);

#define DECLR_ACC2(N) \
  DECLR_ACC1(N,0) \
  DECLR_ACC1(N,1)

#define DECLR_ACC4(N) \
  DECLR_ACC1(N,0) \
  DECLR_ACC1(N,1) \
  DECLR_ACC1(N,2) \
  DECLR_ACC1(N,3)

#define DECLR_A1(N) \
  float64x2_t ra##N;

#define DECLR_B1_(N) \
  float64x2_t rb##N; float64x2_t pAlpha = vdupq_n_f64(alpha);

#define DECLR_B1(N) \
  float64x2_t rb##N;

#define DECLR_B2(N) \
  DECLR_B1(1); \
  DECLR_B1(2);

#define DECLR_A2() \
  DECLR_A1(1) \
  DECLR_A1(2)

#define DECLR_A4() \
  DECLR_A1(1) \
  DECLR_A1(2) \
  DECLR_A1(3) \
  DECLR_A1(4)

#define LOADA1(N) \
  ra##N = vld1q_f64(ptrba + ((N)-1)*2);

#define LOADA1_(N,K) \
  ra##N = vld1q_f64(ptrba + ((K)-1)*2);

#define LOADA2() \
  LOADA1(1) \
  LOADA1(2)

#define LOADA4() \
  LOADA1(1) \
  LOADA1(2) \
  LOADA1(3) \
  LOADA1(4)

#define LOADA4_(K) \
  LOADA1_(1,K*4 + 1) \
  LOADA1_(2,K*4 + 2) \
  LOADA1_(3,K*4 + 3) \
  LOADA1_(4,K*4 + 4)

#define LOADB1(N, M) \
  rb##N = vdupq_n_f64(*(ptrbb + (M)));

#define KERNEL1(N, M, M1) \
  acc##N##_##M = vfmaq_f64(acc##N##_##M , ra##M1 , rb1);

#define KERNEL2x1(N) \
  KERNEL1(N, 0, 1); \
  KERNEL1(N, 1, 2);

#define KERNEL4x1(N) \
  KERNEL1(N, 0, 1); \
  KERNEL1(N, 1, 2); \
  KERNEL1(N, 2, 3); \
  KERNEL1(N, 3, 4);

#define STR_C1(N, M, I) \
  C##N[I+0] += alpha*vgetq_lane_f64(acc##N##_##M, 0); \
  C##N[I+1] += alpha*vgetq_lane_f64(acc##N##_##M, 1);

#define STR_C1_(N, M, I) {\
  float64x2_t pC = vld1q_f64(C##N + I); \
  pC = vfmaq_f64(pC, pAlpha, acc##N##_##M); \
  vst1q_f64(C##N + I, pC); \
  }

#define STR_C2(N) \
  STR_C1(N, 0, 0); \
  STR_C1(N, 1, 2);

#define STR_C4(N) \
  STR_C1(N, 0, 0); \
  STR_C1(N, 1, 2); \
  STR_C1(N, 2, 4); \
  STR_C1(N, 3, 6);

#define MKERNEL_(K)   \
  LOADA4_(K);       \
  LOADB1(1,K*4 + 0);    \
  KERNEL4x1(0);   \
  LOADB1(1,K*4 + 1);    \
  KERNEL4x1(1);   \
  LOADB1(1,K*4 + 2);    \
  KERNEL4x1(2);   \
  LOADB1(1,K*4 + 3);    \
  KERNEL4x1(3);

#define MKERNEL()   \
  LOADA4();       \
  LOADB1(1,0);    \
  KERNEL4x1(0);   \
  LOADB1(1,1);    \
  KERNEL4x1(1);   \
  LOADB1(1,2);    \
  KERNEL4x1(2);   \
  LOADB1(1,3);    \
  KERNEL4x1(3);   \
  ptrba += 8;     \
  ptrbb += 4;

/*inline*/ void kernel_8x4(BLASLONG K, FLOAT alpha,
  FLOAT *ptrba, FLOAT *ptrbb, FLOAT *C0, FLOAT *C1, FLOAT *C2, FLOAT *C3)
  {
    DECLR_ACC4(0);
    DECLR_ACC4(1);
    DECLR_ACC4(2);
    DECLR_ACC4(3);
    
    DECLR_A4();
    DECLR_B1(1);

    BLASLONG k = 0;
    for(; k + 8 <= K; k+=8)
    {
      MKERNEL_(0);
      MKERNEL_(1);
      MKERNEL_(2);
      MKERNEL_(3);
      MKERNEL_(4);
      MKERNEL_(5);
      MKERNEL_(6);
      MKERNEL_(7);
      ptrba += 8*8;
      ptrbb += 4*8;
    }
    // __builtin_prefetch(ptrba + 8*0, 0, 1);
    for(; k + 4 <= K; k+=4)
    {
      MKERNEL_(0);
      MKERNEL_(1);
      // __builtin_prefetch(ptrba + 8*2, 0, 1);
      MKERNEL_(2);
      MKERNEL_(3);
      // __builtin_prefetch(ptrba + 8*4, 0, 1);
      ptrba += 8*4;
      ptrbb += 4*4;
    }
    for(; k < K; k++)
    {
      MKERNEL();
    }

    // __builtin_prefetch(C0, 1, 0);
    STR_C4(0);
    STR_C4(1);
    // __builtin_prefetch(C2, 1, 0);
    STR_C4(2);
    STR_C4(3);
  }

/*inline*/ void kernel_8x2(BLASLONG K, FLOAT alpha,
  FLOAT *ptrba, FLOAT *ptrbb, FLOAT *C0, FLOAT *C1)
  {
    DECLR_ACC4(0);
    DECLR_ACC4(1);

    DECLR_A4();
    DECLR_B1(1);
    for(BLASLONG k = 0; k < K; k++)
    {
      LOADA4();

      LOADB1(1,0);
      KERNEL4x1(0);
      LOADB1(1,1);
      KERNEL4x1(1);

      ptrba += 8;
      ptrbb += 2;
    }

    STR_C4(0);
    STR_C4(1);
  }

/*inline*/ void kernel_8x1(BLASLONG K, FLOAT alpha,
  FLOAT *ptrba, FLOAT *ptrbb, FLOAT *C0)
  {
    DECLR_ACC4(0);

    DECLR_A4();
    DECLR_B1(1);
    for(BLASLONG k = 0; k < K; k++)
    {
      LOADA4();

      LOADB1(1,0);
      KERNEL4x1(0);

      ptrba += 8;
      ptrbb += 1;
    }

    STR_C4(0);
  }

/*inline*/ void kernel_4x4(BLASLONG K, FLOAT alpha,
  FLOAT *ptrba, FLOAT *ptrbb, FLOAT *C0, FLOAT *C1, FLOAT *C2, FLOAT *C3)
  {
    DECLR_ACC2(0);
    DECLR_ACC2(1);
    DECLR_ACC2(2);
    DECLR_ACC2(3);

    DECLR_A2();
    DECLR_B1(1);
    for(BLASLONG k = 0; k < K; k++)
    {
      LOADA2();

      LOADB1(1,0);
      KERNEL2x1(0);
      LOADB1(1,1);
      KERNEL2x1(1);
      LOADB1(1,2);
      KERNEL2x1(2);
      LOADB1(1,3);
      KERNEL2x1(3);

      ptrba += 4;
      ptrbb += 4;
    }

    STR_C2(0);
    STR_C2(1);
    STR_C2(2);
    STR_C2(3);
  }

/*inline*/ void kernel_4x2(BLASLONG K, FLOAT alpha,
  FLOAT *ptrba, FLOAT *ptrbb, FLOAT *C0, FLOAT *C1)
  {
    DECLR_ACC2(0);
    DECLR_ACC2(1);

    DECLR_A2();
    DECLR_B1(1);
    for(BLASLONG k = 0; k < K; k++)
    {
      LOADA2();

      LOADB1(1,0);
      KERNEL2x1(0);
      LOADB1(1,1);
      KERNEL2x1(1);

      ptrba += 4;
      ptrbb += 2;
    }

    STR_C2(0);
    STR_C2(1);
  }

/*inline*/ void kernel_2x4(BLASLONG K, FLOAT alpha,
  FLOAT *ptrba, FLOAT *ptrbb, FLOAT *C0, FLOAT *C1, FLOAT *C2, FLOAT *C3)
  {
    DECLR_ACC1(0,0);
    DECLR_ACC1(1,0);
    DECLR_ACC1(2,0);
    DECLR_ACC1(3,0);

    DECLR_A1(1);
    DECLR_B1(1);
    for(BLASLONG k = 0; k < K; k++)
    {
      LOADA1(1);

      LOADB1(1,0);
      KERNEL1(0,0,1);
      LOADB1(1,1);
      KERNEL1(1,0,1);
      LOADB1(1,2);
      KERNEL1(2, 0, 1);
      LOADB1(1,3);
      KERNEL1(3, 0, 1);

      ptrba += 2;
      ptrbb += 4;
    }

    STR_C1(0, 0, 0);
    STR_C1(1, 0, 0);
    STR_C1(2, 0, 0);
    STR_C1(3, 0, 0);
  }

/*inline*/ void kernel_2x2(BLASLONG K, FLOAT alpha,
  FLOAT *ptrba, FLOAT *ptrbb, FLOAT *C0, FLOAT *C1)
  {
    DECLR_ACC1(0,0);
    DECLR_ACC1(1,0);

    DECLR_A1(1);
    DECLR_B1(1);
    for(BLASLONG k = 0; k < K; k++)
    {
      LOADA1(1);

      LOADB1(1,0);
      KERNEL1(0,0,1);
      LOADB1(1,1);
      KERNEL1(1,0,1);

      ptrba += 2;
      ptrbb += 2;
    }

    STR_C1(0, 0, 0);
    STR_C1(1, 0, 0);
  }

// A m x k
// B k x n
int CNAME(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha,
  FLOAT *sa, FLOAT *sb, FLOAT *C, BLASLONG LDC
  )
{
  BLASLONG i = 0,j = 0,k = 0;
  FLOAT *C0, *C1, *C2, *C3;

  IFLOAT *ptrba,*ptrbb;
  FLOAT res00, res01, res02, res03;
  FLOAT res10, res11, res12, res13;
  FLOAT res20/*, res21, res22, res23*/;
  FLOAT res30/*, res31, res32, res33*/;

  // FLOAT res04, res05, res06, res07;
  // FLOAT res14, res15, res16, res17;
  // FLOAT res24, res25, res26, res27;
  // FLOAT res34, res35, res36, res37;

  for(j = 0; j + 4 <= N; j += 4)
  {
    C0 = C + (0 + j)*LDC;
    C1 = C + (1 + j)*LDC;
    C2 = C + (2 + j)*LDC;
    C3 = C + (3 + j)*LDC;

    ptrba = sa;
    for(i = 0; i + 8 <= M; i+=8)
    {
      ptrbb = sb + j*K;

      kernel_8x4(K, alpha, ptrba, ptrbb, C0, C1, C2, C3);

      ptrba += 8*K;
      ptrbb += 4*K;
      C0 += 8;
      C1 += 8;
      C2 += 8;
      C3 += 8;
    }
    for(; i + 4 <= M; i+=4)
    {
      ptrbb = sb + j*K;

      kernel_4x4(K, alpha, ptrba, ptrbb, C0, C1, C2, C3);

      ptrba += 4*K;
      ptrbb += 4*K;

      C0 += 4;
      C1 += 4;
      C2 += 4;
      C3 += 4;
    }
    for(; i + 2 <= M; i+=2)
    {
      ptrbb = sb + j*K;

      kernel_2x4(K, alpha, ptrba, ptrbb, C0, C1, C2, C3);

      ptrba += 2*K;
      ptrbb += 4*K;

      C0 += 2;
      C1 += 2;
      C2 += 2;
      C3 += 2;
    }
    for(; i < M; i++)
    {
      ptrbb = sb + j*K;

      res00 = 0;
      res10 = 0;
      res20 = 0;
      res30 = 0;

      for(k = 0; k < K; k++)
      {
        res00 += ptrba[0]*ptrbb[0];

        res10 += ptrba[0]*ptrbb[1];

        res20 += ptrba[0]*ptrbb[2];

        res30 += ptrba[0]*ptrbb[3];

        ptrba += 1;
        ptrbb += 4;
      }

      C0[0] += alpha*res00;

      C1[0] += alpha*res10;

      C2[0] += alpha*res20;

      C3[0] += alpha*res30;

      C0 += 1;
      C1 += 1;
      C2 += 1;
      C3 += 1;
    }
  }
  for(; j + 2 <= N; j+=2)
  {
    C0 = C + (0 + j)*LDC;
    C1 = C + (1 + j)*LDC;

    ptrba = sa;
    for(i = 0; i + 8 <= M; i+=8)
    {
      ptrbb = sb + j*K;

      kernel_8x2(K, alpha, ptrba, ptrbb, C0, C1);

      ptrba += 8*K;
      ptrbb += 2*K;
      C0 += 8;
      C1 += 8;
    }
    for(; i + 4 <= M; i+=4)
    {
      ptrbb = sb + j*K;

      kernel_4x2(K, alpha, ptrba, ptrbb, C0, C1);

      ptrba += 4*K;
      ptrbb += 2*K;
      C0 += 4;
      C1 += 4;
    }
    for(; i + 2 <= M; i+=2)
    {
      ptrbb = sb + j*K;

      kernel_2x2(K, alpha, ptrba, ptrbb, C0, C1);

      ptrba += 2*K;
      ptrbb += 2*K;
      C0 += 2;
      C1 += 2;
    }
    for(; i < M; i++)
    {
      ptrbb = sb + j*K;

      res00 = res01 = res02 = res03 = 0;
      res10 = res11 = res12 = res13 = 0;

      for(k = 0; k < K; k++)
      {
        res00 += ptrba[0]*ptrbb[0];

        res10 += ptrba[0]*ptrbb[1];

        ptrba += 1;
        ptrbb += 2;
      }
      C0[0] += alpha*res00;

      C1[0] += alpha*res10;
      
      C0 += 1;
      C1 += 1;
    }
  }
  for(; j < N; j++)
  {
    C0 = C + (0 + j)*LDC;

    ptrba = sa;
    for(i = 0; i + 8 <= M; i+=8)
    {
      ptrbb = sb + j*K;

      kernel_8x1(K, alpha, ptrba, ptrbb, C0);

      ptrba += 8*K;
      ptrbb += 1*K;
      C0 += 8;
    }
    for(; i + 4 <= M; i+=4)
    {
      ptrbb = sb + j*K;
      res00 = res01 = res02 = res03 = 0;

      for(k = 0; k < K; k++)
      {
        res00 += ptrba[0]*ptrbb[0];
        res01 += ptrba[1]*ptrbb[0];
        res02 += ptrba[2]*ptrbb[0];
        res03 += ptrba[3]*ptrbb[0];

        ptrba += 4;
        ptrbb += 1;
      }

      C0[0] += alpha*res00;
      C0[1] += alpha*res01;
      C0[2] += alpha*res02;
      C0[3] += alpha*res03;

      C0 += 4;
    }
    for(; i + 2 <= M; i+=2)
    {
      ptrbb = sb + j*K;
      res00 = res01 = res02 = res03 = 0;

      for(k = 0; k < K; k++)
      {
        res00 += ptrba[0]*ptrbb[0];
        res01 += ptrba[1]*ptrbb[0];

        ptrba += 2;
        ptrbb += 1;
      }

      C0[0] += alpha*res00;
      C0[1] += alpha*res01;

      C0 += 2;

    }
    for(; i < M; i++)
    {
      ptrbb = sb + j*K;
      res00 = res01 = res02 = res03 = 0;

      for(k = 0; k < K; k++)
      {
        res00 += ptrba[0]*ptrbb[0];

        ptrba += 1;
        ptrbb += 1;
      }

      C0[0] += alpha*res00;

      C0 += 1;
    }
  }
  return 0;
}