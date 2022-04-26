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

// #define __DEBUG1__
// #define __DEBUG0__

#ifdef __DEBUG0__
#define print(...) printf(__VA_ARGS__)
#else
#define print(...) 
#endif

#ifdef __DEBUG1__
#define print1(...) printf(__VA_ARGS__)
#else
#define print1(...) 
#endif

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

#define DECLR_B1(N) \
  float64x2_t rb##N;

#define DECLR_A2() \
  DECLR_A1(1) \
  DECLR_A1(2)

#define DECLR_A4() \
  DECLR_A1(1) \
  DECLR_A1(2) \
  DECLR_A1(3) \
  DECLR_A1(4)

#define LOADA1(N) \
  ra##N = vld1q_f64(ptrba + (N-1)*2);

#define LOADA2() \
  LOADA1(1) \
  LOADA1(2)

#define LOADA4() \
  LOADA1(1) \
  LOADA1(2) \
  LOADA1(3) \
  LOADA1(4)

#define LOADB1(N, M) \
  rb##N = vdupq_n_f64(ptrbb[M]);

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

#define STR_C2(N) \
  STR_C1(N, 0, 0); \
  STR_C1(N, 1, 2);

#define STR_C4(N) \
  STR_C1(N, 0, 0); \
  STR_C1(N, 1, 2); \
  STR_C1(N, 2, 4); \
  STR_C1(N, 3, 6);

inline void kernel_8x4(BLASLONG K, FLOAT alpha,
  FLOAT *ptrba, FLOAT *ptrbb, FLOAT *C0, FLOAT *C1, FLOAT *C2, FLOAT *C3)
  {
    DECLR_ACC4(0);
    DECLR_ACC4(1);
    DECLR_ACC4(2);
    DECLR_ACC4(3);

    DECLR_A4();
    DECLR_B1(1);
    for(BLASLONG k = 0; k < K; k++)
    {
      print("A %f %f %f %f\n", ptrba[0], ptrba[1], ptrba[2], ptrba[3]);
      print("A %f %f %f %f\n", ptrba[4], ptrba[5], ptrba[6], ptrba[7]);
      print("B %f %f %f %f\n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);

      LOADA4();

      LOADB1(1,0);
      KERNEL4x1(0);
      LOADB1(1,1);
      KERNEL4x1(1);
      LOADB1(1,2);
      KERNEL4x1(2);
      LOADB1(1,3);
      KERNEL4x1(3);

      ptrba += 8;
      ptrbb += 4;
    }

    print("\n");

    STR_C4(0);
    STR_C4(1);
    STR_C4(2);
    STR_C4(3);
  }

inline void kernel_8x2(BLASLONG K, FLOAT alpha,
  FLOAT *ptrba, FLOAT *ptrbb, FLOAT *C0, FLOAT *C1)
  {
    DECLR_ACC4(0);
    DECLR_ACC4(1);

    DECLR_A4();
    DECLR_B1(1);
    for(BLASLONG k = 0; k < K; k++)
    {
      print("A %f %f %f %f\n", ptrba[0], ptrba[1], ptrba[2], ptrba[3]);
      print("A %f %f %f %f\n", ptrba[4], ptrba[5], ptrba[6], ptrba[7]);
      print("B %f %f\n", ptrbb[0], ptrbb[1]);

      LOADA4();

      LOADB1(1,0);
      KERNEL4x1(0);
      LOADB1(1,1);
      KERNEL4x1(1);

      ptrba += 8;
      ptrbb += 2;
    }

    print("\n");

    STR_C4(0);
    STR_C4(1);
  }

inline void kernel_4x4(BLASLONG K, FLOAT alpha,
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
      print("A %f %f %f %f\n", ptrba[0], ptrba[1], ptrba[2], ptrba[3]);
      print("B %f %f %f %f\n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);

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

    print("\n");

    STR_C2(0);
    STR_C2(1);
    STR_C2(2);
    STR_C2(3);
  }

inline void kernel_4x2(BLASLONG K, FLOAT alpha,
  FLOAT *ptrba, FLOAT *ptrbb, FLOAT *C0, FLOAT *C1)
  {
    DECLR_ACC2(0);
    DECLR_ACC2(1);

    DECLR_A2();
    DECLR_B1(1);
    for(BLASLONG k = 0; k < K; k++)
    {
      print("A %f %f %f %f\n", ptrba[0], ptrba[1], ptrba[2], ptrba[3]);
      print("B %f %f\n", ptrbb[0], ptrbb[1]);

      LOADA2();

      LOADB1(1,0);
      KERNEL2x1(0);
      LOADB1(1,1);
      KERNEL2x1(1);

      ptrba += 4;
      ptrbb += 2;
    }

    print("\n");

    STR_C2(0);
    STR_C2(1);
  }

inline void kernel_2x4(BLASLONG K, FLOAT alpha,
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
      print("A %f %f\n", ptrba[0], ptrba[1]);
      print("B %f %f %f %f\n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);

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

    print("\n");

    STR_C1(0, 0, 0);
    STR_C1(1, 0, 0);
    STR_C1(2, 0, 0);
    STR_C1(3, 0, 0);
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
  FLOAT res20, res21, res22, res23;
  FLOAT res30, res31, res32, res33;

  FLOAT res04, res05, res06, res07;
  FLOAT res14, res15, res16, res17;
  FLOAT res24, res25, res26, res27;
  FLOAT res34, res35, res36, res37;

  print1("M %ld N %ld K %ld A %f\n", M, N, K, alpha);

#ifdef __DEBUG_BLOCKS__
  print("blockA\n");
  for(BLASLONG ii = 0; ii < M*K; ii++)
    print("%f ", sa[ii]);
  print("\n");

  print("blockB\n");
  for(BLASLONG ii = 0; ii < K*N; ii++)
    print("%f ", sb[ii]);
  print("\n");
#endif

  print1("Starting j=4\n");
  for(j = 0; j + 4 <= N; j += 4)
  {
    C0 = C + (0 + j)*LDC;
    C1 = C + (1 + j)*LDC;
    C2 = C + (2 + j)*LDC;
    C3 = C + (3 + j)*LDC;

    ptrba = sa;
    print1("Starting i=8\n");
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
    print1("Starting i=4 (%ld)\n",i);
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
    print1("Starting i=2 (%ld)\n",i);
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
    print1("Starting i=1 (%ld)\n",i);
    for(; i < M; i++)
    {

      ptrbb = sb + j*K;
      res00 = 0;
      res10 = 0;
      res20 = 0;
      res30 = 0;

      for(k = 0; k < K; k++)
      {
        print("A %f\n", ptrba[0]);
        print("B %f %f %f %f\n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);

        res00 += ptrba[0]*ptrbb[0];

        res10 += ptrba[0]*ptrbb[1];

        res20 += ptrba[0]*ptrbb[2];

        res30 += ptrba[0]*ptrbb[3];

        ptrba += 1;
        ptrbb += 4;
      }

      print("\n");

      print1("C %f %f %f %f\n\n", res00, res10, res20, res30);
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
  print1("Starting j=2 (%ld)\n",j);
  for(; j + 2 <= N; j+=2)
  {
    C0 = C + (0 + j)*LDC;
    C1 = C + (1 + j)*LDC;

    ptrba = sa;
    print1("Starting i=8\n");
    for(i = 0; i + 8 <= M; i+=8)
    {
      ptrbb = sb + j*K;

      kernel_8x2(K, alpha, ptrba, ptrbb, C0, C1);

      ptrba += 8*K;
      ptrbb += 2*K;
      C0 += 8;
      C1 += 8;
    }
    print1("Starting i=4 (%ld)\n",i);
    for(; i + 4 <= M; i+=4)
    {
      ptrbb = sb + j*K;

      kernel_4x2(K, alpha, ptrba, ptrbb, C0, C1);

      ptrba += 4*K;
      ptrbb += 2*K;
      C0 += 4;
      C1 += 4;
    }
    print1("Starting i=2 (%ld)\n",i);
    for(; i + 2 <= M; i+=2)
    {
      // print("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb + j*K;
      res00 = res01 = res02 = res03 = 0;
      res10 = res11 = res12 = res13 = 0;
      // res20 = res21 = res22 = res23 = 0;
      // res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_n_f64(0), r2= vdupq_n_f64(0), r3= vdupq_n_f64(0), r4 = vdupq_n_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);

        print("A %f %f\n", ptrba[0], ptrba[1]);
        print("B %f %f\n", ptrbb[0], ptrbb[1]);

        // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // print("A %f\n", ptrba[0]);
        res00 += ptrba[0]*ptrbb[0];
        res01 += ptrba[1]*ptrbb[0];
        // res02 += ptrba[2]*ptrbb[0];
        // res03 += ptrba[3]*ptrbb[0];

        res10 += ptrba[0]*ptrbb[1];
        res11 += ptrba[1]*ptrbb[1];
        // res12 += ptrba[2]*ptrbb[1];
        // res13 += ptrba[3]*ptrbb[1];

        // res20 += ptrba[0]*ptrbb[2];
        // res21 += ptrba[1]*ptrbb[2];
        // res22 += ptrba[2]*ptrbb[2];
        // res23 += ptrba[3]*ptrbb[2];

        // res30 += ptrba[0]*ptrbb[3];
        // res31 += ptrba[1]*ptrbb[3];
        // res32 += ptrba[2]*ptrbb[3];
        // res33 += ptrba[3]*ptrbb[3];
        ptrba += 2;
        ptrbb += 2;
        // r1 = vmlaq_f64(r1, a, b[0]);
        // r2 = vmlaq_f64(r2, a, b[0]);
        // a = vldq1_f64(ptrba + 2);
        // r3 = vmlaq_f64(r3, a, b[1]);
        // r4 = vmlaq_f64(r4, a, b[1]);
      }

      print("\n");

      // for(k = 0; k < K; k++)
      // {
      //   print("A %f | ", ptrba[0]);
      //   // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   print("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }
      // print("C %f %f %f %f\n", res0, res1, res2, res3);
      C0[0] += alpha*res00;
      C0[1] += alpha*res01;
      // C0[2] = alpha*res02;
      // C0[3] = alpha*res03;

      C1[0] += alpha*res10;
      C1[1] += alpha*res11;
      // C1[2] = alpha*res12;
      // C1[3] = alpha*res13;

      // C2[0] += alpha*res20;
      // C2[1] = alpha*res21;
      // C2[2] = alpha*res22;
      // C2[3] = alpha*res23;

      // C3[0] += alpha*res30;
      // C3[1] = alpha*res31;
      // C3[2] = alpha*res32;
      // C3[3] = alpha*res33;
      
      C0 += 2;
      C1 += 2;
      // C2 += 1;
      // C3 += 1;
    }
    print1("Starting i=1 (%ld)\n",i);
    for(; i < M; i++)
    {
      // print("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb + j*K;
      res00 = res01 = res02 = res03 = 0;
      res10 = res11 = res12 = res13 = 0;
      // res20 = res21 = res22 = res23 = 0;
      // res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_n_f64(0), r2= vdupq_n_f64(0), r3= vdupq_n_f64(0), r4 = vdupq_n_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);

        print("A %f\n", ptrba[0]);
        print("B %f %f\n", ptrbb[0]);

        // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // print("A %f\n", ptrba[0]);
        res00 += ptrba[0]*ptrbb[0];
        // res01 += ptrba[1]*ptrbb[0];
        // res02 += ptrba[2]*ptrbb[0];
        // res03 += ptrba[3]*ptrbb[0];

        res10 += ptrba[0]*ptrbb[1];
        // res11 += ptrba[1]*ptrbb[1];
        // res12 += ptrba[2]*ptrbb[1];
        // res13 += ptrba[3]*ptrbb[1];

        // res20 += ptrba[0]*ptrbb[2];
        // res21 += ptrba[1]*ptrbb[2];
        // res22 += ptrba[2]*ptrbb[2];
        // res23 += ptrba[3]*ptrbb[2];

        // res30 += ptrba[0]*ptrbb[3];
        // res31 += ptrba[1]*ptrbb[3];
        // res32 += ptrba[2]*ptrbb[3];
        // res33 += ptrba[3]*ptrbb[3];
        ptrba += 1;
        ptrbb += 2;
        // r1 = vmlaq_f64(r1, a, b[0]);
        // r2 = vmlaq_f64(r2, a, b[0]);
        // a = vldq1_f64(ptrba + 2);
        // r3 = vmlaq_f64(r3, a, b[1]);
        // r4 = vmlaq_f64(r4, a, b[1]);
      }

      print("\n");

      // for(k = 0; k < K; k++)
      // {
      //   print("A %f | ", ptrba[0]);
      //   // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   print("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }
      // print("C %f %f %f %f\n", res0, res1, res2, res3);
      C0[0] += alpha*res00;
      // C0[1] = alpha*res01;
      // C0[2] = alpha*res02;
      // C0[3] = alpha*res03;

      C1[0] += alpha*res10;
      // C1[1] = alpha*res11;
      // C1[2] = alpha*res12;
      // C1[3] = alpha*res13;

      // C2[0] += alpha*res20;
      // C2[1] = alpha*res21;
      // C2[2] = alpha*res22;
      // C2[3] = alpha*res23;

      // C3[0] += alpha*res30;
      // C3[1] = alpha*res31;
      // C3[2] = alpha*res32;
      // C3[3] = alpha*res33;
      
      C0 += 1;
      C1 += 1;
      // C2 += 1;
      // C3 += 1;
    }
    // sb += 1;
  }
  print1("Starting j=1 (%ld) %ld\n",j, N);
  for(; j < N; j++)
  {
    // print("Inside loop %ld %ld %ld\n", i, j, k);
    // C0 = C;
    // C1 = C0 + LDC;
    C0 = C + (0 + j)*LDC;
    // C1 = C + 1*LDC;
    // C2 = C + 2*LDC;
    // C3 = C + 3*LDC;
    ptrba = sa;
    print1("Starting i=8\n");
    for(i = 0; i + 8 <= M; i+=8)
    {
      // print("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb + j*K;
      res00 = res01 = res02 = res03 = 0;
      // res10 = res11 = res12 = res13 = 0;
      // res20 = res21 = res22 = res23 = 0;
      // res30 = res31 = res32 = res33 = 0;

      res04 = res05 = res06 = res07 = 0;
      // float64x2_t r1 = vdupq_n_f64(0), r2= vdupq_n_f64(0), r3= vdupq_n_f64(0), r4 = vdupq_n_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);

        print("A %f %f %f %f\n", ptrba[0], ptrba[1], ptrba[2], ptrba[3]);
        print("A %f %f %f %f\n", ptrba[4], ptrba[5], ptrba[6], ptrba[7]);
        print("B %f\n", ptrbb[0]);

        // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // print("A %f\n", ptrba[0]);
        res00 += ptrba[0]*ptrbb[0];
        res01 += ptrba[1]*ptrbb[0];
        res02 += ptrba[2]*ptrbb[0];
        res03 += ptrba[3]*ptrbb[0];

        res04 += ptrba[4]*ptrbb[0];
        res05 += ptrba[5]*ptrbb[0];
        res06 += ptrba[6]*ptrbb[0];
        res07 += ptrba[7]*ptrbb[0];

        // res10 += ptrba[0]*ptrbb[1];
        // res11 += ptrba[1]*ptrbb[1];
        // res12 += ptrba[2]*ptrbb[1];
        // res13 += ptrba[3]*ptrbb[1];

        // res20 += ptrba[0]*ptrbb[2];
        // res21 += ptrba[1]*ptrbb[2];
        // res22 += ptrba[2]*ptrbb[2];
        // res23 += ptrba[3]*ptrbb[2];

        // res30 += ptrba[0]*ptrbb[3];
        // res31 += ptrba[1]*ptrbb[3];
        // res32 += ptrba[2]*ptrbb[3];
        // res33 += ptrba[3]*ptrbb[3];
        ptrba += 8;
        ptrbb += 1;
        // r1 = vmlaq_f64(r1, a, b[0]);
        // r2 = vmlaq_f64(r2, a, b[0]);
        // a = vldq1_f64(ptrba + 2);
        // r3 = vmlaq_f64(r3, a, b[1]);
        // r4 = vmlaq_f64(r4, a, b[1]);
      }

      print("\n");

      // for(k = 0; k < K; k++)
      // {
      //   print("A %f | ", ptrba[0]);
      //   // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   print("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }

      // print("C %f %f %f %f\n\n", res00, res01, res02, res03);

      C0[0] += alpha*res00;
      C0[1] += alpha*res01;
      C0[2] += alpha*res02;
      C0[3] += alpha*res03;

      C0[4] += alpha*res04;
      C0[5] += alpha*res05;
      C0[6] += alpha*res06;
      C0[7] += alpha*res07;

      // C1[0] += alpha*res10;
      // C1[1] += alpha*res11;
      // C1[2] += alpha*res12;
      // C1[3] += alpha*res13;

      // C2[0] += alpha*res20;
      // C2[1] += alpha*res21;
      // C2[2] += alpha*res22;
      // C2[3] += alpha*res23;

      // C3[0] += alpha*res30;
      // C3[1] += alpha*res31;
      // C3[2] += alpha*res32;
      // C3[3] += alpha*res33;
      
      C0 += 8;
      // C1 += 4;
      // C2 += 4;
      // C3 += 4;
    }
    print1("Starting i=4 (%ld)\n",i);
    for(; i + 4 <= M; i+=4)
    {
      // print("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb + j*K;
      res00 = res01 = res02 = res03 = 0;
      // res10 = res11 = res12 = res13 = 0;
      // res20 = res21 = res22 = res23 = 0;
      // res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_n_f64(0), r2= vdupq_n_f64(0), r3= vdupq_n_f64(0), r4 = vdupq_n_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);

        print("A %f %f %f %f\n", ptrba[0], ptrba[1], ptrba[2], ptrba[3]);
        print("B %f\n", ptrbb[0]);

        // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // print("A %f\n", ptrba[0]);
        res00 += ptrba[0]*ptrbb[0];
        res01 += ptrba[1]*ptrbb[0];
        res02 += ptrba[2]*ptrbb[0];
        res03 += ptrba[3]*ptrbb[0];

        // res10 += ptrba[0]*ptrbb[1];
        // res11 += ptrba[1]*ptrbb[1];
        // res12 += ptrba[2]*ptrbb[1];
        // res13 += ptrba[3]*ptrbb[1];

        // res20 += ptrba[0]*ptrbb[2];
        // res21 += ptrba[1]*ptrbb[2];
        // res22 += ptrba[2]*ptrbb[2];
        // res23 += ptrba[3]*ptrbb[2];

        // res30 += ptrba[0]*ptrbb[3];
        // res31 += ptrba[1]*ptrbb[3];
        // res32 += ptrba[2]*ptrbb[3];
        // res33 += ptrba[3]*ptrbb[3];
        ptrba += 4;
        ptrbb += 1;
        // r1 = vmlaq_f64(r1, a, b[0]);
        // r2 = vmlaq_f64(r2, a, b[0]);
        // a = vldq1_f64(ptrba + 2);
        // r3 = vmlaq_f64(r3, a, b[1]);
        // r4 = vmlaq_f64(r4, a, b[1]);
      }

      print("\n");

      // for(k = 0; k < K; k++)
      // {
      //   print("A %f | ", ptrba[0]);
      //   // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   print("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }

      // print("C %f %f %f %f\n\n", res00, res01, res02, res03);

      C0[0] += alpha*res00;
      C0[1] += alpha*res01;
      C0[2] += alpha*res02;
      C0[3] += alpha*res03;

      // C1[0] += alpha*res10;
      // C1[1] += alpha*res11;
      // C1[2] += alpha*res12;
      // C1[3] += alpha*res13;

      // C2[0] += alpha*res20;
      // C2[1] += alpha*res21;
      // C2[2] += alpha*res22;
      // C2[3] += alpha*res23;

      // C3[0] += alpha*res30;
      // C3[1] += alpha*res31;
      // C3[2] += alpha*res32;
      // C3[3] += alpha*res33;
      
      C0 += 4;
      // C1 += 4;
      // C2 += 4;
      // C3 += 4;
    }
    print1("Starting i=2 (%ld)\n",i);
    for(; i + 2 <= M; i+=2)
    {
      // print("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb + j*K;
      res00 = res01 = res02 = res03 = 0;
      // res10 = res11 = res12 = res13 = 0;
      // res20 = res21 = res22 = res23 = 0;
      // res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_n_f64(0), r2= vdupq_n_f64(0), r3= vdupq_n_f64(0), r4 = vdupq_n_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);

        print("A %f %f\n", ptrba[0], ptrba[1]);
        print("B %f\n", ptrbb[0]);

        // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // print("A %f\n", ptrba[0]);
        res00 += ptrba[0]*ptrbb[0];
        res01 += ptrba[1]*ptrbb[0];
        // res02 += ptrba[2]*ptrbb[0];
        // res03 += ptrba[3]*ptrbb[0];

        // res10 += ptrba[0]*ptrbb[1];
        // res11 += ptrba[1]*ptrbb[1];
        // res12 += ptrba[2]*ptrbb[1];
        // res13 += ptrba[3]*ptrbb[1];

        // res20 += ptrba[0]*ptrbb[2];
        // res21 += ptrba[1]*ptrbb[2];
        // res22 += ptrba[2]*ptrbb[2];
        // res23 += ptrba[3]*ptrbb[2];

        // res30 += ptrba[0]*ptrbb[3];
        // res31 += ptrba[1]*ptrbb[3];
        // res32 += ptrba[2]*ptrbb[3];
        // res33 += ptrba[3]*ptrbb[3];
        ptrba += 2;
        ptrbb += 1;
        // r1 = vmlaq_f64(r1, a, b[0]);
        // r2 = vmlaq_f64(r2, a, b[0]);
        // a = vldq1_f64(ptrba + 2);
        // r3 = vmlaq_f64(r3, a, b[1]);
        // r4 = vmlaq_f64(r4, a, b[1]);
      }

      print("\n");

      // for(k = 0; k < K; k++)
      // {
      //   print("A %f | ", ptrba[0]);
      //   // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   print("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }
      // print("C %f %f %f %f\n", res0, res1, res2, res3);
      C0[0] += alpha*res00;
      C0[1] += alpha*res01;
      // C0[2] = alpha*res02;
      // C0[3] = alpha*res03;

      // C1[0] += alpha*res10;
      // C1[1] = alpha*res11;
      // C1[2] = alpha*res12;
      // C1[3] = alpha*res13;

      // C2[0] += alpha*res20;
      // C2[1] = alpha*res21;
      // C2[2] = alpha*res22;
      // C2[3] = alpha*res23;

      // C3[0] += alpha*res30;
      // C3[1] = alpha*res31;
      // C3[2] = alpha*res32;
      // C3[3] = alpha*res33;
      
      C0 += 2;
      // C1 += 1;
      // C2 += 1;
      // C3 += 1;
    }
    print1("Starting i=1 (%ld)\n",i);
    for(; i < M; i++)
    {
      // print("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb + j*K;
      res00 = res01 = res02 = res03 = 0;
      // res10 = res11 = res12 = res13 = 0;
      // res20 = res21 = res22 = res23 = 0;
      // res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_n_f64(0), r2= vdupq_n_f64(0), r3= vdupq_n_f64(0), r4 = vdupq_n_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);

        print("A %f\n", ptrba[0]);
        print("B %f\n", ptrbb[0]);

        // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // print("A %f\n", ptrba[0]);
        res00 += ptrba[0]*ptrbb[0];
        // res01 += ptrba[1]*ptrbb[0];
        // res02 += ptrba[2]*ptrbb[0];
        // res03 += ptrba[3]*ptrbb[0];

        // res10 += ptrba[0]*ptrbb[1];
        // res11 += ptrba[1]*ptrbb[1];
        // res12 += ptrba[2]*ptrbb[1];
        // res13 += ptrba[3]*ptrbb[1];

        // res20 += ptrba[0]*ptrbb[2];
        // res21 += ptrba[1]*ptrbb[2];
        // res22 += ptrba[2]*ptrbb[2];
        // res23 += ptrba[3]*ptrbb[2];

        // res30 += ptrba[0]*ptrbb[3];
        // res31 += ptrba[1]*ptrbb[3];
        // res32 += ptrba[2]*ptrbb[3];
        // res33 += ptrba[3]*ptrbb[3];
        ptrba += 1;
        ptrbb += 1;
        // r1 = vmlaq_f64(r1, a, b[0]);
        // r2 = vmlaq_f64(r2, a, b[0]);
        // a = vldq1_f64(ptrba + 2);
        // r3 = vmlaq_f64(r3, a, b[1]);
        // r4 = vmlaq_f64(r4, a, b[1]);
      }

      print("\n");

      // for(k = 0; k < K; k++)
      // {
      //   print("A %f | ", ptrba[0]);
      //   // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   print("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }
      // print("C %f %f %f %f\n", res0, res1, res2, res3);
      C0[0] += alpha*res00;
      // C0[1] = alpha*res01;
      // C0[2] = alpha*res02;
      // C0[3] = alpha*res03;

      // C1[0] += alpha*res10;
      // C1[1] = alpha*res11;
      // C1[2] = alpha*res12;
      // C1[3] = alpha*res13;

      // C2[0] += alpha*res20;
      // C2[1] = alpha*res21;
      // C2[2] = alpha*res22;
      // C2[3] = alpha*res23;

      // C3[0] += alpha*res30;
      // C3[1] = alpha*res31;
      // C3[2] = alpha*res32;
      // C3[3] = alpha*res33;
      
      C0 += 1;
      // C1 += 1;
      // C2 += 1;
      // C3 += 1;
    }
    // sb += 1;
  }
  return 0;
}