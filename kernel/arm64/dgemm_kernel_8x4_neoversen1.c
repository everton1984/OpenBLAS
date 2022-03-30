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


// A m x k
// B k x n
int CNAME(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha,
  FLOAT *sa, FLOAT *sb, FLOAT *C, BLASLONG LDC
#ifdef TRMMKERNEL
  ,BLASLONG offset
#endif
  )
{
  // printf("Entering kernel 8x4 neoverse %ld %ld %ld\n", N, M, K);
  BLASLONG i = 0,j = 0,k = 0;
  FLOAT *C0, *C1, *C2, *C3;
  IFLOAT *ptrba,*ptrbb;
  FLOAT res00, res01, res02, res03;
  FLOAT res10, res11, res12, res13;
  FLOAT res20, res21, res22, res23;
  FLOAT res30, res31, res32, res33;

  printf("M %ld N %ld K %ld\n", M, N, K);
  printf("blockA\n");
  for(BLASLONG ii = 0; ii < M*K; ii++)
    printf("%f ", sa[ii]);
  printf("\n");

  printf("blockB\n");
  for(BLASLONG ii = 0; ii < K*N; ii++)
    printf("%f ", sb[ii]);
  printf("\n");

  for(j = 0; j + 4 <= N; j += 4)
  {
    // printf("Inside loop %ld %ld %ld\n", i, j, k);
    // C0 = C;
    // C1 = C0 + LDC;
    C0 = C + 0*LDC;
    C1 = C + 1*LDC;
    C2 = C + 2*LDC;
    C3 = C + 3*LDC;
    ptrba = sa;
    for(i = 0; i + 4 <= M; i+=4)
    {
      // printf("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb;
      res00 = res01 = res02 = res03 = 0;
      res10 = res11 = res12 = res13 = 0;
      res20 = res21 = res22 = res23 = 0;
      res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_f64(0), r2= vdupq_f64(0), r3= vdupq_f64(0), r4 = vdupq_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);
        printf("A %f %f %f %f\n", ptrba[0], ptrba[1], ptrba[2], ptrba[3]);
        printf("B %f %f %f %f\n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // printf("A %f\n", ptrba[0]);
        res00 += ptrba[0]*ptrbb[0];
        res01 += ptrba[1]*ptrbb[0];
        res02 += ptrba[2]*ptrbb[0];
        res03 += ptrba[3]*ptrbb[0];

        res10 += ptrba[0]*ptrbb[1];
        res11 += ptrba[1]*ptrbb[1];
        res12 += ptrba[2]*ptrbb[1];
        res13 += ptrba[3]*ptrbb[1];

        res20 += ptrba[0]*ptrbb[2];
        res21 += ptrba[1]*ptrbb[2];
        res22 += ptrba[2]*ptrbb[2];
        res23 += ptrba[3]*ptrbb[2];

        res30 += ptrba[0]*ptrbb[3];
        res31 += ptrba[1]*ptrbb[3];
        res32 += ptrba[2]*ptrbb[3];
        res33 += ptrba[3]*ptrbb[3];
        ptrba += 4;
        ptrbb += 4;
        // r1 = vmlaq_f64(r1, a, b[0]);
        // r2 = vmlaq_f64(r2, a, b[0]);
        // a = vldq1_f64(ptrba + 2);
        // r3 = vmlaq_f64(r3, a, b[1]);
        // r4 = vmlaq_f64(r4, a, b[1]);
      }
      printf("\n");
      // for(k = 0; k < K; k++)
      // {
      //   printf("A %f | ", ptrba[0]);
      //   // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   printf("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }
      // printf("C %f %f %f %f\n", res0, res1, res2, res3);
      C0[0] += alpha*res00;
      C0[1] += alpha*res01;
      C0[2] += alpha*res02;
      C0[3] += alpha*res03;

      C1[0] += alpha*res10;
      C1[1] += alpha*res11;
      C1[2] += alpha*res12;
      C1[3] += alpha*res13;

      C2[0] += alpha*res20;
      C2[1] += alpha*res21;
      C2[2] += alpha*res22;
      C2[3] += alpha*res23;

      C3[0] += alpha*res30;
      C3[1] += alpha*res31;
      C3[2] += alpha*res32;
      C3[3] += alpha*res33;
      
      C0 += 4;
      C1 += 4;
      C2 += 4;
      C3 += 4;
    }
    for(; i + 2 <= M; i+=2)
    {
      // printf("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb;
      res00 = res01 = res02 = res03 = 0;
      res10 = res11 = res12 = res13 = 0;
      res20 = res21 = res22 = res23 = 0;
      res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_f64(0), r2= vdupq_f64(0), r3= vdupq_f64(0), r4 = vdupq_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);
        printf("A %f %f\n", ptrba[0], ptrba[1]);
        printf("B %f %f %f %f\n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // printf("A %f\n", ptrba[0]);
        res00 += ptrba[0]*ptrbb[0];
        res01 += ptrba[1]*ptrbb[0];
        // res02 += ptrba[2]*ptrbb[0];
        // res03 += ptrba[3]*ptrbb[0];

        res10 += ptrba[0]*ptrbb[1];
        res11 += ptrba[1]*ptrbb[1];
        // res12 += ptrba[2]*ptrbb[1];
        // res13 += ptrba[3]*ptrbb[1];

        res20 += ptrba[0]*ptrbb[2];
        res21 += ptrba[1]*ptrbb[2];
        // res22 += ptrba[2]*ptrbb[2];
        // res23 += ptrba[3]*ptrbb[2];

        res30 += ptrba[0]*ptrbb[3];
        res31 += ptrba[1]*ptrbb[3];
        // res32 += ptrba[2]*ptrbb[3];
        // res33 += ptrba[3]*ptrbb[3];

        ptrba += 2;
        ptrbb += 4;
        // r1 = vmlaq_f64(r1, a, b[0]);
        // r2 = vmlaq_f64(r2, a, b[0]);
        // a = vldq1_f64(ptrba + 2);
        // r3 = vmlaq_f64(r3, a, b[1]);
        // r4 = vmlaq_f64(r4, a, b[1]);
      }
      printf("\n");
      // for(k = 0; k < K; k++)
      // {
      //   printf("A %f | ", ptrba[0]);
      //   // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   printf("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }
      printf("C %f %f %f %f\n\n", res00, res10, res20, res30);
      C0[0] += alpha*res00;
      C0[1] = alpha*res01;
      // C0[2] = alpha*res02;
      // C0[3] = alpha*res03;

      C1[0] += alpha*res10;
      C1[1] = alpha*res11;
      // C1[2] = alpha*res12;
      // C1[3] = alpha*res13;

      C2[0] += alpha*res20;
      C2[1] = alpha*res21;
      // C2[2] = alpha*res22;
      // C2[3] = alpha*res23;

      C3[0] += alpha*res30;
      C3[1] = alpha*res31;
      // C3[2] = alpha*res32;
      // C3[3] = alpha*res33;

      C0 += 2;
      C1 += 2;
      C2 += 2;
      C3 += 2;
    }
    for(; i < M; i++)
    {
      // printf("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb;
      res00 = res01 = res02 = res03 = 0;
      res10 = res11 = res12 = res13 = 0;
      res20 = res21 = res22 = res23 = 0;
      res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_f64(0), r2= vdupq_f64(0), r3= vdupq_f64(0), r4 = vdupq_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);
        printf("A %f\n", ptrba[0]);
        printf("B %f %f %f %f\n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // printf("A %f\n", ptrba[0]);
        res00 += ptrba[0]*ptrbb[0];
        // res01 += ptrba[1]*ptrbb[0];
        // res02 += ptrba[2]*ptrbb[0];
        // res03 += ptrba[3]*ptrbb[0];

        res10 += ptrba[0]*ptrbb[1];
        // res11 += ptrba[1]*ptrbb[1];
        // res12 += ptrba[2]*ptrbb[1];
        // res13 += ptrba[3]*ptrbb[1];

        res20 += ptrba[0]*ptrbb[2];
        // res21 += ptrba[1]*ptrbb[2];
        // res22 += ptrba[2]*ptrbb[2];
        // res23 += ptrba[3]*ptrbb[2];

        res30 += ptrba[0]*ptrbb[3];
        // res31 += ptrba[1]*ptrbb[3];
        // res32 += ptrba[2]*ptrbb[3];
        // res33 += ptrba[3]*ptrbb[3];

        ptrba += 1;
        ptrbb += 4;
        // r1 = vmlaq_f64(r1, a, b[0]);
        // r2 = vmlaq_f64(r2, a, b[0]);
        // a = vldq1_f64(ptrba + 2);
        // r3 = vmlaq_f64(r3, a, b[1]);
        // r4 = vmlaq_f64(r4, a, b[1]);
      }
      printf("\n");
      // for(k = 0; k < K; k++)
      // {
      //   printf("A %f | ", ptrba[0]);
      //   // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   printf("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }
      printf("C %f %f %f %f\n\n", res00, res10, res20, res30);
      C0[0] += alpha*res00;
      // C0[1] = alpha*res01;
      // C0[2] = alpha*res02;
      // C0[3] = alpha*res03;

      C1[0] += alpha*res10;
      // C1[1] = alpha*res11;
      // C1[2] = alpha*res12;
      // C1[3] = alpha*res13;

      C2[0] += alpha*res20;
      // C2[1] = alpha*res21;
      // C2[2] = alpha*res22;
      // C2[3] = alpha*res23;

      C3[0] += alpha*res30;
      // C3[1] = alpha*res31;
      // C3[2] = alpha*res32;
      // C3[3] = alpha*res33;

      C0 += 1;
      C1 += 1;
      C2 += 1;
      C3 += 1;
    }
    // sb += 1;
  }
  for(; j + 2 <= N; j+=2)
  {
    // printf("Inside loop %ld %ld %ld\n", i, j, k);
    // C0 = C;
    // C1 = C0 + LDC;
    C0 = C + 0*LDC;
    C1 = C + 1*LDC;
    // C2 = C + 2*LDC;
    // C3 = C + 3*LDC;
    ptrba = sa;
    for(i = 0; i + 4 <= M; i+=4)
    {
      // printf("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb;
      res00 = res01 = res02 = res03 = 0;
      res10 = res11 = res12 = res13 = 0;
      // res20 = res21 = res22 = res23 = 0;
      // res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_f64(0), r2= vdupq_f64(0), r3= vdupq_f64(0), r4 = vdupq_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);
        printf("A %f %f %f %f\n", ptrba[0], ptrba[1], ptrba[2], ptrba[3]);
        printf("B %f %f\n", ptrbb[0], ptrbb[1]);
        // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // printf("A %f\n", ptrba[0]);
        res00 += ptrba[0]*ptrbb[0];
        res01 += ptrba[1]*ptrbb[0];
        res02 += ptrba[2]*ptrbb[0];
        res03 += ptrba[3]*ptrbb[0];

        res10 += ptrba[0]*ptrbb[1];
        res11 += ptrba[1]*ptrbb[1];
        res12 += ptrba[2]*ptrbb[1];
        res13 += ptrba[3]*ptrbb[1];

        // res20 += ptrba[0]*ptrbb[2];
        // res21 += ptrba[1]*ptrbb[2];
        // res22 += ptrba[2]*ptrbb[2];
        // res23 += ptrba[3]*ptrbb[2];

        // res30 += ptrba[0]*ptrbb[3];
        // res31 += ptrba[1]*ptrbb[3];
        // res32 += ptrba[2]*ptrbb[3];
        // res33 += ptrba[3]*ptrbb[3];
        ptrba += 4;
        ptrbb += 2;
        // r1 = vmlaq_f64(r1, a, b[0]);
        // r2 = vmlaq_f64(r2, a, b[0]);
        // a = vldq1_f64(ptrba + 2);
        // r3 = vmlaq_f64(r3, a, b[1]);
        // r4 = vmlaq_f64(r4, a, b[1]);
      }
      printf("\n");
      // for(k = 0; k < K; k++)
      // {
      //   printf("A %f | ", ptrba[0]);
      //   // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   printf("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }

      printf("C %f %f %f %f\n\n", res00, res01, res02, res03);

      C0[0] += alpha*res00;
      C0[1] += alpha*res01;
      C0[2] += alpha*res02;
      C0[3] += alpha*res03;

      C1[0] += alpha*res10;
      C1[1] += alpha*res11;
      C1[2] += alpha*res12;
      C1[3] += alpha*res13;

      // C2[0] += alpha*res20;
      // C2[1] += alpha*res21;
      // C2[2] += alpha*res22;
      // C2[3] += alpha*res23;

      // C3[0] += alpha*res30;
      // C3[1] += alpha*res31;
      // C3[2] += alpha*res32;
      // C3[3] += alpha*res33;
      
      C0 += 4;
      C1 += 4;
      // C2 += 4;
      // C3 += 4;
    }
    for(; i + 2 <= M; i+=2)
    {
      // printf("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb;
      res00 = res01 = res02 = res03 = 0;
      res10 = res11 = res12 = res13 = 0;
      // res20 = res21 = res22 = res23 = 0;
      // res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_f64(0), r2= vdupq_f64(0), r3= vdupq_f64(0), r4 = vdupq_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);
        printf("A %f %f\n", ptrba[0], ptrba[1]);
        printf("B %f %f\n", ptrbb[0], ptrbb[1]);
        // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // printf("A %f\n", ptrba[0]);
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
      printf("\n");
      // for(k = 0; k < K; k++)
      // {
      //   printf("A %f | ", ptrba[0]);
      //   // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   printf("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }
      // printf("C %f %f %f %f\n", res0, res1, res2, res3);
      C0[0] += alpha*res00;
      C0[1] += alpha*res01;
      // C0[2] = alpha*res02;
      // C0[3] = alpha*res03;

      C1[0] += alpha*res10;
      C1[1] = alpha*res11;
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
    for(; i < M; i++)
    {
      // printf("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb;
      res00 = res01 = res02 = res03 = 0;
      res10 = res11 = res12 = res13 = 0;
      // res20 = res21 = res22 = res23 = 0;
      // res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_f64(0), r2= vdupq_f64(0), r3= vdupq_f64(0), r4 = vdupq_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);
        printf("A %f\n", ptrba[0]);
        printf("B %f %f\n", ptrbb[0]);
        // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // printf("A %f\n", ptrba[0]);
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
      printf("\n");
      // for(k = 0; k < K; k++)
      // {
      //   printf("A %f | ", ptrba[0]);
      //   // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   printf("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }
      // printf("C %f %f %f %f\n", res0, res1, res2, res3);
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
  for(; j < N; j++)
  {
    // printf("Inside loop %ld %ld %ld\n", i, j, k);
    // C0 = C;
    // C1 = C0 + LDC;
    C0 = C + 0*LDC;
    // C1 = C + 1*LDC;
    // C2 = C + 2*LDC;
    // C3 = C + 3*LDC;
    ptrba = sa;
    for(i = 0; i + 4 <= M; i+=4)
    {
      // printf("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb;
      res00 = res01 = res02 = res03 = 0;
      // res10 = res11 = res12 = res13 = 0;
      // res20 = res21 = res22 = res23 = 0;
      // res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_f64(0), r2= vdupq_f64(0), r3= vdupq_f64(0), r4 = vdupq_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);
        printf("A %f %f %f %f\n", ptrba[0], ptrba[1], ptrba[2], ptrba[3]);
        printf("B %f\n", ptrbb[0]);
        // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // printf("A %f\n", ptrba[0]);
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
      printf("\n");
      // for(k = 0; k < K; k++)
      // {
      //   printf("A %f | ", ptrba[0]);
      //   // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   printf("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }

      printf("C %f %f %f %f\n\n", res00, res01, res02, res03);

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
    for(; i + 2 <= M; i+=2)
    {
      // printf("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb;
      res00 = res01 = res02 = res03 = 0;
      // res10 = res11 = res12 = res13 = 0;
      // res20 = res21 = res22 = res23 = 0;
      // res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_f64(0), r2= vdupq_f64(0), r3= vdupq_f64(0), r4 = vdupq_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);
        printf("A %f %f\n", ptrba[0], ptrba[1]);
        printf("B %f\n", ptrbb[0]);
        // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // printf("A %f\n", ptrba[0]);
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
      printf("\n");
      // for(k = 0; k < K; k++)
      // {
      //   printf("A %f | ", ptrba[0]);
      //   // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   printf("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }
      // printf("C %f %f %f %f\n", res0, res1, res2, res3);
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
    for(; i < M; i++)
    {
      // printf("Inside loop2 %ld %ld %ld\n", i, j, k);
      // ptrbb = sb + i*k;
      // ptrba = sa + j;
      //ptrbb = sb;
      ptrbb = sb;
      res00 = res01 = res02 = res03 = 0;
      // res10 = res11 = res12 = res13 = 0;
      // res20 = res21 = res22 = res23 = 0;
      // res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_f64(0), r2= vdupq_f64(0), r3= vdupq_f64(0), r4 = vdupq_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);
        printf("A %f\n", ptrba[0]);
        printf("B %f\n", ptrbb[0]);
        // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // printf("A %f\n", ptrba[0]);
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
      printf("\n");
      // for(k = 0; k < K; k++)
      // {
      //   printf("A %f | ", ptrba[0]);
      //   // printf("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
      //   printf("B %f\n", ptrbb[0]);
      //   res0 += ptrba[0]*ptrbb[0];
      //   res1 += ptrba[1]*ptrbb[0];
      //   res2 += ptrba[2]*ptrbb[0];
      //   res3 += ptrba[3]*ptrbb[0];
      //   ptrba += 4;
      //   ptrbb += 1;
      // }
      // printf("C %f %f %f %f\n", res0, res1, res2, res3);
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