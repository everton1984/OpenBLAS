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

inline void kernel_8x4(BLASLONG K, FLOAT alpha,
  FLOAT *ptrba, FLOAT *ptrbb, FLOAT *C0, FLOAT *C1, FLOAT *C2, FLOAT *C3)
  {

  }

// A m x k
// B k x n
int CNAME(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha,
  FLOAT *sa, FLOAT *sb, FLOAT *C, BLASLONG LDC
  )
{
  // print("Entering kernel 8x4 neoverse %ld %ld %ld\n", N, M, K);
  BLASLONG i = 0,j = 0,k = 0;
  FLOAT *C0, *C1, *C2, *C3;
  // FLOAT *C4, *C5, *C6, *C7;
  IFLOAT *ptrba,*ptrbb;
  FLOAT res00, res01, res02, res03;
  FLOAT res10, res11, res12, res13;
  FLOAT res20, res21, res22, res23;
  FLOAT res30, res31, res32, res33;

  FLOAT res04, res05, res06, res07;
  FLOAT res14, res15, res16, res17;
  FLOAT res24, res25, res26, res27;
  FLOAT res34, res35, res36, res37;

// #define __DEBUG__


  print1("M %ld N %ld K %ld A %f\n", M, N, K, alpha);

  print("blockA\n");
  for(BLASLONG ii = 0; ii < M*K; ii++)
    print("%f ", sa[ii]);
  print("\n");

  print("blockB\n");
  for(BLASLONG ii = 0; ii < K*N; ii++)
    print("%f ", sb[ii]);
  print("\n");


  print1("Starting j=4\n");
  for(j = 0; j + 4 <= N; j += 4)
  {
    // print("Inside loop %ld %ld %ld\n", i, j, k);
    // C0 = C;
    // C1 = C0 + LDC;
    C0 = C + (0 + j)*LDC;
    C1 = C + (1 + j)*LDC;
    C2 = C + (2 + j)*LDC;
    C3 = C + (3 + j)*LDC;

    // C4 = C + 4*LDC;
    // C5 = C + 5*LDC;
    // C6 = C + 6*LDC;
    // C7 = C + 7*LDC;
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
      res10 = res11 = res12 = res13 = 0;
      res20 = res21 = res22 = res23 = 0;
      res30 = res31 = res32 = res33 = 0;

      res04 = res05 = res06 = res07 = 0;
      res14 = res15 = res16 = res17 = 0;
      res24 = res25 = res26 = res27 = 0;
      res34 = res35 = res36 = res37 = 0;

      float64x2_t acc00 = vdupq_n_f64(0), acc01 = vdupq_n_f64(0), acc02 = vdupq_n_f64(0), acc03 = vdupq_n_f64(0);
      float64x2_t acc10 = vdupq_n_f64(0), acc11 = vdupq_n_f64(0), acc12 = vdupq_n_f64(0), acc13 = vdupq_n_f64(0);
      float64x2_t acc20 = vdupq_n_f64(0), acc21 = vdupq_n_f64(0), acc22 = vdupq_n_f64(0), acc23 = vdupq_n_f64(0);
      float64x2_t acc30 = vdupq_n_f64(0), acc31 = vdupq_n_f64(0), acc32 = vdupq_n_f64(0), acc33 = vdupq_n_f64(0);

      float64x2_t ra1 = vdupq_n_f64(0), ra2 = vdupq_n_f64(0), ra3 = vdupq_n_f64(0), ra4 = vdupq_n_f64(0);
      float64x2_t rb1 = vdupq_n_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);

        print("A %f %f %f %f\n", ptrba[0], ptrba[1], ptrba[2], ptrba[3]);
        print("A %f %f %f %f\n", ptrba[4], ptrba[5], ptrba[6], ptrba[7]);
        print("B %f %f %f %f\n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);

        ra1 = vld1q_f64(ptrba + 0*2);
        ra2 = vld1q_f64(ptrba + 1*2);
        ra3 = vld1q_f64(ptrba + 2*2);
        ra4 = vld1q_f64(ptrba + 3*2);

        rb1 = vdupq_n_f64(ptrbb[0]);

        acc00 = vfmaq_f64(acc00, ra1, rb1);
        acc01 = vfmaq_f64(acc01, ra2, rb1);
        acc02 = vfmaq_f64(acc02, ra3, rb1);
        acc03 = vfmaq_f64(acc03, ra4, rb1);

        rb1 = vdupq_n_f64(ptrbb[1]);

        acc10 = vfmaq_f64(acc10, ra1, rb1);
        acc11 = vfmaq_f64(acc11, ra2, rb1);
        acc12 = vfmaq_f64(acc12, ra3, rb1);
        acc13 = vfmaq_f64(acc13, ra4, rb1);

        rb1 = vdupq_n_f64(ptrbb[2]);

        acc20 = vfmaq_f64(acc20, ra1, rb1);
        acc21 = vfmaq_f64(acc21, ra2, rb1);
        acc22 = vfmaq_f64(acc22, ra3, rb1);
        acc23 = vfmaq_f64(acc23, ra4, rb1);

        rb1 = vdupq_n_f64(ptrbb[3]);

        acc30 = vfmaq_f64(acc30, ra1, rb1);
        acc31 = vfmaq_f64(acc31, ra2, rb1);
        acc32 = vfmaq_f64(acc32, ra3, rb1);
        acc33 = vfmaq_f64(acc33, ra4, rb1);
        
        // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // print("A %f\n", ptrba[0]);

        // res00 += ptrba[0]*ptrbb[0];
        // res01 += ptrba[1]*ptrbb[0];
        // res02 += ptrba[2]*ptrbb[0];
        // res03 += ptrba[3]*ptrbb[0];

        // res04 += ptrba[4]*ptrbb[0];
        // res05 += ptrba[5]*ptrbb[0];
        // res06 += ptrba[6]*ptrbb[0];
        // res07 += ptrba[7]*ptrbb[0];

        // res10 += ptrba[0]*ptrbb[1];
        // res11 += ptrba[1]*ptrbb[1];
        // res12 += ptrba[2]*ptrbb[1];
        // res13 += ptrba[3]*ptrbb[1];

        // res14 += ptrba[4]*ptrbb[1];
        // res15 += ptrba[5]*ptrbb[1];
        // res16 += ptrba[6]*ptrbb[1];
        // res17 += ptrba[7]*ptrbb[1];

        // res20 += ptrba[0]*ptrbb[2];
        // res21 += ptrba[1]*ptrbb[2];
        // res22 += ptrba[2]*ptrbb[2];
        // res23 += ptrba[3]*ptrbb[2];

        // res24 += ptrba[4]*ptrbb[2];
        // res25 += ptrba[5]*ptrbb[2];
        // res26 += ptrba[6]*ptrbb[2];
        // res27 += ptrba[7]*ptrbb[2];

        // res30 += ptrba[0]*ptrbb[3];
        // res31 += ptrba[1]*ptrbb[3];
        // res32 += ptrba[2]*ptrbb[3];
        // res33 += ptrba[3]*ptrbb[3];

        // res34 += ptrba[4]*ptrbb[3];
        // res35 += ptrba[5]*ptrbb[3];
        // res36 += ptrba[6]*ptrbb[3];
        // res37 += ptrba[7]*ptrbb[3];

        ptrba += 8;
        ptrbb += 4;
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
      C0[0] += alpha*vgetq_lane_f64(acc00, 0);
      C0[1] += alpha*vgetq_lane_f64(acc00, 1);
      C0[2] += alpha*vgetq_lane_f64(acc01, 0);
      C0[3] += alpha*vgetq_lane_f64(acc01, 1);

      C0[4] += alpha*vgetq_lane_f64(acc02, 0);
      C0[5] += alpha*vgetq_lane_f64(acc02, 1);
      C0[6] += alpha*vgetq_lane_f64(acc03, 0);
      C0[7] += alpha*vgetq_lane_f64(acc03, 1);

      C1[0] += alpha*vgetq_lane_f64(acc10, 0);
      C1[1] += alpha*vgetq_lane_f64(acc10, 1);
      C1[2] += alpha*vgetq_lane_f64(acc11, 0);
      C1[3] += alpha*vgetq_lane_f64(acc11, 1);

      C1[4] += alpha*vgetq_lane_f64(acc12, 0);
      C1[5] += alpha*vgetq_lane_f64(acc12, 1);
      C1[6] += alpha*vgetq_lane_f64(acc13, 0);
      C1[7] += alpha*vgetq_lane_f64(acc13, 1);

      C2[0] += alpha*vgetq_lane_f64(acc20, 0);
      C2[1] += alpha*vgetq_lane_f64(acc20, 1);
      C2[2] += alpha*vgetq_lane_f64(acc21, 0);
      C2[3] += alpha*vgetq_lane_f64(acc21, 1);

      C2[4] += alpha*vgetq_lane_f64(acc22, 0);
      C2[5] += alpha*vgetq_lane_f64(acc22, 1);
      C2[6] += alpha*vgetq_lane_f64(acc23, 0);
      C2[7] += alpha*vgetq_lane_f64(acc23, 1);

      C3[0] += alpha*vgetq_lane_f64(acc30, 0);
      C3[1] += alpha*vgetq_lane_f64(acc30, 1);
      C3[2] += alpha*vgetq_lane_f64(acc31, 0);
      C3[3] += alpha*vgetq_lane_f64(acc31, 1);
      
      C3[4] += alpha*vgetq_lane_f64(acc32, 0);
      C3[5] += alpha*vgetq_lane_f64(acc32, 1);
      C3[6] += alpha*vgetq_lane_f64(acc33, 0);
      C3[7] += alpha*vgetq_lane_f64(acc33, 1);

      // C0[0] += alpha*res00;
      // C0[1] += alpha*res01;
      // C0[2] += alpha*res02;
      // C0[3] += alpha*res03;

      // C0[4] += alpha*res04;
      // C0[5] += alpha*res05;
      // C0[6] += alpha*res06;
      // C0[7] += alpha*res07;

      // C1[0] += alpha*res10;
      // C1[1] += alpha*res11;
      // C1[2] += alpha*res12;
      // C1[3] += alpha*res13;

      // C1[4] += alpha*res14;
      // C1[5] += alpha*res15;
      // C1[6] += alpha*res16;
      // C1[7] += alpha*res17;

      // C2[0] += alpha*res20;
      // C2[1] += alpha*res21;
      // C2[2] += alpha*res22;
      // C2[3] += alpha*res23;

      // C2[4] += alpha*res24;
      // C2[5] += alpha*res25;
      // C2[6] += alpha*res26;
      // C2[7] += alpha*res27;

      // C3[0] += alpha*res30;
      // C3[1] += alpha*res31;
      // C3[2] += alpha*res32;
      // C3[3] += alpha*res33;
      
      // C3[4] += alpha*res34;
      // C3[5] += alpha*res35;
      // C3[6] += alpha*res36;
      // C3[7] += alpha*res37;
      // C4[0] += alpha*res40;
      // C4[1] += alpha*res41;
      // C4[2] += alpha*res42;
      // C4[3] += alpha*res43;

      // C5[0] += alpha*res50;
      // C5[1] += alpha*res51;
      // C5[2] += alpha*res52;
      // C5[3] += alpha*res53;

      // C6[0] += alpha*res60;
      // C6[1] += alpha*res61;
      // C6[2] += alpha*res62;
      // C6[3] += alpha*res63;

      // C7[0] += alpha*res70;
      // C7[1] += alpha*res71;
      // C7[2] += alpha*res72;
      // C7[3] += alpha*res73;

      C0 += 8;
      C1 += 8;
      C2 += 8;
      C3 += 8;

      // C4 += 4;
      // C5 += 4;
      // C6 += 4;
      // C7 += 4;
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
      res10 = res11 = res12 = res13 = 0;
      res20 = res21 = res22 = res23 = 0;
      res30 = res31 = res32 = res33 = 0;

      // float64x2_t r1 = vdupq_n_f64(0), r2= vdupq_n_f64(0), r3= vdupq_n_f64(0), r4 = vdupq_n_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);

        print("A %f %f %f %f\n", ptrba[0], ptrba[1], ptrba[2], ptrba[3]);
        print("B %f %f %f %f\n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);

        // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // print("A %f\n", ptrba[0]);
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
      res20 = res21 = res22 = res23 = 0;
      res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_n_f64(0), r2= vdupq_n_f64(0), r3= vdupq_n_f64(0), r4 = vdupq_n_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);

        print("A %f %f\n", ptrba[0], ptrba[1]);
        print("B %f %f %f %f\n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);

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

      // print("C %f %f %f %f\n\n", res00, res10, res20, res30);
    
      C0[0] += alpha*res00;
      C0[1] += alpha*res01;
      // C0[2] = alpha*res02;
      // C0[3] = alpha*res03;

      C1[0] += alpha*res10;
      C1[1] += alpha*res11;
      // C1[2] = alpha*res12;
      // C1[3] = alpha*res13;

      C2[0] += alpha*res20;
      C2[1] += alpha*res21;
      // C2[2] = alpha*res22;
      // C2[3] = alpha*res23;

      C3[0] += alpha*res30;
      C3[1] += alpha*res31;
      // C3[2] = alpha*res32;
      // C3[3] = alpha*res33;

      C0 += 2;
      C1 += 2;
      C2 += 2;
      C3 += 2;
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
      res20 = res21 = res22 = res23 = 0;
      res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_n_f64(0), r2= vdupq_n_f64(0), r3= vdupq_n_f64(0), r4 = vdupq_n_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);

        print("A %f\n", ptrba[0]);
        print("B %f %f %f %f\n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);

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
      print1("C %f %f %f %f\n\n", res00, res10, res20, res30);
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
  print1("Starting j=2 (%ld)\n",j);
  for(; j + 2 <= N; j+=2)
  {
    // print("Inside loop %ld %ld %ld\n", i, j, k);
    // C0 = C;
    // C1 = C0 + LDC;
    C0 = C + (0 + j)*LDC;
    C1 = C + (1 + j)*LDC;
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
      res10 = res11 = res12 = res13 = 0;

      res04 = res05 = res06 = res07 = 0;
      res14 = res15 = res16 = res17 = 0;
      // res20 = res21 = res22 = res23 = 0;
      // res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_n_f64(0), r2= vdupq_n_f64(0), r3= vdupq_n_f64(0), r4 = vdupq_n_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);

        print("A %f %f %f %f\n", ptrba[0], ptrba[1], ptrba[2], ptrba[3]);
        print("A %f %f %f %f\n", ptrba[4], ptrba[5], ptrba[6], ptrba[7]);
        print("B %f %f\n", ptrbb[0], ptrbb[1]);

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

        res10 += ptrba[0]*ptrbb[1];
        res11 += ptrba[1]*ptrbb[1];
        res12 += ptrba[2]*ptrbb[1];
        res13 += ptrba[3]*ptrbb[1];

        res14 += ptrba[4]*ptrbb[1];
        res15 += ptrba[5]*ptrbb[1];
        res16 += ptrba[6]*ptrbb[1];
        res17 += ptrba[7]*ptrbb[1];

        // res20 += ptrba[0]*ptrbb[2];
        // res21 += ptrba[1]*ptrbb[2];
        // res22 += ptrba[2]*ptrbb[2];
        // res23 += ptrba[3]*ptrbb[2];

        // res30 += ptrba[0]*ptrbb[3];
        // res31 += ptrba[1]*ptrbb[3];
        // res32 += ptrba[2]*ptrbb[3];
        // res33 += ptrba[3]*ptrbb[3];
        ptrba += 8;
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

      // print("C %f %f %f %f\n\n", res00, res01, res02, res03);

      C0[0] += alpha*res00;
      C0[1] += alpha*res01;
      C0[2] += alpha*res02;
      C0[3] += alpha*res03;

      C0[4] += alpha*res04;
      C0[5] += alpha*res05;
      C0[6] += alpha*res06;
      C0[7] += alpha*res07;

      C1[0] += alpha*res10;
      C1[1] += alpha*res11;
      C1[2] += alpha*res12;
      C1[3] += alpha*res13;

      C1[4] += alpha*res14;
      C1[5] += alpha*res15;
      C1[6] += alpha*res16;
      C1[7] += alpha*res17;

      // C2[0] += alpha*res20;
      // C2[1] += alpha*res21;
      // C2[2] += alpha*res22;
      // C2[3] += alpha*res23;

      // C3[0] += alpha*res30;
      // C3[1] += alpha*res31;
      // C3[2] += alpha*res32;
      // C3[3] += alpha*res33;
      
      C0 += 8;
      C1 += 8;
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
      res10 = res11 = res12 = res13 = 0;
      // res20 = res21 = res22 = res23 = 0;
      // res30 = res31 = res32 = res33 = 0;
      // float64x2_t r1 = vdupq_n_f64(0), r2= vdupq_n_f64(0), r3= vdupq_n_f64(0), r4 = vdupq_n_f64(0);
      for(k = 0; k < K; k++)
      {
        // float64x2_t a = vld1q_f64(ptrba);
        // float64x2_t b = vld1q_f64(ptrbb);

        print("A %f %f %f %f\n", ptrba[0], ptrba[1], ptrba[2], ptrba[3]);
        print("B %f %f\n", ptrbb[0], ptrbb[1]);

        // print("B %f %f %f %f \n", ptrbb[0], ptrbb[1], ptrbb[2], ptrbb[3]);
        // print("A %f\n", ptrba[0]);
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