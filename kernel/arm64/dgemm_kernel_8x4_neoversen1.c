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

int CNAME(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha,
  FLOAT *sa, FLOAT *sb, FLOAT *C, BLASLONG LDC)
{
  // printf("Entering kernel 8x4 neoverse %ld %ld %ld\n", N, M, K);
  BLASLONG i = 0,j = 0,k = 0;
  FLOAT *C0,*C1;
  IFLOAT *ptrba,*ptrbb;
  FLOAT res0;

  C0 = C;
  for(j = 0; j < N; j++)
  {
    // printf("Inside loop %ld %ld %ld\n", i, j, k);
    // C0 = C;
    // C1 = C0 + LDC;
    ptrba = sa;
    for(i = 0; i < M; i++)
    {
      // printf("Inside loop2 %ld %ld %ld\n", i, j, k);
      ptrbb = sb + i*k;
      res0 = 0;
      for(k = 0; k < K; k++)
      {
        printf("A(%ld,%ld) = %f | B(%ld,%ld) = %f\n", i, k, ptrba[0], k, j, ptrbb[0]);
        res0 += ptrba[0]*ptrbb[0];
        ptrba += 1;
        ptrbb += 1;
      }
      printf("C(%ld,%ld) = %f\n", i, j, res0);
      C0[i*LDC + j] = alpha*res0;
      // C0 += 1;
    }
  }
  return 0;
}