#include <cblas.h>
#include <Eigen/Dense>
#include <benchmark/benchmark.h>

using namespace Eigen;

template<typename Scalar>
inline void gemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const Scalar alpha, const Scalar  *A,
                 const int lda, const Scalar  *B, const int ldb,
                 const Scalar beta, Scalar  *C, const int ldc)
{
    return cblas_sgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
inline void gemm<double>(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double  *A,
                 const int lda, const double  *B, const int ldb,
                 const double beta, double  *C, const int ldc)
{
    return cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<typename Scalar>
inline void GEMM_Test(benchmark::State& state, const int M, const int K, const int N)
{
    using MyMatrix = Matrix<Scalar, Dynamic, Dynamic>;

    for(auto _ : state)
    {
        state.PauseTiming();
        MyMatrix a = MyMatrix::Random(M, K);
        MyMatrix b = MyMatrix::Random(K, N);
        MyMatrix c = MyMatrix::Zero(M, N);
        state.ResumeTiming();

        gemm<Scalar>(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K, 1, a.data(), M, b.data(), N, 1, c.data(), M);
    }
}

static void BM_Square_SGEMM(benchmark::State& state)
{
    int M = state.range(0);
    int K = state.range(0);
    int N = state.range(0);
    
    GEMM_Test<float>(state, M, K, N);
}
BENCHMARK(BM_Square_SGEMM)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024)->Arg(2048);
BENCHMARK(BM_Square_SGEMM)->Arg(57)->Arg(103)->Arg(201)->Arg(337)->Arg(783)->Arg(1539);

static void BM_Rect_SGEMM(benchmark::State& state)
{
    int M = state.range(0);
    int K = state.range(1);
    int N = state.range(2);
    
    GEMM_Test<float>(state, M, K, N);
}

BENCHMARK(BM_Rect_SGEMM)->Args({40,4,4})->Args({400,4,4})->Args({1024,10,10});
BENCHMARK(BM_Rect_SGEMM)->Args({4,40,4})->Args({4,400,4})->Args({10,1024,10});
BENCHMARK(BM_Rect_SGEMM)->Args({4,4,40})->Args({4,4,400})->Args({10,10,1024});

BENCHMARK_MAIN();
