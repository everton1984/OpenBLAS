#!/bin/bash
CUR_DIR=$(pwd)
TMP_DIR=$CUR_DIR/tmp
mkdir -p $TMP_DIR/.local
make clean && make -j10 && make install PREFIX=$TMP_DIR/.local
cd $CUR_DIR
g++ my_bench/bench.cpp -Ofast \
    -L $CUR_DIR/my_bench/benchmark/build/src/ \
    -L $TMP_DIR/.local/lib \
    -I $TMP_DIR/.local/include \
    -I $CUR_DIR/my_bench/benchmark/include \
    -I $HOME/sources/eigen \
    -lbenchmark -lpthread -lopenblas \
    -o bench_bisect
mv bench_bisect $TMP_DIR/bench_bisect
cd $TMP_DIR
LD_LIBRARY_PATH=./.local/lib bench_bisect --benchmark_repetitions=50 --benchmark_report_aggregates_only=true --benchmark_format=json --benchmark_out=out.json --benchmark_filter=BM_Square_DGEMM/512
SCRIPT='import sys, json
J = json.load(sys.stdin)
for el in J["benchmarks"]:
    if "mean" in el["name"]:
        print(el["cpu_time"])'
echo "$SCRIPT" > tmp.py
cat out.json | python tmp.py > res
cd $CUR_DIR
