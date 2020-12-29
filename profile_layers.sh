#!/bin/bash

set -e

FILE=benchmarks.txt

if [ ! -d "/tmp/benchmark" ]; then
    mkdir "/tmp/benchmark"
fi
if [ ! -d "/tmp/profiling" ]; then
    mkdir "/tmp/profiling"
fi

while read line; do
    echo ${line}

    nsys profile ./benchmark \
        --benchmark_out=/tmp/benchmark/benchmark.csv \
        --benchmark_out_format=csv \
        --benchmark_filter=${line}

    filename=$(echo ${line}|sed -e "s/\//_/g")
    echo ${filename}
    mv "report1.qdrep" "/tmp/profiling/${filename}.qdrep"
done < ${FILE}

