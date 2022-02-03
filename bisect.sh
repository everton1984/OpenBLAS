#!/bin/bash

first_good=ecf034b2

git checkout $first_good
source bisect_run.sh
REF=$(cat tmp/res)
git checkout bench_me

git bisect start
git bisect bad c1c0d5ce
git bisect good $first_good
git bisect run bisect_run.sh $REF
