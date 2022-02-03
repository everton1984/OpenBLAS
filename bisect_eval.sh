#!/bin/bash
source bisect_run.sh
VAL=$(cat ./tmp/res)
REF=$1
exit $(echo "$VAL/$REF > 1.02" | bc)