#!/bin/bash
source bisect_run.sh
VAL=$(cat ./tmp/res)
REF=2180213
exit $(echo "$VAL/$REF > 0.98" | bc)