#!/bin/bash

first_good=ecf034b2

git bisect start
git bisect bad
git bisect good $first_good
git bisect run bisect_run.sh
