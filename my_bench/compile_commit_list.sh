#!/bin/bash
CUR_DIR=$(pwd)

unset COMMIT_LIST_FILE BENCHMARKS_DIR OUT_DIR EIGEN_URL EIGEN_BRANCH

# usage()
# {
#     echo "Usage: $0 [ -b BENCHMARKS_DIR ] [ -o OUT_DIR ] [ -f COMMIT_LIST_FILE ] [ -u EIGEN_URL ] [ -B EIGEN_BRANCH ]"
#     exit 2
# }

# while getopts 'f:b:o:u:B:' opt
# do
#     case $opt in
#         b) BENCHMARKS_DIR=$OPTARG ;;
#         o) OUT_DIR=$OPTARG ;;
#         f) COMMIT_LIST_FILE=$OPTARG ;;
#         u) EIGEN_URL=$OPTARG ;;
#         B) EIGEN_BRANCH=$OPTARG ;;
#     esac
# done

# [ -z $COMMIT_LIST_FILE ] && usage
# [ -z $BENCHMARKS_DIR   ] && usage
# [ -z $OUT_DIR          ] && usage
# [ -z $EIGEN_URL        ] && EIGEN_URL="https://gitlab.com/libeigen/eigen.git"
# [ -z $EIGEN_BRANCH     ] && EIGEN_BRANCH="master"

COMMIT_LIST_FILE=$1

# if [[ ! -e $CUR_DIR/tmp ]]; then 
#     mkdir -p $CUR_DIR/tmp
# fi

# cd $CUR_DIR/tmp

# if [[ ! -e eigen ]]; then 
#     git clone -b $EIGEN_BRANCH $EIGEN_URL &> /dev/null
# fi

# cd eigen
# git checkout $EIGEN_BRANCH &> /dev/null
# git pull origin $EIGEN_BRANCH &> /dev/null
# EIGEN_DIR=$(pwd)

# echo "Eigen dir " $EIGEN_DIR 1>&2
# cd ..

while IFS= read -r COMMIT
do
    echo $COMMIT
    #if [[ ! -d $CUR_DIR/$COMMIT ]]; then
        git checkout $COMMIT
        mkdir -p $CUR_DIR/$COMMIT
        cd ..
        make clean && make -j10 && make install PREFIX=$CUR_DIR/$COMMIT
        cd $CUR_DIR
    #fi
    # rm -Rf build && mkdir build && cd build
    # cmake ..
    # cd $CUR_DIR
    # bash run.sh -e $EIGEN_DIR -b $BENCHMARKS_DIR -o $OUT_DIR
    # cd $EIGEN_DIR
done < $COMMIT_LIST_FILE

git checkout bench_me

