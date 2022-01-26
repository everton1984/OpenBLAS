#!/bin/bash
CUR_DIR=$(pwd)
COMMIT_COUNT=0

unset FROM_YEAR TO_YEAR MAX_COMMITS

FROM_YEAR=2021
TO_YEAR=2022
for ((YEAR = $TO_YEAR; YEAR >= FROM_YEAR; YEAR--)); do
    for ((MONTH = 12; MONTH >= 1; MONTH--)); do
        TYEAR=$YEAR
        NMONTH=$((MONTH + 1))
        if [[ $NMONTH -gt 12 ]]; then
            NMONTH=1
            TYEAR=$((YEAR + 1))
        fi

        AFTER=$YEAR/$MONTH/1
        BEFORE=$TYEAR/$NMONTH/1
        COMMIT=$(git log -n 1 --after=$AFTER --before=$BEFORE --no-decorate --pretty=format:'%h %as' | awk '{gsub(/-/,""); print $1}')
        if [[ ! -z $COMMIT ]]; then
            echo $COMMIT
            COMMIT_COUNT=$((COMMIT_COUNT + 1))
            if [[ ! -z $MAX_COMMITS ]]; then
                if [[ COMMIT_COUNT -ge MAX_COMMITS ]]; then
                    exit
                fi
            fi
        fi
    done
done
