#!/bin/bash

EXP_NAME=$1
N=$2

mkdir -p $EXP_NAME

for rand in 11 4000 4100
do 
    ./run.sh model.bin ./${EXP_NAME}/${rand}_${N}.txt  $N $rand
done