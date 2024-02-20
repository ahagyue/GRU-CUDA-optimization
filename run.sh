#!/bin/bash

: ${NODES:=1}
srun -N $NODES --partition apws --exclusive \
    ./main $@
