#!/bin/bash

for W in $(seq 10 10 100); do
    JOB_NAME="dnb2_tree2_w$W"
    echo "Found job $JOB_NAME."
    sbatch --job-name $JOB_NAME --output $JOB_NAME.1.log submit.sh $W $JOB_NAME.log
done
