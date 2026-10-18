#!/bin/bash
cd /home/jovyan/work/thuc/VLSP-2026/SRED && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$SRED_DATA/private_test/private-test.txt; A=$SRED_DATA/private_test/data
TRANSDUCTIVE=0 scripts/make_final_submission.sh $T $A result-private-nontrans.zip private > exp/logs/private_nontrans.log 2>&1
cp submissions/final_private/results.txt submissions/final_private/results_nontrans.txt
TRANSDUCTIVE=1 scripts/make_final_submission.sh $T $A result-private-trans.zip private > exp/logs/private_trans.log 2>&1
cp submissions/final_private/results.txt submissions/final_private/results_trans.txt
echo ALLDONE >> exp/logs/private_trans.log
