#!/bin/bash

export VLLM_USE_V1=1 # v1 配置 
# ncu -o /home/lisiqi/work/tmp/vllm_ncu1 --launch-skip 10 \
#     --launch-count 5 \
#     --replay-mode application "/home/lisiqi/app/miniconda3/envs/py12/bin/python"  /home/lisiqi/work/github/my_vllm/vllm/examples/my_test/offline_test.py

export VLLM_USE_V1=1 # v1 配置 
ncu -o /home/lisiqi/work/tmp/vllm_ncu1 --launch-skip 10 \
    --launch-count 5 \
    --replay-mode application "/home/lisiqi/app/miniconda3/envs/py12/bin/python"  /home/lisiqi/work/tmp/test.py

