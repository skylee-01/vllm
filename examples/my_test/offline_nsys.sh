#!/bin/bash
# 需要通过sudo sh xx.sh执行


export VLLM_USE_V1=0  # 仅仅在vllm v0版本中生效。
/usr/local/bin/nsys profile --gpu-metrics-devices=0 -t cuda,nvtx,osrt,cudnn,cublas \
        --gpu-metrics-set=ad10x \
        -o /home/lisiqi/work/tmp/vllm_nsys15 --force-overwrite true \
        --python-backtrace=cuda \
        --cudabacktrace=kernel \
        --backtrace=lbr \
        /home/lisiqi/app/miniconda3/envs/py12/bin/python /home/lisiqi/work/github/my_vllm/vllm/examples/my_test/offline_test.py


# export VLLM_USE_V1=1 # v1 配置 
# /usr/local/bin/nsys profile --gpu-metrics-devices=0 --trace-fork-before-exec=true \
#         --cuda-graph-trace=node \
#         --force-overwrite true \
#         -o /home/lisiqi/work/tmp/vllm_nsys10 \
#         /home/lisiqi/app/miniconda3/envs/py12/bin/python /home/lisiqi/work/github/my_vllm/vllm/examples/my_test/offline_test.py

