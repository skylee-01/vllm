#!/bin/bash
# 初始化conda
source /home/lisiqi/app/miniconda3/etc/profile.d/conda.sh
conda activate py12
exprt VLLM_USE_V1=1 
# export VLLM_ATTENTION_BACKEND="FLASHINFER"
python3 -m vllm.entrypoints.openai.api_server --served-model-name base_model \
        --port 9001 \
        --model /home/lisiqi/work/models/Qwen3-4B \
        --enforce-eager \
        --max-num-batched-tokens 100 
        # --enable-prefix-caching --disable-custom-all-reduce \
        # --scheduler-delay-factor 0.5 \
        # --tokenizer-mode auto \
        # --trust-remote-code --dtype float16



