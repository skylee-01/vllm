
source /home/lisiqi/app/miniconda3/etc/profile.d/conda.sh
conda activate py12
# python /home/lisiqi/work/github/my_vllm/vllm/examples/offline_inference/profiling.py \
#     --model /home/lisiqi/work/models/Qwen3-4B --batch-size 4 \
#     --prompt-len 512 --max-model-len 1024 --json Qwen3-4B --save-chrome-traces-folder /home/lisiqi/work/tmp/\
#     --enforce-eager run_num_steps -n 2 


python /home/lisiqi/work/github/my_vllm/vllm/tools/profiler/print_layerwise_table.py \
            --json-trace /home/lisiqi/work/tmp/decode_1.json --phase prefill --table summary

