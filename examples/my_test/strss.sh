# step1 
vllm serve /home/lisiqi/work/models/Qwen3-4B  --max-model-len 1024

# step2
vllm bench serve \
  --backend vllm \
  --model /home/lisiqi/work/models/Qwen3-4B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path /home/lisiqi/work/models/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10