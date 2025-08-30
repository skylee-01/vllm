import os
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration



os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"


print(f"NCCL_IB_DISABLE: {os.environ.get('NCCL_IB_DISABLE')}")
print(f"NCCL_P2P_DISABLE: {os.environ.get('NCCL_P2P_DISABLE')}")

MODEL_ID = "/home/lisiqi/work/models/Qwen2.5-VL-7B-Instruct"
SAVE_DIR = "/home/lisiqi/work/models/Qwen2.5-VL-7B-Instruct-fp8"
OFFLOAD_DIR = "/home/lisiqi/work/models/offload_folder"

# 确保offload目录存在
os.makedirs(OFFLOAD_DIR, exist_ok=True)


# 加载模型和分词器
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # 使用bfloat16减少内存使用
    offload_folder=OFFLOAD_DIR,
    low_cpu_mem_usage=True,
    max_memory={0: "10GiB", "cpu": "30GiB"}  # 限制GPU内存使用
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# 配置量化
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# 应用量化算法
oneshot(model=model, recipe=recipe)

# 保存量化后的模型
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

