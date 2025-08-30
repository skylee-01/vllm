# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch import mode
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    # 尝试使用 Hugging Face 模型名称，或确保本地路径包含所有必需文件
    llm = LLM(model="/home/lisiqi/work/models/Qwen2.5-VL-7B-Instruct-fp8",  # 或者使用完整的本地路径
                enforce_eager=True,
                max_model_len=100,
                # trust_remote_code=True,  # 添加此参数以支持自定义模型
    )
    # Generate texts from the prompts.
    
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
 