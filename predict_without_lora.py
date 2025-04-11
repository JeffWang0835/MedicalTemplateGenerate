# -*- coding: utf-8 -*-
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from config import model_config, system_prompt, generation_config

# 检查是否使用vllm
if model_config.use_vllm:
    try:
        from vllm import LLM, SamplingParams
        print("使用vllm加速模型推理")
        use_vllm = True
    except ImportError:
        print("未安装vllm，将使用标准推理方式")
        use_vllm = False
else:
    use_vllm = False

# 设备设置
device = torch.device(model_config.device if torch.cuda.is_available() else "cpu")

# 根据是否使用vllm加载模型
if use_vllm:
    # 使用vllm加载模型
    model = LLM(
        model=model_config.get_model_path(),
        tensor_parallel_size=model_config.vllm_tensor_parallel_size,
        trust_remote_code=model_config.vllm_trust_remote_code,
        max_num_batched_tokens=model_config.vllm_max_num_batched_tokens,
        max_num_seqs=model_config.vllm_max_num_seqs,
        max_paddings=model_config.vllm_max_paddings,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get_model_path(), 
        trust_remote_code=model_config.trust_remote_code
    )
else:
    # 标准方式加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_config.get_model_path(),
        trust_remote_code=model_config.trust_remote_code
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get_model_path(), 
        trust_remote_code=model_config.trust_remote_code
    )

# Prompt 数组
prompts = [
    "咳嗽伴声音嘶哑3天，呕吐1天",
    "发热2天，伴有头痛",
    "腹痛1周，间歇性加重",
    "流鼻涕5天，打喷嚏频繁",
    "湿疹3天，皮肤干裂",
    "骨折1周，韧带撕裂"
]

# 批量调用并计算时间
for i, prompt in enumerate(prompts):
    start_time = time.time()  # 开始计时

    messages = [
        {"role": "system", "content": system_prompt.content},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    if use_vllm:
        # 使用vllm生成
        sampling_params = SamplingParams(
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            max_tokens=generation_config.max_new_tokens,
            repetition_penalty=generation_config.repetition_penalty
        )
        outputs = model.generate([text], sampling_params)
        response = outputs[0].outputs[0].text
    else:
        # 标准方式生成
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            repetition_penalty=generation_config.repetition_penalty
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    end_time = time.time()  # 结束计时
    elapsed_time = end_time - start_time  # 计算耗时

    # 输出结果
    print(f"=== Prompt {i + 1} ===")
    print("主诉：" + prompt)
    print("现病史：" + response)
    print(f"耗时：{elapsed_time:.4f} 秒\n")