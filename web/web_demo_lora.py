# -*- coding: utf-8 -*-
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from config import model_config, system_prompt, generation_config

# 设备设置
device = torch.device(model_config.device if torch.cuda.is_available() else "cpu")

# 加载基础模型
model_base = AutoModelForCausalLM.from_pretrained(
    model_config.get_model_path(),
    trust_remote_code=model_config.trust_remote_code
).to(device)
tokenizer = AutoTokenizer.from_pretrained(
    model_config.get_model_path(),
    trust_remote_code=model_config.trust_remote_code
)

# 加载LoRA模型
model = PeftModel.from_pretrained(model_base, model_config.get_lora_dir()).to(device)

def generate_history(chief_complaint):
    messages = [
        {"role": "system", "content": system_prompt.content},
        {"role": "user", "content": chief_complaint}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            repetition_penalty=generation_config.repetition_penalty,
            do_sample=True
        )

    response = tokenizer.decode(output[0][len(inputs.input_ids[0]):],
                                skip_special_tokens=True)
    return response


# 创建Gradio界面
demo = gr.Interface(
    fn=generate_history,
    inputs=gr.Textbox(label="输入主诉", placeholder="例如：发热3天"),
    outputs=gr.Textbox(label="生成的现病史"),
    examples=[
        ["咳嗽伴声音嘶哑3天，呕吐1天"],
        ["发热2天，伴有头痛"],
        ["腹痛1周，间歇性加重"]
    ],
    title="现病史生成系统",
    description="输入患者的主诉，自动生成符合规范的现病史"
)

# 启动服务
demo.queue().launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True  # 设置为True可生成公网访问链接
)