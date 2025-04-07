# -*- coding: utf-8 -*-
import os
import sys
import argparse
import json
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from config import model_config, system_prompt, generation_config

# 解析命令行参数
parser = argparse.ArgumentParser(description='启动现病史生成系统')
parser.add_argument('--use_lora', action='store_true', default=True, 
                    help='是否使用LoRA权重，默认为True')
args = parser.parse_args()

# 设备设置
device = torch.device(model_config.device if torch.cuda.is_available() else "cpu")

# 根据参数决定加载模型的方式
if args.use_lora:
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
    print("使用LoRA权重加载模型")
else:
    # 加载合并后的模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_config.get_lora_output(),
        trust_remote_code=model_config.trust_remote_code
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get_lora_output(),
        trust_remote_code=model_config.trust_remote_code
    )
    print("使用合并后的模型权重")

# 初始化历史记录存储
history_records = []

def generate_history(chief_complaint, history_state):
    # 生成现病史
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
    
    # 更新历史记录
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {
        "timestamp": timestamp,
        "chief_complaint": chief_complaint,
        "history": response
    }
    
    # 优先使用state存储，防止多用户并发问题
    updated_history = history_state + [new_entry] if history_state else [new_entry]
    
    # 同时维护全局列表（可选持久化）
    history_records.append(new_entry)
    
    # 保存到文件（可选）
    # with open("history.json", "w") as f:
    #     json.dump(history_records, f, indent=2, ensure_ascii=False)
    
    # 返回两个值：格式化后的HTML展示 + 状态存储
    return format_history(updated_history), updated_history

def format_history(history):
    if not history:
        return "<h3>暂无历史记录</h3>"
    
    html = "<h3>历史记录：</h3><ul>"
    for entry in history:
        html += f"""
        <li style='margin-bottom: 15px; border-bottom: 1px solid #eee'>
            <div><strong>时间：</strong>{entry['timestamp']}</div>
            <div><strong>主诉：</strong>{entry['chief_complaint']}</div>
            <div><strong>现病史：</strong>{entry['history']}</div>
        </li>
        """
    html += "</ul>"
    return html

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 现病史生成系统")
    gr.Markdown("输入患者的主诉，自动生成符合规范的现病史")

    with gr.Row():
        chief_complaint = gr.Textbox(
            label="输入主诉",
            placeholder="例如：发热3天",
            lines=2
        )
    
    with gr.Row():
        generate_btn = gr.Button("生成现病史", variant="primary")
    
    with gr.Row():
        history_output = gr.HTML(label="历史记录")
    
    # 状态存储组件
    history_state = gr.State([])
    
    # 示例展示
    gr.Examples(
        examples=[
            ["咳嗽伴声音嘶哑3天，呕吐1天"],
            ["发热2天，伴有头痛"],
            ["腹痛1周，间歇性加重"]
        ],
        inputs=chief_complaint,
        label="示例输入"
    )
    
    # 事件绑定
    generate_btn.click(
        fn=generate_history,
        inputs=[chief_complaint, history_state],
        outputs=[history_output, history_state]
    )

# 启动服务
demo.queue().launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True
)