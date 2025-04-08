# -*- coding: utf-8 -*-
import os
import sys
import argparse
import json
from datetime import datetime
import pandas as pd

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


def generate_history(chief_complaint, history_state, temperature, top_p):
    # 记录开始时间
    start_time = datetime.now()

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
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=generation_config.repetition_penalty,
            do_sample=True
        )

    response = tokenizer.decode(output[0][len(inputs.input_ids[0]):],
                                skip_special_tokens=True)

    # 计算耗时
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()

    # 更新历史记录
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {
        "timestamp": timestamp,
        "chief_complaint": chief_complaint,
        "history": response,
        "elapsed_time": f"{elapsed_time:.2f}秒"
    }

    # 优先使用state存储，防止多用户并发问题
    updated_history = history_state + [new_entry] if history_state else [new_entry]

    # 同时维护全局列表（可选持久化）
    history_records.append(new_entry)

    # 返回两个值：格式化后的HTML展示 + 状态存储
    return format_history(updated_history), updated_history


def export_history(history_state, file_format):
    # 防止无数据导出
    if not history_state:
        raise gr.Error("当前无历史记录可导出！")

    # 生成时间戳文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"history_export_{timestamp}"

    # 创建导出目录（如果不存在）
    export_dir = os.path.join(ROOT_DIR, "exports")
    os.makedirs(export_dir, exist_ok=True)

    # 转换数据格式
    if file_format == "JSON":
        file_path = os.path.join(export_dir, f"{filename}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history_state, f,
                      indent=2,
                      ensure_ascii=False)

    elif file_format == "Excel":
        # 需要安装 pandas 和 openpyxl
        df = pd.DataFrame(history_state)
        file_path = os.path.join(export_dir, f"{filename}.xlsx")
        df.to_excel(file_path, index=False)

    # 返回文件路径
    return file_path


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
            <div><strong>耗时：</strong>{entry['elapsed_time']}</div>
        </li>
        """
    html += "</ul>"
    return html


# 读取CSS文件
css_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.css")
with open(css_file_path, "r", encoding="utf-8") as f:
    css_content = f.read()

# 创建Gradio界面
with gr.Blocks(css=css_content, theme=gr.themes.Base()) as demo:
    with gr.Column(elem_classes="container"):
        with gr.Column(elem_classes="header"):
            gr.Markdown("# 现病史生成系统")
            gr.Markdown("输入患者的主诉，自动生成符合规范的现病史")

        # 输入框
        with gr.Column(elem_classes="card"):
            with gr.Row():
                with gr.Column(scale=4):
                    chief_complaint = gr.Textbox(
                        label="输入主诉",
                        placeholder="例如：发热3天，咳嗽伴痰多",
                        lines=3,
                        elem_id="chief-complaint"
                    )
                with gr.Column(scale=1, min_width=120):
                    generate_btn = gr.Button("生成现病史", variant="primary")

        # 示例展示
        with gr.Column(elem_classes="card"):
            gr.Markdown("### 示例输入")
            gr.Examples(
                examples=[
                    ["咳嗽伴声音嘶哑3天，呕吐1天"],
                    ["发热2天，伴有头痛"],
                    ["腹痛1周，间歇性加重"]
                ],
                inputs=chief_complaint,
                label=""
            )

        # 参数调节
        with gr.Column(elem_classes="card"):
            with gr.Accordion("生成参数调节", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        temperature = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.7, step=0.05,
                            label="Temperature（随机性）",
                            info="值越高，生成内容越随机"
                        )
                    with gr.Column(scale=1):
                        top_p = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                            label="Top-p（概率截断）",
                            info="值越低，生成内容越保守"
                        )

        with gr.Column(elem_classes="card"):
            gr.Markdown("### 历史记录")
            history_output = gr.HTML(label="")

        # 导出功能
        with gr.Column(elem_classes="card"):
            with gr.Row():
                with gr.Column(scale=1):
                    export_format = gr.Radio(
                        choices=["JSON", "Excel"],
                        value="JSON",
                        label="导出格式"
                    )
                with gr.Column(scale=1):
                    export_btn = gr.Button("导出历史记录", variant="secondary")
                with gr.Column(scale=1):
                    export_file = gr.File(label="下载文件", visible=True)

    # 状态存储组件
    history_state = gr.State([])

    # 事件绑定
    generate_btn.click(
        fn=generate_history,
        inputs=[
            chief_complaint,
            history_state,
            temperature,
            top_p
        ],
        outputs=[history_output, history_state]
    )

    export_btn.click(
        fn=export_history,
        inputs=[history_state, export_format],
        outputs=[export_file]
    )

# 启动服务
export_dir = os.path.join(ROOT_DIR, "exports")
demo.queue().launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    allowed_paths=[export_dir]  # 添加允许访问的导出目录
)