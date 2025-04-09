import os
import sys
import argparse
import json
from datetime import datetime

# 获取项目根目录路径 (假设api目录在项目根目录下)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from config import model_config, system_prompt, generation_config

# 解析命令行参数
parser = argparse.ArgumentParser(description='启动现病史生成API服务')
parser.add_argument('--use_lora', action='store_true', default=True,
                    help='是否使用LoRA权重，默认为True')
parser.add_argument('--host', type=str, default='0.0.0.0', help='服务监听地址')
parser.add_argument('--port', type=int, default=5000, help='服务监听端口')
args = parser.parse_args()

# 设备设置
device = torch.device(model_config.device if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型和tokenizer
model = None
tokenizer = None

print("Loading model...")
if args.use_lora:
    # 加载基础模型
    model_base = AutoModelForCausalLM.from_pretrained(
        model_config.get_model_path(),
        trust_remote_code=model_config.trust_remote_code,
        # torch_dtype=torch.bfloat16 # 根据需要取消注释或修改dtype
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get_model_path(),
        trust_remote_code=model_config.trust_remote_code
    )
    # 加载LoRA模型
    try:
        model = PeftModel.from_pretrained(model_base, model_config.get_lora_dir()).to(device)
        print("使用LoRA权重加载模型成功")
    except Exception as e:
        print(f"加载LoRA权重失败: {e}")
        print("将尝试加载合并后的模型...")
        # 如果LoRA加载失败，尝试加载合并后的模型（如果存在）
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_config.get_lora_output(),
                trust_remote_code=model_config.trust_remote_code,
                # torch_dtype=torch.bfloat16
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.get_lora_output(),
                trust_remote_code=model_config.trust_remote_code
            )
            print("加载合并后的模型成功")
        except Exception as inner_e:
            print(f"加载合并后的模型也失败: {inner_e}")
            sys.exit("无法加载模型，请检查模型路径和配置。")

else:
    # 加载合并后的模型和tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.get_lora_output(),
            trust_remote_code=model_config.trust_remote_code,
            # torch_dtype=torch.bfloat16
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.get_lora_output(),
            trust_remote_code=model_config.trust_remote_code
        )
        print("使用合并后的模型权重加载模型成功")
    except Exception as e:
        print(f"加载合并后的模型失败: {e}")
        sys.exit("无法加载模型，请检查模型路径和配置。")

if model is None or tokenizer is None:
    sys.exit("模型或Tokenizer未能成功加载。")

model.eval() # 设置为评估模式

# 创建 Flask 应用
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_history_api():
    start_time = datetime.now()
    try:
        data = request.get_json()
        if not data or 'chief_complaint' not in data:
            return jsonify({"error": "请求体需要包含 JSON 格式的 'chief_complaint' 字段"}), 400

        chief_complaint = data['chief_complaint']
        temperature = data.get('temperature', generation_config.temperature) # 允许请求中覆盖默认值
        top_p = data.get('top_p', generation_config.top_p) # 允许请求中覆盖默认值
        max_new_tokens = data.get('max_new_tokens', generation_config.max_new_tokens) # 允许请求中覆盖默认值

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
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=generation_config.repetition_penalty,
                do_sample=True if temperature > 0 else False # temperature为0时不进行采样
            )

        response = tokenizer.decode(output[0][len(inputs.input_ids[0]):],
                                    skip_special_tokens=True)

        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        return jsonify({
            "chief_complaint": chief_complaint,
            "generated_history": response,
            "elapsed_time_seconds": round(elapsed_time, 2),
            "model_used": "LoRA" if args.use_lora else "Merged"
        })

    except Exception as e:
        app.logger.error(f"生成过程中发生错误: {e}")
        return jsonify({"error": "生成过程中发生内部错误", "details": str(e)}), 500

if __name__ == '__main__':
    # 运行 Flask 应用
    # 使用 waitress 或 gunicorn 部署生产环境
    app.run(host=args.host, port=args.port, debug=False) # debug=False 用于生产

