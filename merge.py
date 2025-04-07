# -*- coding: utf-8 -*-
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
import transformers
import torch
from peft import PeftModel
from config import model_config

# 设置设备
device = torch.device(model_config.device if torch.cuda.is_available() else "cpu")

# 加载基础模型和tokenizer
model_base = transformers.AutoModelForCausalLM.from_pretrained(
    model_config.get_model_path(),
    trust_remote_code=model_config.trust_remote_code
).to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_config.get_model_path(),
    trust_remote_code=model_config.trust_remote_code
)

# 加载LoRA模型
model = PeftModel.from_pretrained(model_base, model_config.get_lora_dir()).to(device)

# 合并模型
print("开始合并模型...")
model = model.merge_and_unload()

# 保存合并后的模型
print(f"保存合并后的模型到 {model_config.get_lora_dir()}")
model.save_pretrained(model_config.get_lora_dir())
tokenizer.save_pretrained(model_config.get_lora_dir())
print("模型合并完成！")
