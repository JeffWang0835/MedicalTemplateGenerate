# -*- coding: utf-8 -*-
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from config import model_config

# 设备设置
device = torch.device(model_config.device if torch.cuda.is_available() else "cpu")

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_config.model_name, 
    trust_remote_code=model_config.trust_remote_code
)
model = AutoModelForCausalLM.from_pretrained(
    model_config.model_name, 
    trust_remote_code=model_config.trust_remote_code
)
model = PeftModel.from_pretrained(model, model_config.lora_dir).to(device)
print(model)

# 合并model, 同时保存 token
model = model.merge_and_unload()
model.save_pretrained("lora_output")
tokenizer.save_pretrained("lora_output")
