# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model_path = "model/Qwen2-1.5B-Instruct"
lora_dir = "output"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = PeftModel.from_pretrained(model, lora_dir)
model.to(device)

prompt = """
5月至今上腹靠右隐痛，右背隐痛带酸，便秘，喜睡，时有腹痛，头痛，腰酸症状？
"""
messages = [
    {"role": "system", "content": "你是一个医疗方面的专家，可以根据患者的问题进行解答。"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=258)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

