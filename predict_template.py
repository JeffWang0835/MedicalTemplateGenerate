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

prompt = """咳嗽伴声音嘶哑3天，呕吐1天"""

systemContent = """
你是一个医疗方面的专家，根据医生提供的主诉和科室信息生成患者的现病史。科室信息：儿科
示例输入：
    发热3天
示例输出：
    3天前发热，最高体温t℃，无寒战、抽搐，偶咳嗽，可闻及痰响，伴鼻阻、流涕，无发热，无呕吐，腹泻，无皮疹，无结膜充血，精神可。
请按照我的回答实例回答，不要出现多余的澄清性文字，不要使用示例输入的结果。
请像上面的示例一样回答出该例子对应的输出，输出的时候，只返回给我现病史文本，不准输出你的中间思考过程，直接输出现病史，如果你中间写了思考步骤，我的程序会报错。不要出现这种的说明文本，直接给我主诉对应的现病史
约束：在现病史中，应该是时间+主诉症状描述，时间描述应自然融入，使用如“X天前”，“h小时前”等表达方式，避免生硬的时间顺序
    """
messages = [
    {"role": "system", "content": systemContent},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# print(text)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=258)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("主诉：" + prompt)
print("现病史：" + response)

