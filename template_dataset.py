# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import torch
import json
import numpy as np
from config import system_prompt


class TemplateDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = self.max_source_length + self.max_target_length

        self.data = []
        if data_path:
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f:
                    json_line = json.loads(line)
                    self.data.append({
                        "chief_complaint": json_line["chief_complaint"],
                        "history": json_line["template"],
                        "department": json_line["department"]
                    })
        print("data load ， size：", len(self.data))

    def preprocess(self, chief_complaint, template, department):
        messages = [
            {"role": "system", "content": system_prompt.content + (f"\n科室：{department}" if department else "")},
            {"role": "user", "content": chief_complaint}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 输入编码（主诉 + 科室信息）
        instruction = self.tokenizer(prompt, add_special_tokens=False, max_length=self.max_source_length)
        # 输出编码（现病史）
        response = self.tokenizer(template, add_special_tokens=False, max_length=self.max_target_length)

        # 输入 + 输出 + 填充
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]

        # 截断到 max_seq_length
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            labels = labels[:self.max_seq_length]

        return input_ids, attention_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, attention_mask, labels = self.preprocess(
            item_data["chief_complaint"],
            item_data["history"],
            item_data["department"]  # 传递科室信息
        )
        return {
            "input_ids": torch.LongTensor(np.array(input_ids)),
            "attention_mask": torch.LongTensor(np.array(attention_mask)),
            "labels": torch.LongTensor(np.array(labels))
        }

    def __len__(self):
        return len(self.data)

