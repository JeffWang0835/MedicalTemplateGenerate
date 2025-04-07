# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
from template_dataset import TemplateDataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import os, time, sys

model_name = "model/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def custom_collate_fn(batch):
    """
    自定义 collate_fn，用于处理变长序列数据。
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # 填充到批次内的最大长度
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 是忽略索引

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def train_model(model, train_loader, val_loader, optimizer, gradient_accumulation_steps,
                device, num_epochs, model_output_dir, writer):
    batch_step = 0
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        for index, data in enumerate(tqdm(train_loader, file=sys.stdout, desc="Train Epoch: " + str(epoch))):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            # 反向传播，计算当前梯度
            loss.backward()
            # 梯度累积步数
            if (index % gradient_accumulation_steps == 0 and index != 0) or index == len(train_loader) - 1:
                # 更新网络参数
                optimizer.step()
                # 清空过往梯度
                optimizer.zero_grad()
                writer.add_scalar('Loss/train', loss, batch_step)
                batch_step += 1
            # 100轮打印一次 loss
            if index % 100 == 0 or index == len(train_loader) - 1:
                time2 = time.time()
                tqdm.write(
                    f"{index}, epoch: {epoch} -loss: {str(loss)} ; each step's time spent: {(str(float(time2 - time1) / float(index + 0.0001)))}")
        # 验证
        model.eval()
        val_loss = validate_model(model, val_loader, device)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"val loss: {val_loss} , epoch: {epoch}")
        print("Save Model To ", model_output_dir)
        model.save_pretrained(model_output_dir)


def validate_model(model, val_loader, device):
    running_loss = 0.0
    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader, file=sys.stdout, desc="Validation Data")):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            running_loss += loss.item()
    return running_loss / len(val_loader)


def main():
    # 基础模型位置
    model_name = "model/Qwen2-1.5B-Instruct"
    # 训练集
    train_json_path = "./data/train_template.json"
    # 验证集
    val_json_path = "./data/val_template.json"
    max_source_length = 64
    max_target_length = 256
    epochs = 10
    batch_size = 1
    lr = 1e-4
    gradient_accumulation_steps = 16
    lora_rank = 8
    lora_alpha = 32
    model_output_dir = "output"
    logs_dir = "logs"
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载分词器和模型

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.is_parallelizable = True
    model.model_parallel = True
    model.print_trainable_parameters()
    print("Start Load Train Data...")
    train_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 0,
    }
    training_set = TemplateDataset(train_json_path, tokenizer, max_source_length, max_target_length)
    training_loader = DataLoader(training_set, collate_fn=custom_collate_fn, **train_params)
    print("Start Load Validation Data...")
    val_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0,
    }
    val_set = TemplateDataset(val_json_path, tokenizer, max_source_length, max_target_length)
    val_loader = DataLoader(val_set, collate_fn=custom_collate_fn, **val_params)
    # 日志记录
    writer = SummaryWriter(logs_dir)
    # 优化器
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    model = model.to(device)
    # 开始训练
    print("Start Training...")
    train_model(
        model=model,
        train_loader=training_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        gradient_accumulation_steps=gradient_accumulation_steps,
        device=device,
        num_epochs=epochs,
        model_output_dir=model_output_dir,
        writer=writer
    )
    writer.close()


if __name__ == '__main__':
    main()

    #from transformers import AutoModelForCausalLM

    #model_path = "./model/Qwen2-1.5B-Instruct"
    #model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    #print(model)

