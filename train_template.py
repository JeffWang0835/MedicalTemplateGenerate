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
from config import model_config, training_config, lora_config

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_config.model_name, trust_remote_code=model_config.trust_remote_code)

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
    # 设备
    device = torch.device(model_config.device if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name, trust_remote_code=model_config.trust_remote_code)
    
    # 设置LoRA配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_config.target_modules,
        inference_mode=False,
        r=lora_config.lora_rank,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout
    )
    model = get_peft_model(model, peft_config)
    model.is_parallelizable = True
    model.model_parallel = True
    model.print_trainable_parameters()

    # 加载训练数据
    print("Start Load Train Data...")
    train_params = {
        "batch_size": training_config.batch_size,
        "shuffle": True,
        "num_workers": training_config.num_workers,
    }
    training_set = TemplateDataset(
        training_config.train_json_path, 
        tokenizer, 
        training_config.max_source_length, 
        training_config.max_target_length
    )
    training_loader = DataLoader(training_set, collate_fn=custom_collate_fn, **train_params)

    # 加载验证数据
    print("Start Load Validation Data...")
    val_params = {
        "batch_size": training_config.batch_size,
        "shuffle": False,
        "num_workers": training_config.num_workers,
    }
    val_set = TemplateDataset(
        training_config.val_json_path, 
        tokenizer, 
        training_config.max_source_length, 
        training_config.max_target_length
    )
    val_loader = DataLoader(val_set, collate_fn=custom_collate_fn, **val_params)

    # 日志记录
    writer = SummaryWriter("logs")
    
    # 优化器
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=training_config.learning_rate)
    model = model.to(device)

    # 开始训练
    print("Start Training...")
    train_model(
        model=model,
        train_loader=training_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        device=device,
        num_epochs=training_config.epochs,
        model_output_dir=model_config.lora_dir,
        writer=writer
    )
    writer.close()


if __name__ == '__main__':
    main()

    #from transformers import AutoModelForCausalLM

    #model_path = "./model/Qwen2-1.5B-Instruct"
    #model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    #print(model)

