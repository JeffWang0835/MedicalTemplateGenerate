# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig as HFGenerationConfig
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_chinese import Rouge
from peft import PeftModel
from config import system_prompt
import jieba

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from config import model_config, generation_config, training_config
from template_dataset import TemplateDataset

def load_model_and_tokenizer(use_lora=False):
    # use_lora: 是否使用LoRA微调后的模型
    print(f"正在加载{'LoRA微调后的' if use_lora else '原始'}模型...")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get_model_path(),
        trust_remote_code=model_config.trust_remote_code
    )
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_config.get_model_path(),
        trust_remote_code=model_config.trust_remote_code,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # 如果使用LoRA，加载微调后的模型
    if use_lora:
        model = PeftModel.from_pretrained(model, model_config.get_lora_dir())
    
    model.eval()
    return model, tokenizer

def generate_template(model, tokenizer, chief_complaint, department):
    # 构建输入提示
    messages = [
        {"role": "system", "content": system_prompt.content + (f"\n科室：{department}" if department else "")},
        {"role": "user", "content": chief_complaint}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # 设置生成参数
    gen_config = HFGenerationConfig(
        max_new_tokens=generation_config.max_new_tokens,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        repetition_penalty=generation_config.repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    
    # 生成输出
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=gen_config
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取助手回复部分
    assistant_start = generated_text.find("assistant")
    if assistant_start != -1:
        # 找到assistant标记后的第一个换行符
        newline_pos = generated_text.find("\n", assistant_start)
        if newline_pos != -1:
            generated_text = generated_text[newline_pos:].strip()
    
    # 清理输出文本
    generated_text = generated_text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    generated_text = generated_text.strip()
    
    return generated_text

def compute_metrics(predictions, references):
    # ROUGE
    tokenized_preds_rouge = [" ".join(jieba.cut(pred)) for pred in predictions]
    tokenized_refs_rouge = [" ".join(jieba.cut(ref)) for ref in references]

    # BLEU
    tokenized_predictions_bleu = [[list(pred)] for pred in predictions]
    tokenized_references_bleu = [list(ref) for ref in references]

    # 计算ROUGE分数
    rouge = Rouge()
    rouge_scores = rouge.get_scores(tokenized_preds_rouge, tokenized_refs_rouge, avg=True)

    # 计算BLEU分数
    smoother = SmoothingFunction().method1
    bleu_score = corpus_bleu(
        [[r] for r in tokenized_references_bleu],  # 这里使用BLEU专用的分词结果
        [p[0] for p in tokenized_predictions_bleu],
        smoothing_function=smoother
    )

    metrics = {
        "bleu": bleu_score * 100,
        "rouge-1": rouge_scores["rouge-1"]["f"] * 100,
        "rouge-2": rouge_scores["rouge-2"]["f"] * 100,
        "rouge-l": rouge_scores["rouge-l"]["f"] * 100
    }

    return metrics

def evaluate():
    """评估原始模型和微调后模型的性能"""
    # 加载验证数据
    with open(training_config.val_json_path, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]
    
    # 加载tokenizer (共用同一个tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get_model_path(),
        trust_remote_code=model_config.trust_remote_code
    )
    
    results = {}
    
    # 评估原始模型
    base_model, _ = load_model_and_tokenizer(use_lora=False)
    base_predictions = []
    references = []
    
    print("正在评估原始模型...")
    for item in tqdm(val_data):
        chief_complaint = item["chief_complaint"]
        template = item["template"]
        department = item["department"]
        
        generated_template = generate_template(base_model, tokenizer, chief_complaint, department)
        base_predictions.append(generated_template)
        references.append(template)
        
        # 打印第一个样本的输入输出，用于调试
        if len(base_predictions) == 1:
            print("\n第一个样本的输入输出示例:")
            print(f"主诉: {chief_complaint}")
            print(f"科室: {department}")
            print(f"参考模板: {template}")
            print(f"生成模板: {generated_template}")
    
    base_metrics = compute_metrics(base_predictions, references)
    results["base_model"] = base_metrics
    
    # 释放内存
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 评估微调后的模型
    tuned_model, _ = load_model_and_tokenizer(use_lora=True)
    tuned_predictions = []
    
    print("正在评估微调后模型...")
    for item in tqdm(val_data):
        chief_complaint = item["chief_complaint"]
        department = item["department"]
        
        generated_template = generate_template(tuned_model, tokenizer, chief_complaint, department)
        tuned_predictions.append(generated_template)
        
        # 打印第一个样本的输入输出，用于调试
        if len(tuned_predictions) == 1:
            print("\n第一个样本的输入输出示例:")
            print(f"主诉: {chief_complaint}")
            print(f"科室: {department}")
            print(f"参考模板: {references[0]}")
            print(f"生成模板: {generated_template}")
    
    tuned_metrics = compute_metrics(tuned_predictions, references)
    results["tuned_model"] = tuned_metrics
    
    # 保存结果到文件
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 打印结果
    print("\n评估结果:")
    print("原始模型:")
    for metric, value in results["base_model"].items():
        print(f"  {metric}: {value:.2f}")
    
    print("\n微调后模型:")
    for metric, value in results["tuned_model"].items():
        print(f"  {metric}: {value:.2f}")
    
    # 计算性能提升
    print("\n性能提升:")
    for metric in results["base_model"].keys():
        improvement = results["tuned_model"][metric] - results["base_model"][metric]
        print(f"  {metric}: {improvement:+.2f}")

if __name__ == "__main__":
    evaluate() 