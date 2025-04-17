# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge
import jieba

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from config import model_config, training_config
from evaluate_model import load_model_and_tokenizer, generate_template

def calculate_single_metrics(prediction, reference):
    # ROUGE
    tokenized_pred_rouge = " ".join(jieba.cut(prediction))
    tokenized_ref_rouge = " ".join(jieba.cut(reference))

    # BLEU
    tokenized_prediction_bleu = list(prediction)
    tokenized_reference_bleu = list(reference)

    # 计算ROUGE分数
    rouge = Rouge()
    rouge_scores = rouge.get_scores(tokenized_pred_rouge, tokenized_ref_rouge)[0]

    # 计算BLEU分数
    smoother = SmoothingFunction().method1
    bleu_score = sentence_bleu(
        [tokenized_reference_bleu],
        tokenized_prediction_bleu,
        smoothing_function=smoother
    )

    metrics = {
        "bleu": bleu_score * 100,
        "rouge-1": rouge_scores["rouge-1"]["f"] * 100,
        "rouge-2": rouge_scores["rouge-2"]["f"] * 100,
        "rouge-l": rouge_scores["rouge-l"]["f"] * 100
    }

    return metrics

def calculate_and_sort_metrics():
    # 加载验证数据
    with open(training_config.val_json_path, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]
    
    results = []
    
    # 先处理原始模型
    print("正在加载原始模型...")
    base_model, tokenizer = load_model_and_tokenizer(use_lora=False)
    
    print("原始模型生成中...")
    base_templates = []
    for item in tqdm(val_data):
        chief_complaint = item["chief_complaint"]
        department = item["department"]
        base_template = generate_template(base_model, tokenizer, chief_complaint, department)
        base_templates.append(base_template)
    
    # 释放内存
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 再处理微调模型
    print("正在加载微调模型...")
    tuned_model, tokenizer = load_model_and_tokenizer(use_lora=True)
    
    print("微调模型生成中...")
    for i, item in enumerate(tqdm(val_data)):
        chief_complaint = item["chief_complaint"]
        template = item["template"]
        department = item["department"]
        
        # 使用已保存的原始模型结果
        base_template = base_templates[i]
        
        # 生成微调模型模板
        tuned_template = generate_template(tuned_model, tokenizer, chief_complaint, department)
        
        # 计算指标
        base_metrics = calculate_single_metrics(base_template, template)
        tuned_metrics = calculate_single_metrics(tuned_template, template)
        
        # 计算平均分数
        avg_score = (tuned_metrics["bleu"] + tuned_metrics["rouge-1"] +
                    tuned_metrics["rouge-2"] + tuned_metrics["rouge-l"]) / 4
        
        results.append({
            "chief_complaint": chief_complaint,
            "department": department,
            "reference_template": template,
            "base_template": base_template,
            "tuned_template": tuned_template,
            "base_metrics": base_metrics,
            "tuned_metrics": tuned_metrics,
            "avg_score": avg_score
        })
    
    # 按平均分数降序排序
    results.sort(key=lambda x: x["avg_score"], reverse=True)
    
    # 保存结果到文件
    with open("sorted_metrics_results.txt", "w", encoding="utf-8") as f:
        for result in results:
            f.write(f"主诉：{result['chief_complaint']}\n")
            f.write(f"科室：{result['department']}\n")
            f.write(f"参考模板：{result['reference_template']}\n")
            f.write(f"原始模型生成模板：{result['base_template']}\n")
            f.write(f"微调模型生成模板：{result['tuned_template']}\n")
            f.write(f"原始模型指标：\n")
            f.write(f"  BLEU: {result['base_metrics']['bleu']:.2f}\n")
            f.write(f"  ROUGE-1: {result['base_metrics']['rouge-1']:.2f}\n")
            f.write(f"  ROUGE-2: {result['base_metrics']['rouge-2']:.2f}\n")
            f.write(f"  ROUGE-L: {result['base_metrics']['rouge-l']:.2f}\n")
            f.write(f"微调模型指标：\n")
            f.write(f"  BLEU: {result['tuned_metrics']['bleu']:.2f}\n")
            f.write(f"  ROUGE-1: {result['tuned_metrics']['rouge-1']:.2f}\n")
            f.write(f"  ROUGE-2: {result['tuned_metrics']['rouge-2']:.2f}\n")
            f.write(f"  ROUGE-L: {result['tuned_metrics']['rouge-l']:.2f}\n")
            f.write(f"平均分数：{result['avg_score']:.2f}\n")
            f.write("-" * 100 + "\n")
    
    print("结果已保存到 sorted_metrics_results.txt")

if __name__ == "__main__":
    calculate_and_sort_metrics() 