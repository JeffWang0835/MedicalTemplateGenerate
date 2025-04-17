# -*- coding: utf-8 -*-
import pandas as pd
import re

def parse_metrics_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分割每个样本
    samples = content.split('-' * 100)
    
    data = []
    for sample in samples:
        if not sample.strip():
            continue
            
        # 使用正则表达式提取信息
        chief_complaint = re.search(r'主诉：(.*?)(?=\n)', sample)
        department = re.search(r'科室：(.*?)(?=\n)', sample)
        reference_template = re.search(r'参考模板：(.*?)(?=\n)', sample)
        base_template = re.search(r'原始模型生成模板：(.*?)(?=\n)', sample)
        tuned_template = re.search(r'微调模型生成模板：(.*?)(?=\n)', sample)
        
        # 提取原始模型指标
        base_metrics_section = re.search(r'原始模型指标：(.*?)(?=微调模型指标：)', sample, re.DOTALL)
        if base_metrics_section:
            base_metrics = {
                'bleu': re.search(r'BLEU: (.*?)(?=\n)', base_metrics_section.group(1)),
                'rouge1': re.search(r'ROUGE-1: (.*?)(?=\n)', base_metrics_section.group(1)),
                'rouge2': re.search(r'ROUGE-2: (.*?)(?=\n)', base_metrics_section.group(1)),
                'rougel': re.search(r'ROUGE-L: (.*?)(?=\n)', base_metrics_section.group(1))
            }
        else:
            base_metrics = {'bleu': None, 'rouge1': None, 'rouge2': None, 'rougel': None}
        
        # 提取微调模型指标
        tuned_metrics_section = re.search(r'微调模型指标：(.*?)(?=平均分数：)', sample, re.DOTALL)
        if tuned_metrics_section:
            tuned_metrics = {
                'bleu': re.search(r'BLEU: (.*?)(?=\n)', tuned_metrics_section.group(1)),
                'rouge1': re.search(r'ROUGE-1: (.*?)(?=\n)', tuned_metrics_section.group(1)),
                'rouge2': re.search(r'ROUGE-2: (.*?)(?=\n)', tuned_metrics_section.group(1)),
                'rougel': re.search(r'ROUGE-L: (.*?)(?=\n)', tuned_metrics_section.group(1))
            }
        else:
            tuned_metrics = {'bleu': None, 'rouge1': None, 'rouge2': None, 'rougel': None}
        
        avg_score = re.search(r'平均分数：(.*?)(?=\n)', sample)
        
        if all([chief_complaint, department, reference_template, base_template, tuned_template, avg_score]):
            data.append({
                '主诉': chief_complaint.group(1),
                '科室': department.group(1),
                '参考模板': reference_template.group(1),
                '原始模型生成模板': base_template.group(1),
                '微调模型生成模板': tuned_template.group(1),
                '原始模型BLEU': float(base_metrics['bleu'].group(1)) if base_metrics['bleu'] else None,
                '原始模型ROUGE-1': float(base_metrics['rouge1'].group(1)) if base_metrics['rouge1'] else None,
                '原始模型ROUGE-2': float(base_metrics['rouge2'].group(1)) if base_metrics['rouge2'] else None,
                '原始模型ROUGE-L': float(base_metrics['rougel'].group(1)) if base_metrics['rougel'] else None,
                '微调模型BLEU': float(tuned_metrics['bleu'].group(1)) if tuned_metrics['bleu'] else None,
                '微调模型ROUGE-1': float(tuned_metrics['rouge1'].group(1)) if tuned_metrics['rouge1'] else None,
                '微调模型ROUGE-2': float(tuned_metrics['rouge2'].group(1)) if tuned_metrics['rouge2'] else None,
                '微调模型ROUGE-L': float(tuned_metrics['rougel'].group(1)) if tuned_metrics['rougel'] else None,
                '平均分数': float(avg_score.group(1))
            })
    
    return data

def generate_excel():
    # 解析文本文件
    data = parse_metrics_file('sorted_metrics_results.txt')
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 设置列宽
    writer = pd.ExcelWriter('metrics_results.xlsx', engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='评估结果')
    
    # 获取工作表对象
    worksheet = writer.sheets['评估结果']
    
    # 设置列宽
    worksheet.set_column('A:A', 30)  # 主诉
    worksheet.set_column('B:B', 15)  # 科室
    worksheet.set_column('C:E', 50)  # 模板
    worksheet.set_column('F:N', 15)  # 指标
    
    # 保存Excel文件
    writer.close()
    print("Excel文件已生成：metrics_results.xlsx")

if __name__ == "__main__":
    generate_excel()